import pandas
import torch
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from ribs.archives import CVTArchive, GridArchive
from ribs.visualize import cvt_archive_heatmap, grid_archive_heatmap
from typing import Optional

from baselines.PPGA.algorithm.config_ppga import DotDict
from baselines.PPGA.models.actor_critic import Actor, PGAMEActor
from baselines.PPGA.models.vectorized import VectorizedActor
from baselines.PPGA.envs.brax_custom.brax_env import make_vec_env_brax_ppga
from baselines.PPGA.envs.brax_custom import reward_offset


from baselines.qdax.core.neuroevolution.networks.networks import MLP
from baselines.qdax import environments
from baselines.qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from jax._src.flatten_util import ravel_pytree


def save_heatmap(archive, heatmap_path, emitter_loc: Optional[tuple[float, ...]] = None,
                 forces: Optional[tuple[float, ...]] = None):
    """Saves a heatmap of the archive to the given path.
    Args:
        archive (GridArchive or CVTArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
        emitter_loc: Where the emitter is in the archive. Determined by the measures of the mean solution point
        force: the direction that the emitter is being pushed towards. Determined by the gradient coefficients of
        the mean solution point
    """
    if isinstance(archive, GridArchive):
        plt.figure(figsize=(8, 6))
        grid_archive_heatmap(archive, emitter_loc=emitter_loc, forces=forces)
        plt.tight_layout()
        plt.savefig(heatmap_path)
    elif isinstance(archive, CVTArchive):
        plt.figure(figsize=(16, 12))
        cvt_archive_heatmap(archive)
        plt.tight_layout()
        plt.savefig(heatmap_path)
    plt.close('all')


def load_scheduler_from_checkpoint(scheduler_path, seed, device):
    assert os.path.exists(scheduler_path), f'Error! {scheduler_path=} does not exist'
    with open(scheduler_path, 'rb') as f:
        scheduler = pickle.load(f)
    # reinstantiate the pytorch generator with the correct seed
    scheduler.emitters[0].opt.problem._generator = torch.Generator(device=device)
    scheduler.emitters[0].opt.problem._generator.manual_seed(seed)
    return scheduler


def load_archive(archive_path):
    assert os.path.exists(archive_path), f'Error! {archive_path=} does not exist'
    with open(archive_path, 'rb') as f:
        archive = pickle.load(f)
    return archive


def evaluate(vec_agent, vec_env, num_dims, use_action_means=True, normalize_obs=False, compute_avg=True):
    """
    Evaluate all agents for one episode
    :param vec_agent: Vectorized agents for vectorized inference
    :returns: Sum rewards and measures for all agents
    """

    total_reward = np.zeros(vec_env.num_envs)
    traj_length = 0
    num_steps = 1000
    device = torch.device('cuda')

    obs = vec_env.reset()
    obs = obs.to(device)
    dones = torch.BoolTensor([False for _ in range(vec_env.num_envs)])
    all_dones = torch.zeros((num_steps, vec_env.num_envs)).to(device)
    measures_acc = torch.zeros((num_steps, vec_env.num_envs, num_dims)).to(device)
    measures = torch.zeros((vec_env.num_envs, num_dims)).to(device)

    if normalize_obs:
        repeats = vec_env.num_envs // vec_agent.num_models
        obs_mean = [normalizer.obs_rms.mean for normalizer in vec_agent.obs_normalizers]
        obs_mean = torch.vstack(obs_mean).to(device)
        obs_mean = torch.repeat_interleave(obs_mean, dim=0, repeats=repeats)
        obs_var = [normalizer.obs_rms.var for normalizer in vec_agent.obs_normalizers]
        obs_var = torch.vstack(obs_var).to(device)
        obs_var = torch.repeat_interleave(obs_var, dim=0, repeats=repeats)

    while not torch.all(dones):
        with torch.no_grad():
            if normalize_obs:
                obs = (obs - obs_mean) / torch.sqrt(obs_var + 1e-8)
            if use_action_means:
                acts = vec_agent(obs)
            else:
                acts, _, _ = vec_agent.get_action(obs)
            acts = acts.to(torch.float32)
            obs, rew, next_dones, infos = vec_env.step(acts)
            measures_acc[traj_length] = infos['measures']
            obs = obs.to(device)
            total_reward += rew.detach().cpu().numpy() * ~dones.cpu().numpy()
            dones = torch.logical_or(dones, next_dones.cpu())
            all_dones[traj_length] = dones.long().clone()
            traj_length += 1

    # the first done in each env is where that trajectory ends
    traj_lengths = torch.argmax(all_dones, dim=0) + 1
    avg_traj_lengths = traj_lengths.to(torch.float32).reshape((vec_agent.num_models, vec_env.num_envs // vec_agent.num_models)).mean(dim=1).cpu().numpy()
    # TODO: figure out how to vectorize this
    for i in range(vec_env.num_envs):
        measures[i] = measures_acc[:traj_lengths[i], i].sum(dim=0) / traj_lengths[i]
    if compute_avg:
        measures = measures.reshape(vec_agent.num_models, vec_env.num_envs // vec_agent.num_models, -1).mean(dim=1)
    else:
        measures = measures.reshape(vec_agent.num_models, vec_env.num_envs // vec_agent.num_models, -1)

    metadata = np.array([{'traj_length': t} for t in avg_traj_lengths])
    total_reward = total_reward.reshape((vec_agent.num_models, vec_env.num_envs // vec_agent.num_models))
    if compute_avg:
        total_reward = total_reward.mean(axis=1)
        return total_reward.reshape(-1, ), measures.reshape(-1, num_dims).detach().cpu().numpy(), metadata
    else:
        return total_reward, measures.detach().cpu().numpy(), metadata



def pgame_repertoire_to_pyribs_archive(cp_path, env_cfg, save_path=None):
    # define the environment
    env_name = env_cfg.env_name
    seed = env_cfg.seed
    episode_length = env_cfg.seed
    env = environments.create(env_name, episode_length=episode_length)
    env_batch_size = 1

    # define the MLP architecture
    policy_layer_sizes = (128, 128) + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # setup initial variables and rng keys
    random_key = jax.random.PRNGKey(seed)
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=env_batch_size)
    fake_batch = jnp.zeros(shape=(env_batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    def load_archive(random_key):
        random_key, subkey = jax.random.split(random_key)
        fake_batch = jnp.zeros(shape=(env.observation_size,))
        fake_params = policy_network.init(subkey, fake_batch)

        _, reconstruction_fn = ravel_pytree(fake_params)
        repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=cp_path)
        return repertoire

    repertoire = load_archive(random_key)
    active_inds = jnp.where(repertoire.fitnesses != -jnp.inf)[0]

    flax_params = repertoire.genotypes['params']

    def flax_to_torch_model(model_ind):
        pytorch_model = PGAMEActor(obs_shape=env.observation_size, action_shape=(env.action_size,))
        pytorch_params = dict(pytorch_model.named_parameters())
        for i in range(len(flax_params)):
            pytorch_params[f'actor_mean.{2*i}.weight'].data = torch.from_numpy(flax_params[f'Dense_{i}']['kernel'][model_idx]._value.T.copy())
            pytorch_params[f'actor_mean.{2*i}.bias'].data = torch.from_numpy(flax_params[f'Dense_{i}']['bias'][model_idx]._value.T.copy())
        return pytorch_model.serialize()

    solution_batch = []
    for model_idx in active_inds:
        solution_batch.append(flax_to_torch_model(model_idx))
        print(f'Finished processing model {model_idx}')
    solution_batch = np.array(solution_batch)
    obj_batch = np.array(repertoire.fitnesses[active_inds])
    measures_batch = np.array(repertoire.descriptors[active_inds])

    archive_dims = [env_cfg.grid_size] * env_cfg.num_dims
    num_dims = env_cfg.num_dims
    bounds = [(0., 1.0) for _ in range(num_dims)]
    archive = GridArchive(solution_dim=solution_batch.shape[1],
                          dims=archive_dims,
                          ranges=bounds,
                          threshold_min=-np.inf,
                          seed=seed)
    archive.add(solution_batch, obj_batch, measures_batch)

    if save_path is not None:
        archive_fp = os.path.join(save_path, f'{env_name}_original_archive.pkl')
        with open(archive_fp, 'wb') as f:
            pickle.dump(archive, f)

    return archive, solution_batch


def pgame_checkpoint_to_objective_df(cp_path):
    fitness_fp = os.path.join(cp_path, 'fitnesses.npy')
    fitnesses = np.load(fitness_fp)
    fitnesses = fitnesses[np.where(fitnesses != -np.inf)]

    df = pandas.DataFrame(fitnesses, columns=['objective'])
    return df


def evaluate_pga_me_archive(archive_dir):
    '''
    Convert a qdax checkpoint into a ribs archive and evaluate it
    :param checkpoint_dir: directory to find the centroids, descriptors, and gentoypes files
    '''
    archive_path = os.path.join(archive_dir, 'ribs_archive.pkl')
    pgame_archive = load_archive(archive_path)

    env_cfg = DotDict({'env_name': 'walker2d', 'num_dims': 2, 'seed': 1111})
    env_cfg.env_batch_size = len(pgame_archive)
    vec_env = make_vec_env_brax_ppga(env_cfg)
    obs_shape, action_shape = vec_env.single_observation_space.shape, vec_env.single_action_space.shape
    device = torch.device('cuda')

    solutions = pgame_archive.to_numpy()[:, env_cfg.num_dims + 2:]

    agents = [PGAMEActor(obs_shape[0], action_shape).deserialize(solution).to(device) for solution in solutions]
    cfg = DotDict(
        {'normalize_obs': False, 'normalize_rewards': False, 'num_envs': solutions.shape[0], 'num_dims': env_cfg.num_dims})
    vec_agent = VectorizedActor(cfg, agents, PGAMEActor, obs_shape=obs_shape, action_shape=action_shape).to(device)
    objs, measures = evaluate(vec_agent, vec_env, env_cfg.num_dims)

    archive_dims = [100, 100]
    bounds = [(0., 1.0) for _ in range(env_cfg.num_dims)]
    archive = GridArchive(solution_dim=solutions.shape[1],
                          dims=archive_dims,
                          ranges=bounds,
                          threshold_min=0.0,
                          seed=env_cfg.seed)
    archive.add(solutions, objs, measures)

    analysis_dir = os.path.join(archive_dir, 'post_hoc_analysis')
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)
    heatmap_path = os.path.join(analysis_dir, 'pga_me_no_autoreset_heatmap.png')
    save_heatmap(archive, heatmap_path)


def archive_df_to_archive(archive_df: pandas.DataFrame, **kwargs):
    solution_batch = archive_df.filter(regex='solution*').to_numpy()
    measures_batch = archive_df.filter(regex='measure*').to_numpy()
    obj_batch = archive_df.filter(regex='objective').to_numpy().flatten()
    metadata_batch = archive_df.filter(regex='metadata').to_numpy().flatten()
    archive = GridArchive(**kwargs)
    archive.add(solution_batch, obj_batch, measures_batch, metadata_batch)
    return archive


if __name__ == '__main__':
    pass
