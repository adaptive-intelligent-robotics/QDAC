import torch
from ribs.archives import ArchiveBase

v = torch.ones(1, device='cuda')  # init torch cuda before jax

from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np


from utils.analysis_repertoire import AnalysisRepertoire
from baselines.PPGA.envs.brax_custom.brax_env import make_vec_env_brax_ppga
from baselines.PPGA.models.actor_critic import Actor
from baselines.PPGA.models.vectorized import VectorizedActor
from baselines.PPGA.utils.archive_utils import evaluate


def find_unique_folder(directory_path):
    # Create a Path object for the directory
    directory = Path(directory_path)

    # Filter out files, leaving only directories
    folders = [item for item in directory.iterdir() if item.is_dir()]

    # Check if there's exactly one folder and return its path if so
    if len(folders) == 1:
        return folders[0]
    else:
        raise ValueError("Expected exactly one folder in the directory")


def load_repertoire_ppga(folder_load: str):
    from pathlib import Path

    folder_load = Path(folder_load)
    folder_load = folder_load / "experiments"
    folder_load = find_unique_folder(folder_load)
    folder_load = find_unique_folder(folder_load)
    folder_load = folder_load / "checkpoints"

    # sort the folders by name
    folders = sorted(folder_load.iterdir(), key=lambda x: int(x.name.split('_')[1]))
    folder_load = folders[-1]

    # get file with name archive_df_00000560.pkl and scheduler_00000560.pkl
    archive_path = list(folder_load.glob('archive_df_*.pkl'))[0]
    scheduler_path = list(folder_load.glob('scheduler_*.pkl'))[0]

    # now lets load in a saved archive dataframe and scheduler
    # change this to be your own checkpoint path
    with open(str(archive_path), 'rb') as f:
        archive_df = pickle.load(f)
    with open(str(scheduler_path), 'rb') as f:
        scheduler = pickle.load(f)

    archive = scheduler.archive

    return archive

def get_env(hydra_config, cfg):
    if hydra_config.feat == "angle_notrigo":
        feat = "angle"
    else:
        feat = hydra_config.feat
    vec_env = make_vec_env_brax_ppga(task_name=hydra_config.task, feat_name=feat, batch_size=cfg.env_batch_size,
                                     seed=cfg.seed, backend=cfg.backend, clip_obs_rew=cfg.clip_obs_rew, episode_length=hydra_config.algo.episode_length)
    return vec_env

def reevaluate_ppga_archive(cfg, hydra_config, original_archive: ArchiveBase, vec_env, solution_batch_size, num_reevals, centroids, transform_descs_fn=None):
    num_sols = len(centroids)

    obs_shape, action_shape = vec_env.single_observation_space.shape, vec_env.single_action_space.shape
    device = torch.device('cuda')


    agents = []
    descriptors = []
    for elite in original_archive:
        desc = elite.measures
        agent = Actor(obs_shape=obs_shape[0], action_shape=action_shape, normalize_obs=cfg.normalize_obs).deserialize(elite.solution).to(device)
        if cfg.normalize_obs:
            agent.obs_normalizer.load_state_dict(elite.metadata['obs_normalizer'])
        agents.append(agent)
        descriptors.append(desc)
    agents = np.array(agents)
    repertoire_descriptors = jnp.asarray(descriptors)
    if transform_descs_fn is not None:
        repertoire_descriptors = transform_descs_fn(repertoire_descriptors)

    print("DESC SHAPE", repertoire_descriptors.shape)


    distances_to_centroids = jax.vmap(lambda x, y: jnp.linalg.norm(x - y, axis=-1), in_axes=(None, 0))(repertoire_descriptors,
                                                                                                       centroids)
    indices = jax.vmap(jnp.argmin)(distances_to_centroids)

    # select the best agent for each centroid
    all_agents = [agents[index] for index in indices]
    agents = np.asarray(all_agents)

    all_objs, all_measures = [], []
    for i in range(0, num_sols, solution_batch_size):  # TODO: modify solution batch size and/or num_sols
        agent_batch = agents[i: i + solution_batch_size]
        if cfg.env_batch_size % len(agent_batch) != 0 and len(original_archive) % solution_batch_size != 0:
            print(f'[WARNING] Changing env batch size from {cfg.env_batch_size} to {len(agent_batch) * num_reevals}')
            del vec_env
            cfg.env_batch_size = len(agent_batch) * num_reevals
            vec_env = get_env(hydra_config, cfg)
        print(f'Evaluating solution batch {i}/{num_sols}')
        vec_inference = VectorizedActor(agent_batch, Actor, obs_shape=obs_shape, action_shape=action_shape, normalize_obs=cfg.normalize_obs).to(device)
        objs, measures, _ = evaluate(vec_inference, vec_env, cfg.num_dims, normalize_obs=cfg.normalize_obs, compute_avg=False)
        all_objs.append(objs)
        all_measures.append(measures)

    for index in range(len(all_objs)):
        print(np.mean(all_objs[index], axis=1), np.mean(all_measures[index], axis=1))
        print(all_objs[index].shape)

    all_objs = np.concatenate(all_objs, axis=0)
    all_measures = np.concatenate(all_measures, axis=0)

    print(f'{all_objs.shape=}, {all_measures.shape=}')
    assert all_objs.shape == (num_sols, num_reevals), f'{all_objs.shape=}, {num_sols=}, {num_reevals=}'
    assert all_measures.shape == (num_sols, num_reevals, cfg.num_dims), f'{all_measures.shape=}, {num_sols=}, {num_reevals=}'

    # create a new archive
    all_objs = jnp.asarray(all_objs)
    all_measures = jnp.asarray(all_measures)

    analysis_repertoire = AnalysisRepertoire.create(fitnesses=all_objs, descriptors=all_measures, centroids=centroids)

    return analysis_repertoire
