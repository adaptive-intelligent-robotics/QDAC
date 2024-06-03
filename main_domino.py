from typing import Tuple, Any, Callable
import functools
import os
import time
import pickle

import jax
import jax.numpy as jnp
from flax import serialization

from brax.envs import State as EnvState

from baselines.qdax.baselines.domino import DOMINOConfig, DOMINO, DOMINOTrainingState, DOMINOTransition
from baselines.qdax.core.neuroevolution.mdp_utils import TrainingState
from baselines.qdax.environments import create
from baselines.qdax.core.neuroevolution.buffers.buffer import ReplayBuffer, Transition
from baselines.qdax.core.neuroevolution.sac_td3_utils import warmstart_buffer
from baselines.qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from baselines.qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from baselines.qdax.types import Metrics
from baselines.qdax.utils.metrics import CSVLogger, default_qd_metrics

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import wandb
from utils.env_utils import Config


@functools.partial(
    jax.jit,
    static_argnames=(
        "env_batch_size",
        "grad_updates_per_step",
        "play_step_fn",
        "update_fn",
    ),
)
def domino_do_iteration_fn(
    training_state_tree: TrainingState,
    env_state_tree: EnvState,
    replay_buffer_tree: ReplayBuffer,
    env_batch_size: int,
    grad_updates_per_step: float,
    play_step_fn: Callable[
        [EnvState, TrainingState],
        Tuple[
            EnvState,
            TrainingState,
            Transition,
        ],
    ],
    update_fn: Callable[
        [TrainingState, ReplayBuffer],
        Tuple[
            TrainingState,
            ReplayBuffer,
            Metrics,
        ],
    ],
) -> Tuple[TrainingState, EnvState, ReplayBuffer, Metrics]:
    """Performs one environment step (over all env simultaneously) followed by one
    training step. The number of updates is controlled by the parameter
    `grad_updates_per_step` (0 means no update while 1 means `env_batch_size`
    updates). Returns the updated states, the updated buffer and the aggregated
    metrics.
    """

    def _scan_update_fn(
        carry: Tuple[TrainingState, ReplayBuffer], unused_arg: Any
    ) -> Tuple[Tuple[TrainingState, ReplayBuffer], Metrics]:
        training_state, replay_buffer, metrics = update_fn(*carry)
        return (training_state, replay_buffer), metrics

    # play steps in the environment
    env_state_tree, training_state_tree, transitions_tree = jax.vmap(play_step_fn)(env_state_tree, training_state_tree)  # TODO: how to deal with the skill?

    # insert transitions in replay buffer
    replay_buffer_tree = jax.vmap(ReplayBuffer.insert)(replay_buffer_tree, transitions_tree)
    num_updates = 1  # TODO: one update per step?

    (training_state_tree, replay_buffer_tree), metrics = jax.lax.scan(
        _scan_update_fn,
        (training_state_tree, replay_buffer_tree),
        (),
        length=num_updates,
    )

    return training_state_tree, env_state_tree, replay_buffer_tree, metrics


@hydra.main(version_base="1.2", config_path="configs/", config_name="domino")
def main(config: Config) -> None:
    wandb.init(
        config=OmegaConf.to_container(config, resolve=True),
        project="QDAC",
        name=config.algo.name,
    )

    os.mkdir("./repertoire/")
    os.mkdir("./actor/")

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    # batch_size_eval = config.algo.num_skills
    env = create(config.task + "_" + config.feat, batch_size=config.algo.env_batch_size, episode_length=config.algo.episode_length, backend=config.algo.backend)
    env_eval = create(config.task + "_" + config.feat, batch_size=config.algo.env_batch_size, episode_length=config.algo.episode_length, backend=config.algo.backend, eval_metrics=True)

    # Init replay buffer
    dummy_transition = DOMINOTransition.init_dummy(
        observation_dim=env.observation_size,
        action_dim=env.action_size,
        descriptor_dim=env.behavior_descriptor_length,
        num_skills=config.algo.num_skills,
    )

    list_replay_buffers = []
    for _ in range(config.algo.num_skills):
        one_replay_buffer = ReplayBuffer.init(
            buffer_size=config.algo.replay_buffer_size, transition=dummy_transition
        )
        list_replay_buffers.append(one_replay_buffer)
    replay_buffer_tree = jax.tree_map(lambda *x: jnp.stack(x, axis=0), *list_replay_buffers)



    # Define config
    domino_config = DOMINOConfig(
        # SAC config
        batch_size=config.algo.batch_size,
        episode_length=config.algo.episode_length,
        tau=config.algo.soft_tau_update,
        normalize_observations=config.algo.normalize_observations,
        learning_rate=config.algo.learning_rate,
        alpha_init=config.algo.alpha_init,
        discount=config.algo.discount,
        reward_scaling=config.algo.reward_scaling,
        hidden_layer_sizes=config.algo.hidden_layer_sizes,
        fix_alpha=config.algo.fix_alpha,

        # DOMINO config
        skill_type="categorical",
        num_skills=config.algo.num_skills,
        descriptor_full_state=False,

        # Those values are taken from the DOMINO paper for DMControl environments
        optimality_ratio=config.algo.optimality_ratio,  # TO change!
        alpha_d_v_avg=config.algo.alpha_d_v_avg,
        alpha_d_sfs_avg=config.algo.alpha_d_sfs_avg,

        learning_rate_lagrange=config.algo.learning_rate_lagrange,
    )

    # Define an instance of DOMINO
    domino = DOMINO(config=domino_config, action_size=env.action_size)

    # Init env state
    random_key, random_subkey = jax.random.split(random_key)
    random_key_tree = jax.random.split(random_subkey, config.algo.num_skills)
    env_state_tree = jax.vmap(env.reset)(random_key_tree)

    # Init skills
    # env_state.info["skills"] = jax.vmap(domino._sample_z_from_prior)(random_keys)  # TODO

    if config.algo.descriptor_full_state:
        descriptor_size = env.observation_size
    else:
        descriptor_size = env.behavior_descriptor_length

    # Init training state

    list_training_states = []
    for _ in range(config.algo.num_skills):
        random_key, random_subkey = jax.random.split(random_key)
        one_training_state = domino.init(
            random_subkey,
            action_size=env.action_size,
            observation_size=env.observation_size,
            descriptor_size=descriptor_size,
        )
        list_training_states.append(one_training_state)
    training_state_tree = jax.tree_map(lambda *x: jnp.stack(x, axis=0), *list_training_states)
    # training_state = domino.init(
    #     random_subkey,
    #     action_size=env.action_size,
    #     observation_size=env.observation_size,
    #     descriptor_size=descriptor_size,
    # )

    # Make play_step functions scannable by passing static args beforehand
    play_step = functools.partial(
        domino.play_step_fn,
        env=env,
        deterministic=False,
    )

    eval_skills = jnp.eye(config.algo.num_skills)  # TODO
    play_eval_step = functools.partial(
        domino.play_step_fn,
        env=env_eval,
        deterministic=True,
    )

    eval_policy = functools.partial(
        domino.eval_policy_fn,
        env=env_eval,
        play_step_fn=play_eval_step,
    )

    # Warmstart the buffer
    warmstart_buffer_fn = functools.partial(
        warmstart_buffer,
        num_warmstart_steps=config.algo.warmup_steps,
        env_batch_size=config.algo.env_batch_size,
        play_step_fn=play_step,
    )
    replay_buffer_tree, _, training_state_tree = jax.vmap(warmstart_buffer_fn)(replay_buffer_tree, training_state_tree, env_state_tree)

    # Fix static arguments - prepare for scan
    do_iteration = functools.partial(
        domino_do_iteration_fn,
        env_batch_size=config.algo.env_batch_size,
        grad_updates_per_step=config.algo.grad_updates_per_step,
        play_step_fn=play_step,
        update_fn=domino.update,
    )

    # Create passive archive
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=config.algo.num_init_cvt_samples,
        num_centroids=config.algo.num_centroids,
        minval=env.behavior_descriptor_limits[0][0],
        maxval=env.behavior_descriptor_limits[1][0],
        random_key=random_key,
    )

    # Select first policy using tree_map
    policy_params_dummy = jax.tree_map(lambda x: x[0], training_state_tree.policy_params)
    repertoire = MapElitesRepertoire.init_default(genotype=policy_params_dummy, centroids=centroids)

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = 0

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * env.episode_length,
    )

    # Define a function that enables do_iteration to be scanned
    @jax.jit
    def _scan_do_iteration(
        carry: Tuple[DOMINOTrainingState, EnvState, ReplayBuffer, MapElitesRepertoire],
        _,
    ) -> Tuple[Tuple[DOMINOTrainingState, EnvState, ReplayBuffer, MapElitesRepertoire], Any]:
        _training_state, _env_state, _replay_buffer, _repertoire = carry

        # Train
        (
            _training_state,
            _env_state,
            _replay_buffer,
            _metrics,
        ) = do_iteration(_training_state, _env_state, _replay_buffer)
        _metrics = jax.tree_util.tree_map(lambda current_metric: jnp.mean(current_metric), _metrics)

        return (_training_state, _env_state, _replay_buffer, _repertoire,), _metrics

    list_keys_metrics = ["iteration", "qd_score", "coverage", "max_fitness", "mean_fitness", "return", "return_diversity", "actor_loss", "critic_loss", "critic_norm_gradient", "lagrange_loss", "alpha_loss", "time"]
    list_keys_metrics.extend(["return_no_diversity_{}".format(index_fitness) for index_fitness in range(config.algo.num_skills)])
    list_keys_metrics.extend(["return_diversity_{}".format(index_fitness) for index_fitness in range(config.algo.num_skills)])
    list_keys_metrics.extend(["min_avg_sfs_dists_{}".format(index_fitness) for index_fitness in range(config.algo.num_skills)])
    list_keys_metrics.extend(["lagrange_param_{}".format(index_fitness) for index_fitness in range(config.algo.num_skills)])
    list_keys_metrics.extend(["avg_reward_{}".format(index_fitness) for index_fitness in range(config.algo.num_skills)])
    list_keys_metrics.extend(["min_desc_dists_{}".format(index_fitness) for index_fitness in range(config.algo.num_skills)])

    metrics = dict.fromkeys(list_keys_metrics, jnp.array([]))
    csv_logger = CSVLogger(
        "./log.csv",
        header=list(metrics.keys())
    )

    # Main loop
    num_loops = int(config.algo.num_iterations / config.algo.log_period)
    for i in range(num_loops):
        start_time = time.time()
        (training_state_tree, env_state_tree, replay_buffer_tree, repertoire), current_metrics = jax.lax.scan(
            _scan_do_iteration,
            (training_state_tree, env_state_tree, replay_buffer_tree, repertoire,),
            (),
            length=config.algo.log_period,
        )
        timelapse = time.time() - start_time

        # Eval
        all_skills = jnp.eye(config.algo.num_skills)
        (
            _,
            true_returns,
            diversity_returns,
            state_desc,
        ) = jax.vmap(eval_policy, in_axes=(0, 0, None))(training_state_tree, all_skills, training_state_tree.avg_sfs)
        descriptors = jnp.nanmean(state_desc, axis=1)  # In this project, the descriptors are the mean of the state descriptors
        descriptors = jnp.mean(descriptors, axis=1)  # average over batch descriptors obtained by each policy
        true_returns = jnp.mean(true_returns, axis=1)  # average over batch descriptors obtained by each policy
        repertoire = repertoire.add(
            training_state_tree.policy_params,
            descriptors,
            true_returns,)
        metrics_repertoire = metrics_function(repertoire)
        metrics_repertoire["return"] = jnp.mean(true_returns)
        metrics_repertoire["return_diversity"] = jnp.mean(diversity_returns)
        # metrics_repertoire["mean_desc_dists"] = jnp.mean(mean_desc_dists)

        def dist(x, y):
            return jnp.sqrt(jnp.sum((x - y) ** 2))

        v_dist = jax.vmap(dist, in_axes=(0, None))
        vv_dist = jax.vmap(v_dist, in_axes=(None, 0))

        def min_dist(X):
            dist_matrix = vv_dist(X, X)
            dist_matrix = dist_matrix.at[jnp.eye(X.shape[0]).astype(jnp.bool_)].set(jnp.inf)
            return jnp.min(dist_matrix, axis=-1)

        min_avg_sfs_dists = min_dist(training_state_tree.avg_sfs)
        min_desc_dists = min_dist(descriptors)

        for index_fitness, (fitness, fitness_diversity, lagrange_param, avg_rewards, min_avg_sfs_dist, min_desc_dist) in enumerate(zip(true_returns,
                                                                                                                diversity_returns,
                                                                                                                training_state_tree.lagrange_params["params"],
                                                                                                                training_state_tree.avg_values,
                                                                                                                min_avg_sfs_dists,
                                                                                                                min_desc_dists,
                                                                                                                )):
            metrics_repertoire["return_no_diversity_{}".format(index_fitness)] = fitness
            metrics_repertoire["return_diversity_{}".format(index_fitness)] = fitness_diversity
            metrics_repertoire["lagrange_param_{}".format(index_fitness)] = lagrange_param
            metrics_repertoire["avg_reward_{}".format(index_fitness)] = avg_rewards
            metrics_repertoire["min_avg_sfs_dists_{}".format(index_fitness)] = min_avg_sfs_dist
            metrics_repertoire["min_desc_dists_{}".format(index_fitness)] = min_desc_dist


        metrics_repertoire = jax.tree_util.tree_map(lambda metric: jnp.repeat(metric, config.algo.log_period), metrics_repertoire)

        # Metrics
        # current_metrics = jax.tree_map(lambda metric: jnp.mean(metric, axis=1), current_metrics)  # Averaging over all policies.
        current_metrics["iteration"] = jnp.arange(1+config.algo.log_period*i, 1+config.algo.log_period*(i+1), dtype=jnp.int32)
        current_metrics["time"] = jnp.repeat(timelapse, config.algo.log_period)

        # Use tree_map to print the shapes of current metrics
        print("current_metrics shapes", jax.tree_map(lambda x: x.shape, current_metrics))
        current_metrics = {**current_metrics, **metrics_repertoire}
        metrics = jax.tree_util.tree_map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)

        # Log
        log_metrics = jax.tree_util.tree_map(lambda metric: metric[-1], metrics)
        csv_logger.log(log_metrics)
        wandb.log(log_metrics)

        # Metrics
        with open("./metrics.pickle", "wb") as metrics_file:
            pickle.dump(metrics, metrics_file)

        # Actor
        state_dict = serialization.to_state_dict(training_state_tree.policy_params)
        with open("./actor/actor_{}.pickle".format(int(metrics["iteration"][-1])), "wb") as params_file:
            pickle.dump(state_dict, params_file)

    # Actor
    state_dict = serialization.to_state_dict(training_state_tree.policy_params)
    with open("./actor/actor.pickle", "wb") as params_file:
        pickle.dump(state_dict, params_file)

    # Repertoire
    repertoire.save(path="./repertoire/")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    main()
