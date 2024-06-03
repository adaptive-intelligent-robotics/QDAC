from typing import Tuple, Any
import functools
import os
import time
import pickle

import jax
import jax.numpy as jnp
from flax import serialization

from brax.envs import State as EnvState

from baselines.qdax import environments
from baselines.qdax.baselines.scopa import SCOPA, SCOPAConfig, SCOPATrainingState
from baselines.qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from baselines.qdax.core.neuroevolution.buffers.trajectory_buffer import TrajectoryBuffer
from baselines.qdax.core.neuroevolution.sac_td3_utils import do_iteration_fn, warmstart_buffer
from baselines.qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from baselines.qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from baselines.qdax.utils.metrics import CSVLogger, default_qd_metrics

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import wandb
from utils.env_utils import Config


@hydra.main(version_base="1.2", config_path="configs/", config_name="qdac")
def main(config: Config) -> None:
    wandb.init(
        config=OmegaConf.to_container(config, resolve=True),
        project="QDAC",
        name=config.algo.name,
    )

    os.mkdir("./repertoire/")
    os.mkdir("./actor/")
    os.mkdir("./critic/")
    os.mkdir("./successor_features/")
    os.mkdir("./lagrange/")
    os.mkdir("./alpha/")

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = environments.create(config.task + "_" + config.feat, batch_size=config.algo.env_batch_size, episode_length=config.algo.episode_length, backend=config.algo.backend)
    env_eval = environments.create(config.task + "_" + config.feat, batch_size=config.algo.env_batch_size, episode_length=config.algo.episode_length, backend=config.algo.backend, eval_metrics=True)

    # Init replay buffer
    dummy_transition = QDTransition.init_dummy(
        observation_dim=env.observation_size,
        action_dim=env.action_size,
        descriptor_dim=env.feat_size,
    )
    replay_buffer = TrajectoryBuffer.init(
        buffer_size=config.algo.replay_buffer_size,
        transition=dummy_transition,
        env_batch_size=config.algo.env_batch_size,
        episode_length=config.algo.episode_length,
    )

    # Define config
    scopa_config = SCOPAConfig(
        # SAC
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
        # SCOPA
        delta=config.algo.delta,
        delta_pseudo_huber_loss=config.algo.delta_pseudo_huber_loss,
        lambda_=config.algo.lambda_,
        num_random_skill_batches=config.algo.num_random_skill_batches,
        normalize_feat=config.algo.normalize_feat,
    )

    # Define an instance of SCOPA
    scopa = SCOPA(config=scopa_config, env=env)

    # Init env state
    random_key, random_subkey = jax.random.split(random_key)
    env_state = jax.jit(env.reset)(rng=random_subkey)
    env_state.info["skills"] = jnp.nan * jnp.ones((config.algo.env_batch_size, env.feat_size))

    # Init training state
    random_key, random_subkey = jax.random.split(random_key)
    train_state = scopa.init(random_subkey)

    # Make play_step functions scannable by passing static args beforehand
    play_step = functools.partial(
        scopa.play_step_fn,
        env=env,
        deterministic=False,
    )
    play_eval_step = functools.partial(
        scopa.play_step_fn,
        env=env_eval,
        deterministic=True,
    )
    eval_actor = functools.partial(
        scopa.eval_actor_fn,
        env=env_eval,
        play_step_fn=play_eval_step,
    )

    # Warmstart the buffer
    replay_buffer, _, train_state = warmstart_buffer(
        replay_buffer=replay_buffer,
        training_state=train_state,
        env_state=env_state,
        num_warmstart_steps=config.algo.warmup_steps,
        env_batch_size=config.algo.env_batch_size,
        play_step_fn=play_step,
    )

    # Fix static arguments - prepare for scan
    do_iteration = functools.partial(
        do_iteration_fn,
        env_batch_size=config.algo.env_batch_size,
        grad_updates_per_step=config.algo.grad_updates_per_step,
        play_step_fn=play_step,
        update_fn=scopa.update,
    )

    # Create passive archive
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=config.algo.num_init_cvt_samples,
        num_centroids=config.algo.num_centroids,
        minval=env.feat_space["vector"].low,
        maxval=env.feat_space["vector"].high,
        random_key=random_key,
    )
    repertoire = MapElitesRepertoire.init_default(genotype=train_state.actor_params, centroids=centroids)

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
        carry: Tuple[SCOPATrainingState, EnvState, ReplayBuffer],
        unused_arg: Any,
    ) -> Tuple[Tuple[SCOPATrainingState, EnvState, ReplayBuffer], Any]:
        train_state, env_state, replay_buffer, repertoire = carry

        # Train
        (
            train_state,
            env_state,
            replay_buffer,
            metrics,
        ) = do_iteration(train_state, env_state, replay_buffer)
        metrics = jax.tree_util.tree_map(lambda current_metric: jnp.mean(current_metric), metrics)

        return (train_state, env_state, replay_buffer, repertoire,), metrics

    metrics = dict.fromkeys(["iteration", "qd_score", "coverage", "max_fitness", "mean_fitness", "return_mean", "return_max", "distance_to_skill", "actor_loss", "critic_loss", "successor_features_loss", "lagrange_loss", "alpha_loss", "lagrange_coeffs_mean", "time"], jnp.array([]))
    csv_logger = CSVLogger(
        "./log.csv",
        header=list(metrics.keys())
    )

    # Main loop
    num_loops = int(config.algo.num_iterations / config.algo.log_period)
    for i in range(num_loops):
        start_time = time.time()
        (train_state, env_state, replay_buffer, repertoire), current_metrics = jax.lax.scan(
            _scan_do_iteration,
            (train_state, env_state, replay_buffer, repertoire,),
            (),
            length=config.algo.log_period,
        )
        timelapse = time.time() - start_time

        # Eval
        returns, observed_skills, distance_to_skills, lagrange_coeffs_mean = eval_actor(train_state)
        repertoire = repertoire.add(
            jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), config.algo.env_batch_size, axis=0), train_state.actor_params),
            observed_skills,
            returns,)
        metrics_repertoire = metrics_function(repertoire)
        metrics_repertoire = jax.tree_util.tree_map(lambda metric: jnp.repeat(metric, config.algo.log_period), metrics_repertoire)

        # Metrics
        current_metrics["iteration"] = jnp.arange(1+config.algo.log_period*i, 1+config.algo.log_period*(i+1), dtype=jnp.int32)
        current_metrics["time"] = jnp.repeat(timelapse, config.algo.log_period)
        current_metrics["return_mean"] = jnp.repeat(jnp.mean(returns), config.algo.log_period)
        current_metrics["return_max"] = jnp.repeat(jnp.max(returns), config.algo.log_period)
        current_metrics["distance_to_skill"] = jnp.repeat(jnp.mean(distance_to_skills), config.algo.log_period)
        current_metrics["lagrange_coeffs_mean"] = jnp.repeat(jnp.mean(lagrange_coeffs_mean), config.algo.log_period)
        current_metrics = current_metrics | metrics_repertoire
        metrics = jax.tree_util.tree_map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)

        # Log
        log_metrics = jax.tree_util.tree_map(lambda metric: metric[-1], metrics)
        csv_logger.log(log_metrics)
        wandb.log(log_metrics)

        # Metrics
        with open("./metrics.pickle", "wb") as metrics_file:
            pickle.dump(metrics, metrics_file)

        # Actor
        state_dict = serialization.to_state_dict(train_state.actor_params)
        with open("./actor/actor_{}.pickle".format(int(metrics["iteration"][-1])), "wb") as params_file:
            pickle.dump(state_dict, params_file)

    # Actor
    state_dict = serialization.to_state_dict(train_state.actor_params)
    with open("./actor/actor.pickle", "wb") as params_file:
        pickle.dump(state_dict, params_file)

    # Critic
    state_dict = serialization.to_state_dict(train_state.critic_params)
    with open("./critic/critic.pickle", "wb") as params_file:
        pickle.dump(state_dict, params_file)

    # Successor Features
    state_dict = serialization.to_state_dict(train_state.successor_features_params)
    with open("./successor_features/successor_features.pickle", "wb") as params_file:
        pickle.dump(state_dict, params_file)

    # Lagrange
    state_dict = serialization.to_state_dict(train_state.lagrange_params)
    with open("./lagrange/lagrange.pickle", "wb") as params_file:
        pickle.dump(state_dict, params_file)

    # Alpha
    state_dict = serialization.to_state_dict(train_state.alpha_params)
    with open("./alpha/alpha.pickle", "wb") as params_file:
        pickle.dump(state_dict, params_file)

    # Repertoire
    repertoire.save(path="./repertoire/")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    main()
