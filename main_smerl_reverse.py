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
from baselines.qdax.baselines.diayn_smerl import DIAYNSMERL, DiaynSmerlConfig, DiaynTrainingState
from baselines.qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from baselines.qdax.core.neuroevolution.buffers.trajectory_buffer import TrajectoryBuffer
from baselines.qdax.core.neuroevolution.sac_td3_utils import do_iteration_fn, warmstart_buffer
from baselines.qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from baselines.qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from baselines.qdax.utils.metrics import CSVLogger, default_qd_metrics
from baselines.qdax.utils.plotting import plot_skills_trajectory

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import wandb
from utils.env_utils import Config


@hydra.main(version_base="1.2", config_path="configs/", config_name="smerl_reverse")
def main(config: Config) -> None:
    wandb.init(
        config=OmegaConf.to_container(config, resolve=True),
        project="QDAC",
        name=config.algo.name,
    )

    os.mkdir("./repertoire/")
    os.mkdir("./actor/")
    os.mkdir("./discriminator/")

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = environments.create(config.task + "_" + config.feat, batch_size=config.algo.env_batch_size, episode_length=config.algo.episode_length, backend=config.algo.backend)
    env_eval = environments.create(config.task + "_" + config.feat, batch_size=config.algo.env_batch_size, episode_length=config.algo.episode_length, backend=config.algo.backend, eval_metrics=True)

    # Init replay buffer
    dummy_transition = QDTransition.init_dummy(
        observation_dim=env.observation_size + config.algo.num_skills,
        action_dim=env.action_size,
        descriptor_dim=env.behavior_descriptor_length,
    )
    replay_buffer = TrajectoryBuffer.init(
        buffer_size=config.algo.replay_buffer_size,
        transition=dummy_transition,
        env_batch_size=config.algo.env_batch_size,
        episode_length=config.algo.episode_length,
    )

    # Define config
    smerl_config = DiaynSmerlConfig(
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
        # DIAYN config
        skill_type=config.algo.skill_type,
        num_skills=config.algo.num_skills,
        descriptor_full_state=config.algo.descriptor_full_state,
        extrinsic_reward=False,
        beta=1.,
        # SMERL
        reverse=True,
        diversity_reward_scale=config.algo.diversity_reward_scale,
        smerl_target=config.algo.smerl_target,
        smerl_margin=config.algo.smerl_margin,
    )

    # Define an instance of DIAYN
    smerl = DIAYNSMERL(config=smerl_config, action_size=env.action_size)

    # Init env state
    random_key, random_subkey = jax.random.split(random_key)
    env_state = jax.jit(env.reset)(rng=random_subkey)

    # Init skills
    random_keys = jax.random.split(random_key, config.algo.env_batch_size+1)
    random_keys, random_key = random_keys[:-1], random_keys[-1]
    env_state.info["skills"] = jax.vmap(smerl._sample_z_from_prior)(random_keys)

    if config.algo.descriptor_full_state:
        descriptor_size = env.observation_size
    else:
        descriptor_size = env.behavior_descriptor_length

    # Init training state
    random_key, random_subkey = jax.random.split(random_key)
    training_state = smerl.init(
        random_subkey,
        action_size=env.action_size,
        observation_size=env.observation_size,
        descriptor_size=descriptor_size,
    )

    # Make play_step functions scannable by passing static args beforehand
    play_step = functools.partial(
        smerl.play_step_fn,
        env=env,
        deterministic=False,
    )
    play_eval_step = functools.partial(
        smerl.play_step_fn,
        env=env_eval,
        deterministic=True,
    )
    eval_policy = functools.partial(
        smerl.eval_policy_fn,
        env=env_eval,
        play_step_fn=play_eval_step,
    )

    # Warmstart the buffer
    replay_buffer, _, training_state = warmstart_buffer(
        replay_buffer=replay_buffer,
        training_state=training_state,
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
        update_fn=smerl.update,
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
    repertoire = MapElitesRepertoire.init_default(genotype=training_state.policy_params, centroids=centroids)

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
        carry: Tuple[DiaynTrainingState, EnvState, ReplayBuffer],
        unused_arg: Any,
    ) -> Tuple[Tuple[DiaynTrainingState, EnvState, ReplayBuffer], Any]:
        training_state, env_state, replay_buffer, repertoire = carry

        # Train
        (
            training_state,
            env_state,
            replay_buffer,
            metrics,
        ) = do_iteration(training_state, env_state, replay_buffer)
        metrics = jax.tree_util.tree_map(lambda current_metric: jnp.mean(current_metric), metrics)

        return (training_state, env_state, replay_buffer, repertoire,), metrics

    metrics = dict.fromkeys(["iteration", "qd_score", "coverage", "max_fitness", "mean_fitness", "mean_fitness_diversity", "actor_loss", "critic_loss", "discriminator_loss", "alpha_loss", "time"], jnp.array([]))
    csv_logger = CSVLogger(
        "./log.csv",
        header=list(metrics.keys())
    )

    # Main loop
    num_loops = int(config.algo.num_iterations / config.algo.log_period)
    for i in range(num_loops):
        start_time = time.time()
        (training_state, env_state, replay_buffer, repertoire), current_metrics = jax.lax.scan(
            _scan_do_iteration,
            (training_state, env_state, replay_buffer, repertoire,),
            (),
            length=config.algo.log_period,
        )
        timelapse = time.time() - start_time

        # Eval
        (
            _,
            fitnesses,
            fitnesses_diversity,
            state_desc,
        ) = eval_policy(training_state)
        descriptors = jnp.nanmean(state_desc, axis=0)  # In this project, the descriptors are the mean of the state descriptors
        repertoire = repertoire.add(
            jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), config.algo.env_batch_size, axis=0), training_state.policy_params),
            descriptors,
            fitnesses,)
        metrics_repertoire = metrics_function(repertoire)
        metrics_repertoire["mean_fitness_diversity"] = jnp.mean(fitnesses_diversity)
        metrics_repertoire = jax.tree_util.tree_map(lambda metric: jnp.repeat(metric, config.algo.log_period), metrics_repertoire)

        # Metrics
        current_metrics["iteration"] = jnp.arange(1+config.algo.log_period*i, 1+config.algo.log_period*(i+1), dtype=jnp.int32)
        current_metrics["time"] = jnp.repeat(timelapse, config.algo.log_period)
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
        state_dict = serialization.to_state_dict(training_state.policy_params)
        with open("./actor/actor_{}.pickle".format(int(metrics["iteration"][-1])), "wb") as params_file:
            pickle.dump(state_dict, params_file)

        # Discriminator
        state_dict = serialization.to_state_dict(training_state.discriminator_params)
        with open("./discriminator/discriminator_{}.pickle".format(int(metrics["iteration"][-1])), "wb") as params_file:
            pickle.dump(state_dict, params_file)

    # Actor
    state_dict = serialization.to_state_dict(training_state.policy_params)
    with open("./actor/actor.pickle", "wb") as params_file:
        pickle.dump(state_dict, params_file)

    # Discriminator
    state_dict = serialization.to_state_dict(training_state.discriminator_params)
    with open("./discriminator/discriminator.pickle", "wb") as params_file:
        pickle.dump(state_dict, params_file)

    # Repertoire
    repertoire.save(path="./repertoire/")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    main()
