import dataclasses
from typing import Tuple, Any, Callable
import functools
import os
import time
import pickle

import jax
import jax.numpy as jnp
from flax import serialization

from baselines.qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from baselines.qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs as scoring_function
from baselines.qdax.environments import get_feat_mean
from baselines.qdax.environments import create
from baselines.qdax.core.map_elites import MAPElites
from baselines.qdax.core.emitters.mutation_operators import isoline_variation
from baselines.qdax.core.containers.archive import score_euclidean_novelty
from baselines.qdax.core.emitters.qpg_emitter import QualityPGConfig
from baselines.qdax.core.emitters.dpg_emitter import DiversityPGConfig
from baselines.qdax.core.emitters.qdpg_emitter import QDPGEmitter, QDPGEmitterConfig
from baselines.qdax.core.neuroevolution.buffers.buffer import QDTransition
from baselines.qdax.core.neuroevolution.networks.networks import MLP
from baselines.qdax.types import Centroid
from baselines.qdax.utils.metrics import CSVLogger, default_qd_metrics
from baselines.qdax.utils.plotting import plot_map_elites_results

import hydra
from hydra.core.config_store import ConfigStore
import wandb
from omegaconf import OmegaConf
from utils.env_utils import Config

@dataclasses.dataclass
class TaskInfo:
    env: Any
    reset_fn: Callable
    centroids: Centroid

    # policy network
    policy_network: MLP

    # population of controllers
    init_params: Any

    # Define the function to play a step with the policy in the environment
    scoring_fn: Callable


class FactoryQDPGTask:
    def __init__(self, config):
        self.config = config


    @classmethod
    def get_scoring_fn(cls, policy_network, env, reset_fn):
        def play_step_fn(env_state, policy_params, random_key):
            actions = policy_network.apply(policy_params, env_state.obs)
            state_desc = env_state.info["feat"]
            next_state = env.step(env_state, actions)

            transition = QDTransition(
                obs=env_state.obs,
                next_obs=next_state.obs,
                rewards=next_state.reward,
                dones=next_state.done,
                truncations=next_state.info["truncation"],
                actions=actions,
                state_desc=state_desc,
                next_state_desc=next_state.info["feat"],
                desc=jnp.zeros(env.behavior_descriptor_length, ) * jnp.nan,
                desc_prime=jnp.zeros(env.behavior_descriptor_length, ) * jnp.nan,
            )

            return next_state, policy_params, random_key, transition

        # Prepare the scoring function
        scoring_fn = functools.partial(
            scoring_function,
            episode_length=env.episode_length,
            play_reset_fn=reset_fn,
            play_step_fn=play_step_fn,
            behavior_descriptor_extractor=get_feat_mean,
        )

        return scoring_fn

    def get_init_params(self, policy_network, env, random_key):
        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, num=self.config.algo.env_batch_size)
        fake_batch_obs = jnp.zeros(shape=(self.config.algo.env_batch_size, env.observation_size))
        init_params = jax.vmap(policy_network.init)(keys, fake_batch_obs)
        return init_params

    def get_centroids(self, env, random_key):
        random_key, subkey = jax.random.split(random_key)
        centroids, random_key = compute_cvt_centroids(
            num_descriptors=env.behavior_descriptor_length,
            num_init_cvt_samples=self.config.algo.num_init_cvt_samples,
            num_centroids=self.config.algo.num_centroids,
            minval=env.behavior_descriptor_limits[0][0],
            maxval=env.behavior_descriptor_limits[1][0],
            random_key=subkey,
        )
        return centroids

    def policy_network(self, env):
        try:
            policy_layer_sizes = self.config.algo.policy_hidden_layer_sizes + (env.action_size,)
        except:
            policy_layer_sizes = self.config.algo.hidden_layer_sizes + (env.action_size,)

        policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
        )
        return policy_network

    def get_env(self):
        env = create(self.config.task + "_" + self.config.feat, episode_length=self.config.algo.episode_length,
                     backend=self.config.algo.backend)
        reset_fn = jax.jit(env.reset)
        return env, reset_fn

    def get_task_info(self, random_key):
        env, reset_fn = self.get_env()

        random_key, subkey = jax.random.split(random_key)
        centroids = self.get_centroids(env, subkey)

        policy_network = self.policy_network(env)

        random_key, subkey = jax.random.split(random_key)
        init_params = self.get_init_params(policy_network, env, subkey)

        scoring_fn = self.get_scoring_fn(policy_network, env, reset_fn)
        return TaskInfo(env, reset_fn, centroids, policy_network, init_params, scoring_fn)


@hydra.main(version_base="1.2", config_path="configs/", config_name="qd_pg")
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

    factory = FactoryQDPGTask(config)

    random_key, subkey = jax.random.split(random_key)
    task_info = factory.get_task_info(subkey)

    env = task_info.env
    reset_fn = task_info.reset_fn
    centroids = task_info.centroids
    policy_network = task_info.policy_network
    init_params = task_info.init_params
    scoring_fn = task_info.scoring_fn


    param_count = sum(x[0].size for x in jax.tree_util.tree_leaves(init_params))
    print("Number of parameters in policy_network: ", param_count)

    @jax.jit
    def evaluate_repertoire(random_key, repertoire):
        repertoire_empty = repertoire.fitnesses == -jnp.inf

        fitnesses, descriptors, extra_scores, random_key = scoring_fn(
            repertoire.genotypes, random_key
        )

        # Compute repertoire QD score
        qd_score = jnp.sum((1.0 - repertoire_empty) * fitnesses).astype(float)

        # Compute repertoire desc error mean
        error = jnp.linalg.norm(repertoire.descriptors - descriptors, axis=1)
        dem = (jnp.sum((1.0 - repertoire_empty) * error) / jnp.sum(1.0 - repertoire_empty)).astype(float)

        return random_key, qd_score, dem

    # @jax.jit
    def evaluate_actor(random_key, actor_params):
        actors_params = jax.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), config.algo.batch_size, axis=0), actor_params)
        fitnesses, _, _, random_key = scoring_fn(actors_params, random_key)
        return random_key, fitnesses.mean()

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = 0

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * env.episode_length,
    )

    # Define the Quality PG emitter config
    qpg_emitter_config = QualityPGConfig(
        env_batch_size=config.algo.qpg_batch_size,
        batch_size=config.algo.batch_size,
        critic_hidden_layer_size=config.algo.critic_hidden_layer_size,
        critic_learning_rate=config.algo.critic_learning_rate,
        actor_learning_rate=config.algo.actor_learning_rate,
        policy_learning_rate=config.algo.policy_learning_rate,
        noise_clip=config.algo.noise_clip,
        policy_noise=config.algo.policy_noise,
        discount=config.algo.discount,
        reward_scaling=config.algo.reward_scaling,
        replay_buffer_size=config.algo.replay_buffer_size,
        soft_tau_update=config.algo.soft_tau_update,
        num_critic_training_steps=config.algo.num_q_critic_training_steps,
        num_pg_training_steps=config.algo.num_pg_training_steps,
        policy_delay=config.algo.policy_delay,
    )

    # Define the Diversity PG emitter config
    dpg_emitter_config = DiversityPGConfig(
        env_batch_size=config.algo.dpg_batch_size,
        batch_size=config.algo.batch_size,
        critic_hidden_layer_size=config.algo.critic_hidden_layer_size,
        critic_learning_rate=config.algo.critic_learning_rate,
        actor_learning_rate=config.algo.actor_learning_rate,
        policy_learning_rate=config.algo.policy_learning_rate,
        noise_clip=config.algo.noise_clip,
        policy_noise=config.algo.policy_noise,
        discount=config.algo.discount,
        reward_scaling=config.algo.reward_scaling,
        replay_buffer_size=config.algo.replay_buffer_size,
        soft_tau_update=config.algo.soft_tau_update,
        num_critic_training_steps=config.algo.num_d_critic_training_steps,
        num_pg_training_steps=config.algo.num_pg_training_steps,
        policy_delay=config.algo.policy_delay,
        archive_acceptance_threshold=config.algo.archive_acceptance_threshold,
        archive_max_size=config.algo.archive_max_size,
    )

    # Define the QDPG Emitter config
    qdpg_emitter_config = QDPGEmitterConfig(
        qpg_config=qpg_emitter_config,
        dpg_config=dpg_emitter_config,
        iso_sigma=config.algo.iso_sigma,
        line_sigma=config.algo.line_sigma,
        ga_batch_size=config.algo.ga_batch_size,
    )

    # Get the emitter
    score_novelty = jax.jit(
        functools.partial(
            score_euclidean_novelty,
            num_nearest_neighb=config.algo.num_nearest_neighb,
            scaling_ratio=config.algo.novelty_scaling_ratio,
        )
    )

    # define the QDPG emitter
    qdpg_emitter = QDPGEmitter(
        config=qdpg_emitter_config,
        policy_network=policy_network,
        env=env,
        score_novelty=score_novelty,
    )

    # Instantiate MAP Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=qdpg_emitter,
        metrics_function=metrics_function,
    )

    # compute initial repertoire
    repertoire, emitter_state, random_key = map_elites.init(init_params, centroids, random_key)

    num_loops = int(config.algo.num_iterations / config.algo.log_period)

    metrics = dict.fromkeys(["iteration", "qd_score", "coverage", "max_fitness", "mean_fitness", "qd_score_repertoire", "dem_repertoire", "q_actor_fitness", "d_actor_fitness", "time"], jnp.array([]))
    csv_logger = CSVLogger(
        "./log.csv",
        header=list(metrics.keys())
    )

    # Main loop
    map_elites_scan_update = map_elites.scan_update
    for i in range(num_loops):
        start_time = time.time()
        (repertoire, emitter_state, random_key,), current_metrics = jax.lax.scan(
            map_elites_scan_update,
            (repertoire, emitter_state, random_key),
            (),
            length=config.algo.log_period,
        )
        timelapse = time.time() - start_time

        # Metrics
        random_key, qd_score_repertoire, dem_repertoire = evaluate_repertoire(random_key, repertoire)
        random_key, fitness_q_actor = evaluate_actor(random_key, emitter_state.emitter_states[0].actor_params)
        random_key, fitness_d_actor = evaluate_actor(random_key, emitter_state.emitter_states[1].actor_params)

        current_metrics["iteration"] = jnp.arange(1+config.algo.log_period*i, 1+config.algo.log_period*(i+1), dtype=jnp.int32)
        current_metrics["time"] = jnp.repeat(timelapse, config.algo.log_period)
        current_metrics["qd_score_repertoire"] = jnp.repeat(qd_score_repertoire, config.algo.log_period)
        current_metrics["dem_repertoire"] = jnp.repeat(dem_repertoire, config.algo.log_period)
        current_metrics["q_actor_fitness"] = jnp.repeat(fitness_q_actor, config.algo.log_period)
        current_metrics["d_actor_fitness"] = jnp.repeat(fitness_d_actor, config.algo.log_period)
        metrics = jax.tree_util.tree_map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)

        # Log
        log_metrics = jax.tree_util.tree_map(lambda metric: metric[-1], metrics)
        csv_logger.log(log_metrics)
        wandb.log(log_metrics)

        # Metrics
        with open("./metrics.pickle", "wb") as metrics_file:
            pickle.dump(metrics, metrics_file)

        # Actor
        state_dict = serialization.to_state_dict(emitter_state.emitter_states[0].actor_params)
        with open("./actor/actor_{}.pickle".format(int(metrics["iteration"][-1])), "wb") as params_file:
            pickle.dump(state_dict, params_file)

    # Actor
    state_dict = serialization.to_state_dict(emitter_state.emitter_states[0].actor_params)
    with open("./actor/actor.pickle", "wb") as params_file:
        pickle.dump(state_dict, params_file)

    # Repertoire
    repertoire.save(path="./repertoire/")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main", node=Config)
    main()
