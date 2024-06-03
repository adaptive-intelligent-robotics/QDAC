import argparse
import os
import warnings
import pickle
from pathlib import Path
from typing import Tuple

import yaml
import functools

import jax
import jax.numpy as jnp
from flax import serialization
import pandas as pd

import dreamerv3
from baselines.PPGA.algorithm.config_ppga import PPGAConfig
from baselines.PPGA.envs.brax_custom.brax_env import make_vec_env_brax_ppga
from baselines.qdax.baselines.domino import DOMINOTransition, DOMINOTrainingState
from baselines.qdax.types import EnvState, RNGKey, Action, Params, Observation
from dreamerv3 import embodied
from dreamerv3.agent import ImagActorCritic
from dreamerv3.embodied.core.goal_sampler import GoalSamplerCyclic
from baselines.qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from utils.analysis_repertoire import AnalysisRepertoire, AnalysisLatentRepertoire
from baselines.qdax import environments
from baselines.qdax.core.neuroevolution.networks.networks import MLPDC, MLP
from baselines.qdax.baselines.diayn_smerl import DIAYNSMERL, DiaynSmerlConfig
from baselines.qdax.core.neuroevolution.buffers.buffer import QDTransition
from baselines.qdax.tasks.brax_envs import reset_based_scoring_actor_dc_function_brax_envs as scoring_actor_dc_function
from baselines.qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs as scoring_actor_function
from baselines.qdax.environments import get_feat_mean
from jax.flatten_util import ravel_pytree
from baselines.qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from dreamerv3.embodied.core.space import AngularSpace

from brax.training.distribution import NormalTanhDistribution

from omegaconf import OmegaConf

from utils.eval_ppga_utils import reevaluate_ppga_archive, load_repertoire_ppga
from utils.env_utils import get_env
warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


task = "ant"

folder_path = Path("/project/analysis/results_paper/")
task_folder_name = "ant_feet_contact"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()

args = get_args()
path = Path(args.path)
algo = args.algo
seed = args.seed


def eval_ours(run_path, gravity_coef):
    config_path = list((run_path / "wandb").iterdir())[0] / "files" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    argv = [
    "--task={}".format(config["task"]["value"]),
    "--feat={}".format(config["feat"]["value"]),
    "--backend={}".format(config["backend"]["value"]),

    "--run.from_checkpoint={}".format(str(run_path / "checkpoint.ckpt")),
    "--envs.amount=2048",
    ]

    # Create config
    logdir = str(run_path)
    config = embodied.Config(dreamerv3.configs["defaults"])
    config = config.update(dreamerv3.configs["brax"])
    config = config.update({
    "logdir": logdir,
    "run.train_ratio": 32,
    "run.log_every": 60,  # Seconds
    "batch_size": 16,
    })
    config = embodied.Flags(config).parse(argv=argv)

    # Create logger
    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(step, [
    embodied.logger.TerminalOutput(),
    embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
    embodied.logger.TensorBoardOutput(logdir),
    # embodied.logger.WandBOutput(logdir, config),
    # embodied.logger.MLFlowOutput(logdir.name),
    ])

    # Create environment
    env = get_env(config, mode="train", gravity_coef=gravity_coef)

    # Create agent and replay buffer
    agent = dreamerv3.Agent(env.obs_space, env.act_space, env.feat_space, step, config)
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)

    # Create goal sampler
    resolution = ImagActorCritic.get_resolution(env.feat_space, config)
    grid_shape = (resolution,) * env.feat_space['vector'].shape[0]
    print("grid_shape", grid_shape)
    goals = compute_euclidean_centroids(grid_shape, minval=env.feat_space['vector'].low, maxval=env.feat_space['vector'].high)
    goal_sampler_cyclic = GoalSamplerCyclic(feat_space=env.feat_space, 
                                            goal_list=list(goals),
                                            number_visits_per_goal=n_visits_per_goal)
    embodied.run.eval_only(agent,
                           env,
                           goal_sampler=goal_sampler_cyclic,
                           period_sample_goals=float('inf'),
                           logger=logger,
                           args=args,)

    ours_repertoire = AnalysisRepertoire.create_from_path_collection_results(run_path / "results_dreamer.pkl")
    # plot_repertoire = ours_repertoire.replace(descriptors=jnp.mean(ours_repertoire.descriptors, axis=1), fitnesses=jnp.mean(ours_repertoire.fitnesses, axis=1))
    # fig, _ = plot_2d_map_elites_repertoire(plot_repertoire.centroids, plot_repertoire.fitnesses, minval=env.feat_space['vector'].low, maxval=env.feat_space['vector'].high)
    # fig.savefig("/project/output/hierarchy/ours_fitness.png")
    # fig, _ = plot_2d_map_elites_repertoire(plot_repertoire.centroids, -jnp.linalg.norm(plot_repertoire.centroids-plot_repertoire.descriptors, axis=-1), minval=env.feat_space['vector'].low, maxval=env.feat_space['vector'].high)
    # fig.savefig("/project/output/hierarchy/ours_distance_to_goal.png")
    return ours_repertoire

def eval_smerl(run_path, gravity_coef):
    with open(run_path / ".hydra" / "config.yaml") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.load(run_path / ".hydra" / "config.yaml")

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = environments.create(config.task + "gravity" + "_" + config.feat,
                              episode_length=config.algo.episode_length,
                              backend=config.algo.backend,
                              gravity_coef=gravity_coef,)

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
        reverse=False,
        diversity_reward_scale=config.algo.diversity_reward_scale,
        smerl_target=config.algo.smerl_target,
        smerl_margin=config.algo.smerl_margin,
    )

    # Define an instance of DIAYN
    smerl = DIAYNSMERL(config=smerl_config, action_size=env.action_size)

    random_key, random_subkey_1, random_subkey_2 = jax.random.split(random_key, 3)
    fake_obs = jnp.zeros((env.observation_size + config.algo.num_skills,))
    fake_goal = jnp.zeros((config.algo.num_skills,))
    fake_actor_params = smerl._policy.init(random_subkey_1, fake_obs)
    fake_discriminator_params = smerl._discriminator.init(random_subkey_2, fake_goal)

    with open(run_path / "actor/actor.pickle", "rb") as params_file:
        state_dict = pickle.load(params_file)
    actor_params = serialization.from_state_dict(fake_actor_params, state_dict)

    with open(run_path / "discriminator/discriminator.pickle", "rb") as params_file:
        state_dict = pickle.load(params_file)
    discriminator_params = serialization.from_state_dict(fake_discriminator_params, state_dict)

    # Create grid
    if config.task == "ant" and config.feat == "feet_contact":
        resolution = 5
    else:
        resolution = 50
    grid_shape = (resolution,) * env.feat_space['vector'].shape[0]
    print("grid_shape", grid_shape)
    goals = compute_euclidean_centroids(grid_shape, minval=env.feat_space['vector'].low, maxval=env.feat_space['vector'].high)
    latent_goals, _ = smerl._discriminator.apply(discriminator_params, goals)

    reset_fn = jax.jit(env.reset)

    @jax.jit
    def play_step_fn(env_state, params, latent_goal, random_key):
        actions, random_key = smerl.select_action(
                    obs=jnp.concatenate([env_state.obs, latent_goal], axis=0),
                    policy_params=params,
                    random_key=random_key,
                    deterministic=True,
                )
        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            truncations=next_state.info["truncation"],
            actions=actions,
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
            desc=jnp.zeros(env.behavior_descriptor_length,) * jnp.nan,
            desc_prime=jnp.zeros(env.behavior_descriptor_length,) * jnp.nan,
        )

        return next_state, params, latent_goal, random_key, transition

    # Prepare the scoring function
    scoring_fn = jax.jit(functools.partial(
        scoring_actor_dc_function,
        episode_length=config.algo.episode_length,
        play_reset_fn=reset_fn,
        play_step_actor_dc_fn=play_step_fn,
        behavior_descriptor_extractor=get_feat_mean,
    ))

    @jax.jit
    def evaluate_actor(random_key, params, latent_goals):
        params = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), latent_goals.shape[0], axis=0), params)
        fitnesses, descriptors, extra_scores, random_key = scoring_fn(
            params, latent_goals, random_key
        )
        return fitnesses, descriptors, extra_scores, random_key
    
    fitnesses_list = []
    descriptor_list = []
    for _ in range(n_visits_per_goal):
        fitnesses, descriptors, extra_scores, random_key = evaluate_actor(random_key, actor_params, latent_goals)
        fitnesses_list.append(fitnesses)
        descriptor_list.append(descriptors)

    smerl_repertoire = AnalysisLatentRepertoire(
        centroids=goals,
        latent_goals=latent_goals,
        fitnesses=jnp.stack(fitnesses_list, axis=1),
        descriptors=jnp.stack(descriptor_list, axis=1))
    # plot_repertoire = smerl_repertoire.replace(descriptors=jnp.mean(smerl_repertoire.descriptors, axis=1), fitnesses=jnp.mean(smerl_repertoire.fitnesses, axis=1))
    # fig, _ = plot_2d_map_elites_repertoire(plot_repertoire.centroids, plot_repertoire.fitnesses, minval=env.feat_space['vector'].low, maxval=env.feat_space['vector'].high)
    # fig.savefig("/project/output/hierarchy/smerl_fitness.png")
    # fig, _ = plot_2d_map_elites_repertoire(plot_repertoire.centroids, -jnp.linalg.norm(goals-descriptors, axis=-1), minval=env.feat_space['vector'].low, maxval=env.feat_space['vector'].high)
    # fig.savefig("/project/output/hierarchy/smerl_distance_to_goal.png")
    return smerl_repertoire

def eval_smerl_reverse(run_path, gravity_coef):
    with open(run_path / ".hydra" / "config.yaml") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.load(run_path / ".hydra" / "config.yaml")

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = environments.create(config.task + "gravity" + "_" + config.feat,
                              episode_length=config.algo.episode_length,
                              backend=config.algo.backend,
                              gravity_coef=gravity_coef,)

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

    random_key, random_subkey_1, random_subkey_2 = jax.random.split(random_key, 3)
    fake_obs = jnp.zeros((env.observation_size + config.algo.num_skills,))
    fake_goal = jnp.zeros((config.algo.num_skills,))
    fake_actor_params = smerl._policy.init(random_subkey_1, fake_obs)
    fake_discriminator_params = smerl._discriminator.init(random_subkey_2, fake_goal)

    with open(run_path / "actor/actor.pickle", "rb") as params_file:
        state_dict = pickle.load(params_file)
    actor_params = serialization.from_state_dict(fake_actor_params, state_dict)

    with open(run_path / "discriminator/discriminator.pickle", "rb") as params_file:
        state_dict = pickle.load(params_file)
    discriminator_params = serialization.from_state_dict(fake_discriminator_params, state_dict)

    # Create grid
    if config.task == "ant" and config.feat == "feet_contact":
        resolution = 5
    else:
        resolution = 50
    grid_shape = (resolution,) * env.feat_space['vector'].shape[0]
    goals = compute_euclidean_centroids(grid_shape, minval=env.feat_space['vector'].low, maxval=env.feat_space['vector'].high)
    latent_goals, _ = smerl._discriminator.apply(discriminator_params, goals)

    reset_fn = jax.jit(env.reset)

    @jax.jit
    def play_step_fn(env_state, params, latent_goal, random_key):
        actions, random_key = smerl.select_action(
                    obs=jnp.concatenate([env_state.obs, latent_goal], axis=0),
                    policy_params=params,
                    random_key=random_key,
                    deterministic=True,
                )
        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            truncations=next_state.info["truncation"],
            actions=actions,
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
            desc=jnp.zeros(env.behavior_descriptor_length,) * jnp.nan,
            desc_prime=jnp.zeros(env.behavior_descriptor_length,) * jnp.nan,
        )

        return next_state, params, latent_goal, random_key, transition

    # Prepare the scoring function
    scoring_fn = jax.jit(functools.partial(
        scoring_actor_dc_function,
        episode_length=config.algo.episode_length,
        play_reset_fn=reset_fn,
        play_step_actor_dc_fn=play_step_fn,
        behavior_descriptor_extractor=get_feat_mean,
    ))

    @jax.jit
    def evaluate_actor(random_key, params, latent_goals):
        params = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), latent_goals.shape[0], axis=0), params)
        fitnesses, descriptors, extra_scores, random_key = scoring_fn(
            params, latent_goals, random_key
        )
        return fitnesses, descriptors, extra_scores, random_key
    
    fitnesses_list = []
    descriptor_list = []
    for _ in range(n_visits_per_goal):
        fitnesses, descriptors, extra_scores, random_key = evaluate_actor(random_key, actor_params, latent_goals)
        fitnesses_list.append(fitnesses)
        descriptor_list.append(descriptors)

    smerl_repertoire = AnalysisLatentRepertoire(
        centroids=goals,
        latent_goals=latent_goals,
        fitnesses=jnp.stack(fitnesses_list, axis=1),
        descriptors=jnp.stack(descriptor_list, axis=1))
    # plot_repertoire = smerl_repertoire.replace(descriptors=jnp.mean(smerl_repertoire.descriptors, axis=1), fitnesses=jnp.mean(smerl_repertoire.fitnesses, axis=1))
    # fig, _ = plot_2d_map_elites_repertoire(plot_repertoire.centroids, plot_repertoire.fitnesses, minval=env.feat_space['vector'].low, maxval=env.feat_space['vector'].high)
    # fig.savefig("/project/output/hierarchy/smerl_reverse_fitness.png")
    # fig, _ = plot_2d_map_elites_repertoire(plot_repertoire.centroids, -jnp.linalg.norm(goals-descriptors, axis=-1), minval=env.feat_space['vector'].low, maxval=env.feat_space['vector'].high)
    # fig.savefig("/project/output/hierarchy/smerl_reverse_distance_to_goal.png")
    return smerl_repertoire

def eval_uvfa(run_path, gravity_coef):
    config_path = run_path / "wandb" / "latest-run" / "files" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    argv = [
    "--task={}".format(config["task"]["value"]),
    "--feat={}".format(config["feat"]["value"]),
    "--backend={}".format(config["backend"]["value"]),

    "--run.from_checkpoint={}".format(str(run_path / "checkpoint.ckpt")),
    "--envs.amount=2048",
    ]

    # Create config
    logdir = str(run_path)
    config = embodied.Config(dreamerv3.configs["defaults"])
    config = config.update(dreamerv3.configs["brax"])
    config = config.update({
    "logdir": logdir,
    "run.train_ratio": 32,
    "run.log_every": 60,  # Seconds
    "batch_size": 16,
    })
    config = embodied.Flags(config).parse(argv=argv)

    # Create logger
    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(step, [
    embodied.logger.TerminalOutput(),
    embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
    embodied.logger.TensorBoardOutput(logdir),
    # embodied.logger.WandBOutput(logdir, config),
    # embodied.logger.MLFlowOutput(logdir.name),
    ])

    # Create environment
    env = get_env(config, mode="train", gravity_coef=gravity_coef)

    # Create agent and replay buffer
    agent = dreamerv3.Agent(env.obs_space, env.act_space, env.feat_space, step, config)
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)

    # Create goal sampler
    resolution = ImagActorCritic.get_resolution(env.feat_space, config)
    grid_shape = (resolution,) * env.feat_space['vector'].shape[0]
    goals = compute_euclidean_centroids(grid_shape, minval=env.feat_space['vector'].low, maxval=env.feat_space['vector'].high)
    goal_sampler_cyclic = GoalSamplerCyclic(feat_space=env.feat_space, 
                                            goal_list=list(goals),
                                            number_visits_per_goal=n_visits_per_goal)
    embodied.run.eval_only(agent,
                           env,
                           goal_sampler=goal_sampler_cyclic,
                           period_sample_goals=float('inf'),
                           logger=logger,
                           args=args,)

    ours_repertoire = AnalysisRepertoire.create_from_path_collection_results(run_path / "results_dreamer.pkl")
    # plot_repertoire = ours_repertoire.replace(descriptors=jnp.mean(ours_repertoire.descriptors, axis=1), fitnesses=jnp.mean(ours_repertoire.fitnesses, axis=1))
    # fig, _ = plot_2d_map_elites_repertoire(plot_repertoire.centroids, plot_repertoire.fitnesses, minval=env.feat_space['vector'].low, maxval=env.feat_space['vector'].high)
    # fig.savefig("/project/output/hierarchy/ours_fitness.png")
    # fig, _ = plot_2d_map_elites_repertoire(plot_repertoire.centroids, -jnp.linalg.norm(plot_repertoire.centroids-plot_repertoire.descriptors, axis=-1), minval=env.feat_space['vector'].low, maxval=env.feat_space['vector'].high)
    # fig.savefig("/project/output/hierarchy/ours_distance_to_goal.png")
    return ours_repertoire

def eval_dcg_me(run_path, gravity_coef):
    with open(run_path / ".hydra" / "config.yaml") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.load(run_path / ".hydra" / "config.yaml")

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = environments.create(config.task + "gravity" + "_" + config.feat,
                              episode_length=config.algo.episode_length,
                              backend=config.algo.backend,
                              gravity_coef=gravity_coef,)
    reset_fn = jax.jit(env.reset)

    # Init policy network
    policy_layer_sizes = config.algo.policy_hidden_layer_sizes + (env.action_size,)
    actor_dc_network = MLPDC(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    fake_obs = jnp.zeros(shape=(env.observation_size,))
    fake_desc = jnp.zeros(shape=(env.behavior_descriptor_length,))
    fake_actor_params = actor_dc_network.init(subkey, fake_obs, fake_desc)

    with open(run_path / "actor/actor.pickle", "rb") as params_file:
        state_dict = pickle.load(params_file)
    actor_params = serialization.from_state_dict(fake_actor_params, state_dict)

    # Create grid
    if config.task == "ant" and config.feat == "feet_contact":
        resolution = 5
    else:
        resolution = 50
    grid_shape = (resolution,) * env.feat_space['vector'].shape[0]
    goals = compute_euclidean_centroids(grid_shape, minval=env.feat_space['vector'].low, maxval=env.feat_space['vector'].high)

    def play_step_actor_dc_fn(env_state, actor_dc_params, desc, random_key):
        actions = actor_dc_network.apply(actor_dc_params, env_state.obs, desc/env.behavior_descriptor_limits[1][0])
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
            desc=jnp.zeros(env.behavior_descriptor_length,) * jnp.nan,
            desc_prime=desc/env.behavior_descriptor_limits[1][0],
        )

        return next_state, actor_dc_params, desc, random_key, transition

    # Prepare the scoring function
    scoring_fn = jax.jit(functools.partial(
        scoring_actor_dc_function,
        episode_length=env.episode_length,
        play_reset_fn=reset_fn,
        play_step_actor_dc_fn=play_step_actor_dc_fn,
        behavior_descriptor_extractor=get_feat_mean,
    ))

    @jax.jit
    def evaluate_actor(random_key, params, goals):
        params = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), goals.shape[0], axis=0), params)
        fitnesses, descriptors, extra_scores, random_key = scoring_fn(
            params, goals, random_key
        )
        return fitnesses, descriptors, extra_scores, random_key
    
    fitnesses_list = []
    descriptor_list = []
    for _ in range(n_visits_per_goal):
        fitnesses, descriptors, extra_scores, random_key = evaluate_actor(random_key, actor_params, goals)
        fitnesses_list.append(fitnesses)
        descriptor_list.append(descriptors)

    smerl_repertoire = AnalysisRepertoire(
        centroids=goals,
        fitnesses=jnp.stack(fitnesses_list, axis=1),
        descriptors=jnp.stack(descriptor_list, axis=1))
    # plot_repertoire = smerl_repertoire.replace(descriptors=jnp.mean(smerl_repertoire.descriptors, axis=1), fitnesses=jnp.mean(smerl_repertoire.fitnesses, axis=1))
    # fig, _ = plot_2d_map_elites_repertoire(plot_repertoire.centroids, plot_repertoire.fitnesses, minval=env.feat_space['vector'].low, maxval=env.feat_space['vector'].high)
    # fig.savefig("/project/output/hierarchy/dcg_me_fitness.png")
    # fig, _ = plot_2d_map_elites_repertoire(plot_repertoire.centroids, -jnp.linalg.norm(goals-descriptors, axis=-1), minval=env.feat_space['vector'].low, maxval=env.feat_space['vector'].high)
    # fig.savefig("/project/output/hierarchy/dcg_me_distance_to_goal.png")
    return smerl_repertoire


def eval_qd_pg(run_path, gravity_coef):
    with open(run_path / ".hydra" / "config.yaml") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.load(run_path / ".hydra" / "config.yaml")

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = environments.create(config.task + "gravity" + "_" + config.feat,
                              episode_length=config.algo.episode_length,
                              backend=config.algo.backend,
                              gravity_coef=gravity_coef,)
    reset_fn = jax.jit(env.reset)

    # Init policy network
    policy_layer_sizes = config.algo.policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init params
    random_key, subkey = jax.random.split(random_key)
    fake_obs = jnp.zeros(shape=(env.observation_size,))
    fake_params = policy_network.init(subkey, fake_obs)

    # Init repertoire
    _, reconstruction_fn = ravel_pytree(fake_params)

    # Build the repertoire
    repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=str(run_path) + "/repertoire/")

    # Create grid
    if config.task == "ant" and config.feat == "feet_contact":
        resolution = 5
    else:
        resolution = 50
    grid_shape = (resolution,) * env.feat_space['vector'].shape[0]
    goals = compute_euclidean_centroids(grid_shape, minval=env.feat_space['vector'].low, maxval=env.feat_space['vector'].high)

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
    scoring_fn = jax.jit(functools.partial(
        scoring_actor_function,
        episode_length=env.episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=get_feat_mean,
    ))

    @jax.jit
    def evaluate_actor(random_key, goals):
        distances = jax.vmap(lambda x, y: jnp.linalg.norm(x - y, axis=-1), in_axes=(None, 0))(repertoire.descriptors, goals)
        indices = jax.vmap(jnp.argmin)(distances)
        params = jax.tree_util.tree_map(lambda x: x[indices], repertoire.genotypes)
        fitnesses, descriptors, extra_scores, random_key = scoring_fn(params, random_key)
        return fitnesses, descriptors, extra_scores, random_key

    fitnesses_list = []
    descriptor_list = []
    for _ in range(n_visits_per_goal):
        fitnesses, descriptors, extra_scores, random_key = evaluate_actor(random_key, goals)
        fitnesses_list.append(fitnesses)
        descriptor_list.append(descriptors)

    smerl_repertoire = AnalysisRepertoire(
        centroids=goals,
        fitnesses=jnp.stack(fitnesses_list, axis=1),
        descriptors=jnp.stack(descriptor_list, axis=1))
    return smerl_repertoire



def eval_ppga(run_path, gravity_coef):
    with open(run_path / ".hydra" / "config.yaml") as f:
        hydra_config = yaml.safe_load(f)
    hydra_config = OmegaConf.load(run_path / ".hydra" / "config.yaml")

    resolution = 5
    n_visits_per_skill = 1

    # Init a random key
    random_key = jax.random.PRNGKey(hydra_config.seed)

    cfg = PPGAConfig.create(hydra_config)
    cfg = cfg.as_dot_dict()

    cfg = PPGAConfig.create(hydra_config)
    cfg = cfg.as_dot_dict()

    cfg.num_emitters = 1

    cfg.batch_size = int(cfg.env_batch_size * cfg.rollout_length)
    cfg.num_envs = int(cfg.env_batch_size)

    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)

    if hydra_config.feat == "angle_notrigo":
        print("Angular space - USE NOTRIGO")
        hydra_config.feat = "angle"

    random_key = jax.random.PRNGKey(hydra_config.seed)
    # reevaluator = ReEvaluator(task_info.scoring_fn, num_reevals)

    repertoire_ppga = load_repertoire_ppga(str(run_path) + '/')

    num_sols = len(repertoire_ppga)
    print(f'{num_sols=}')
    solution_batch_size = 1000
    cfg.env_batch_size = n_visits_per_skill * solution_batch_size
    vec_env = make_vec_env_brax_ppga(task_name=hydra_config.task + "gravity", feat_name=hydra_config.feat,
                                     batch_size=cfg.env_batch_size,
                                     seed=cfg.seed, backend=cfg.backend, clip_obs_rew=cfg.clip_obs_rew,
                                     episode_length=hydra_config.algo.episode_length,
                                     gravity_coef=gravity_coef,
                                     )

    cfg.obs_shape = vec_env.single_observation_space.shape
    cfg.action_shape = vec_env.single_action_space.shape

    cfg.bd_min = vec_env.behavior_descriptor_limits[0][0]
    cfg.bd_max = vec_env.behavior_descriptor_limits[1][0]

    if hydra_config.feat == "angle":
        print("Angular space")
        shape_feat_space = vec_env.env.feat_space["vector"].shape[0] // 2
        cfg.num_dims = 2
    else:
        print("Not angular space", vec_env.env.feat_space["vector"].__class__)
        shape_feat_space = vec_env.env.feat_space["vector"].shape[0]
    grid_shape = (resolution,) * shape_feat_space
    centroids = compute_euclidean_centroids(grid_shape, minval=vec_env.env.feat_space['vector'].low,
                                            maxval=vec_env.env.feat_space['vector'].high)
    centroids = vec_env.env.feat_space['vector'].transform(centroids)

    num_centroids = len(centroids)
    print(f'{num_centroids=}')

    def _highest_divider_below_or_equal_1000(n):
        # Check if the number is less than or equal to 1000, or if 1000 is a divisor of the number
        if n <= 1000 or n % 1000 == 0:
            return min(n, 1000)  # Return the number itself if it's below 1000, or 1000 if it's a divisor

        # Start from the minimum of 1000 and n (to include 1000 and the number itself as possible divisors)
        for i in range(min(1000, n), 0, -1):
            if n % i == 0:
                return i

    solution_batch_size = _highest_divider_below_or_equal_1000(num_centroids)
    print(f'{solution_batch_size=}')
    cfg.env_batch_size = n_visits_per_skill * solution_batch_size

    del vec_env

    vec_env = make_vec_env_brax_ppga(task_name=hydra_config.task + "gravity", feat_name=hydra_config.feat,
                                     batch_size=cfg.env_batch_size,
                                     seed=cfg.seed, backend=cfg.backend, clip_obs_rew=cfg.clip_obs_rew,
                                     episode_length=hydra_config.algo.episode_length,
                                     gravity_coef=gravity_coef,
                                     )
    if hydra_config.feat == "angle":
        transform_descs_fn = vec_env.env.feat_space['vector'].transform
    else:
        transform_descs_fn = None

    analysis_repertoire = reevaluate_ppga_archive(cfg, hydra_config, repertoire_ppga, vec_env, solution_batch_size,
                                                  n_visits_per_skill, centroids,
                                                  transform_descs_fn=transform_descs_fn)  # TODO

    return analysis_repertoire



def eval_domino(run_path, gravity_coef):
    with open(run_path / ".hydra" / "config.yaml") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.load(run_path / ".hydra" / "config.yaml")

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = environments.create(config.task + "gravity" + "_" + config.feat,
                              episode_length=config.algo.episode_length,
                              backend=config.algo.backend,
                              gravity_coef=gravity_coef,
                              )
    reset_fn = jax.jit(env.reset)





    num_policies = config.algo.num_skills

    from baselines.qdax.baselines.domino_networks import make_domino_networks


    # Init policy network
    _policy, _ = make_domino_networks(
        action_size=env.action_size,
        hidden_layer_sizes=config.algo.hidden_layer_sizes,
    )
    # actor_dc_network = MLPDC(
    #     layer_sizes=policy_layer_sizes,
    #     kernel_init=jax.nn.initializers.lecun_uniform(),
    #     final_activation=jnp.tanh,
    # )
    # TODO: add domino policy network

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    array_subkeys = jax.random.split(subkey, num_policies)
    fake_obs = jnp.zeros(shape=(env.observation_size,))
    fake_desc = jnp.zeros(shape=(env.behavior_descriptor_length,))
    fake_actor_params = jax.vmap(_policy.init, in_axes=(0, None))(array_subkeys, fake_obs)

    with open(run_path / "actor/actor.pickle", "rb") as params_file:
        state_dict = pickle.load(params_file)
    actor_params = serialization.from_state_dict(fake_actor_params, state_dict)


    # Build the repertoire
    # repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=str(run_path) + "/repertoire/")

    # Create grid
    resolution = 50
    grid_shape = (resolution,) * env.feat_space['vector'].shape[0]
    # goals = compute_euclidean_centroids(grid_shape, minval=env.feat_space['vector'].low, maxval=env.feat_space['vector'].high)

    _parametric_action_distribution = NormalTanhDistribution(
        event_size=env.action_size
    )
    _sample_action_fn = _parametric_action_distribution.sample

    def select_action(
        obs: Observation,
        policy_params: Params,
        random_key: RNGKey,
        deterministic: bool = False,
    ) -> Tuple[Action, RNGKey]:
        """Selects an action acording to SAC policy.

        Args:
            obs: agent observation(s)
            policy_params: parameters of the agent's policy
            random_key: jax random key
            deterministic: whether to select action in a deterministic way.
                Defaults to False.

        Returns:
            The selected action and a new random key.
        """

        dist_params = _policy.apply(policy_params, obs)
        if not deterministic:
            random_key, key_sample = jax.random.split(random_key)
            actions = _sample_action_fn(dist_params, key_sample)

        else:
            # The first half of parameters is for mean and the second half for variance
            actions = jax.nn.tanh(dist_params[..., : dist_params.shape[-1] // 2])

        return actions, random_key

    def play_step_fn(
        env_state: EnvState,
        policy_params,
        random_key,
    ) -> Tuple[EnvState, DOMINOTrainingState, RNGKey, QDTransition]:
        """Plays a step in the environment. Concatenates skills to the observation
        vector, selects an action according to SAC rule and performs the environment
        step.

        Args:
            env_state: the current environment state
            training_state: the DIAYN training state
            skills: the skills concatenated to the observation vector
            env: the environment
            deterministic: whether or not to select action in a deterministic way.
                Defaults to False.

        Returns:
            the new environment state
            the new DIAYN training state
            the played transition
        """

        # Sample skill from prior when new episode starts
        random_keys = jax.random.split(random_key, env_state.obs.shape[0]+1)
        random_key, random_keys = random_keys[0], random_keys[1:]

        obs = env_state.obs

        # If the env does not support state descriptor, we set it to (0,0)
        if "state_descriptor" in env_state.info:
            state_desc = env_state.info["state_descriptor"]
        else:
            state_desc = jnp.zeros((env_state.obs.shape[0], 2))

        actions, random_key = select_action(
            obs=obs,
            policy_params=policy_params,
            random_key=random_key,
            deterministic=True,
        )

        next_env_state = env.step(env_state, actions)
        next_obs = next_env_state.obs
        if "state_descriptor" in next_env_state.info:
            next_state_desc = next_env_state.info["state_descriptor"]
        else:
            next_state_desc = jnp.zeros((next_env_state.obs.shape[0], 2))
        truncations = next_env_state.info["truncation"]
        transition = DOMINOTransition(
            obs=obs,
            next_obs=next_obs,
            state_desc=state_desc,
            next_state_desc=next_state_desc,
            rewards=next_env_state.reward,
            dones=next_env_state.done,
            actions=actions,
            truncations=truncations,
            desc=jnp.zeros((env_state.obs.shape[0], env.behavior_descriptor_length,)) * jnp.nan,
            desc_prime=jnp.zeros((env_state.obs.shape[0], env.behavior_descriptor_length,)) * jnp.nan,
        )

        return next_env_state, policy_params, random_key, transition

    # Prepare the scoring function
    scoring_fn = jax.jit(functools.partial(
        scoring_actor_function,
        episode_length=env.episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=get_feat_mean,
    ))

    @jax.jit
    def evaluate_actor(random_key,):
        # distances = jax.vmap(lambda x, y: jnp.linalg.norm(x - y, axis=-1), in_axes=(None, 0))(repertoire.descriptors, goals)
        # indices = jax.vmap(jnp.argmin)(distances)
        # params = jax.tree_util.tree_map(lambda x: x[indices], repertoire.genotypes)
        random_key, subkey = jax.random.split(random_key)
        fitnesses, descriptors, extra_scores, random_key = scoring_fn(actor_params, subkey)
        return fitnesses, descriptors, extra_scores

    fitnesses_list = []
    descriptor_list = []
    for _ in range(n_visits_per_goal):
        random_key, subkey = jax.random.split(random_key)
        fitnesses, descriptors, extra_scores = evaluate_actor(subkey,)
        fitnesses_list.append(fitnesses)
        descriptor_list.append(descriptors)

    analysis_repertoire = AnalysisRepertoire(
        centroids=jnp.stack(descriptor_list, axis=1),
        fitnesses=jnp.stack(fitnesses_list, axis=1),
        descriptors=jnp.stack(descriptor_list, axis=1),
    )
    return analysis_repertoire


def eval_scopa(run_path, gravity_coef):
    with open(run_path / ".hydra" / "config.yaml") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.load(run_path / ".hydra" / "config.yaml")

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = environments.create(config.task + "gravity" + "_" + config.feat,
                              episode_length=config.algo.episode_length,
                              backend=config.algo.backend,
                              gravity_coef=gravity_coef,)
    reset_fn = jax.jit(env.reset)

    # Init policy network
    policy_layer_sizes = config.algo.hidden_layer_sizes + (2 * env.action_size,)
    actor_network = MLPDC(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    fake_obs = jnp.zeros(shape=(env.observation_size,))
    fake_desc = jnp.zeros(shape=(env.behavior_descriptor_length,))
    fake_actor_params = actor_network.init(subkey, fake_obs, fake_desc)

    with open(run_path / "actor/actor.pickle", "rb") as params_file:
        state_dict = pickle.load(params_file)
    actor_params = serialization.from_state_dict(fake_actor_params, state_dict)

    @jax.jit
    def normalize_feat(feat: jnp.ndarray) -> jnp.ndarray:
        if isinstance(env.feat_space["vector"], AngularSpace):
            return feat
        else:
            return 2*(feat - env.feat_space["vector"].low)/(env.feat_space["vector"].high - env.feat_space["vector"].low) - 1


    # Create grid
    if config.task == "ant" and config.feat == "feet_contact":
        resolution = 5
    else:
        resolution = 50
    grid_shape = (resolution,) * env.feat_space['vector'].shape[0]
    goals = compute_euclidean_centroids(grid_shape, minval=env.feat_space['vector'].low, maxval=env.feat_space['vector'].high)

    def play_step_actor_fn(env_state, actor_params, skill, random_key):
        skill_normalized = normalize_feat(skill)
        dist_params = actor_network.apply(actor_params, env_state.obs, skill_normalized)
        actions = jax.nn.tanh(dist_params[..., : dist_params.shape[-1] // 2])
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
            desc=jnp.zeros(env.behavior_descriptor_length,) * jnp.nan,
            desc_prime=skill,
        )

        return next_state, actor_params, skill, random_key, transition

    # Prepare the scoring function
    scoring_fn = jax.jit(functools.partial(
        scoring_actor_dc_function,
        episode_length=env.episode_length,
        play_reset_fn=reset_fn,
        play_step_actor_dc_fn=play_step_actor_fn,
        behavior_descriptor_extractor=get_feat_mean,
    ))

    @jax.jit
    def evaluate_actor(random_key, params, goals):
        params = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), goals.shape[0], axis=0), params)
        fitnesses, descriptors, extra_scores, random_key = scoring_fn(
            params, goals, random_key
        )
        return fitnesses, descriptors, extra_scores, random_key
    
    fitnesses_list = []
    descriptor_list = []
    for _ in range(n_visits_per_goal):
        fitnesses, descriptors, extra_scores, random_key = evaluate_actor(random_key, actor_params, goals)
        fitnesses_list.append(fitnesses)
        descriptor_list.append(descriptors)

    smerl_repertoire = AnalysisRepertoire(
        centroids=goals,
        fitnesses=jnp.stack(fitnesses_list, axis=1),
        descriptors=jnp.stack(descriptor_list, axis=1))
    return smerl_repertoire


random_key = jax.random.PRNGKey(1234)

n_gravity_coef = 20 # 40
n_visits_per_goal = 1

dict_eval_fn = {
    "qdac_mb": eval_ours,
    "smerl": eval_smerl,
    "smerl_reverse": eval_smerl_reverse,
    "uvfa": eval_uvfa,
    "dcg_me": eval_dcg_me,
    "qd_pg": eval_qd_pg,
    "qdac": eval_scopa,
    "domino": eval_domino,
    "ppga": eval_ppga,
}
eval_fn = dict_eval_fn[algo]

df = pd.DataFrame(columns=["algo", "seed", "gravity_coef", "fitness", "distance_to_goal"])

for j, gravity_coef in enumerate(jnp.linspace(0.5, 3, n_gravity_coef)):
    print(f"seed: {seed}, gravity_coef: {gravity_coef}, {j}/{n_gravity_coef}")
    print(f"{algo}: {gravity_coef}")
    analysis_repertoire = eval_fn(path, gravity_coef)
    df.loc[len(df)] = [algo, seed, gravity_coef, jnp.max(jnp.median(analysis_repertoire.fitnesses, axis=-1)), jnp.mean(-jnp.linalg.norm(analysis_repertoire.centroids - jnp.mean(analysis_repertoire.descriptors, axis=1), axis=-1))]

    path_csv = os.path.join(
        path,
        f"gravity_{algo}_{task}_{seed}_{n_gravity_coef}_{n_visits_per_goal}.csv"
    )
    df.to_csv(path_csv)
