import os

from brax.io.torch import jax_to_torch, torch_to_jax

from baselines.PPGA.envs.brax_custom.brax_env import make_vec_env_brax_ppga

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import torch

torch.ones(1, device='cuda')  # init torch cuda before jax

from baselines.PPGA.models.vectorized import VectorizedActor


from pathlib import Path

import numpy as np
import yaml
import pickle
from dataclasses import dataclass
import functools
import time

import jax
import jax.numpy as jnp
from flax import serialization

import brax
from brax import envs
import dreamerv3
from baselines.PPGA.models.actor_critic import Actor
from baselines.qdax.baselines.sac_discrete import SACDiscrete
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_brax, antwall
from dreamerv3.embodied.core import feat_wrappers
from baselines.qdax import environments
from baselines.qdax.core.neuroevolution.networks.networks import MLPDC, MLP
from baselines.qdax.baselines.diayn_smerl import DIAYNSMERL, DiaynSmerlConfig
from baselines.qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from baselines.qdax.baselines.sac import SacConfig, SAC
from baselines.qdax.core.neuroevolution.sac_td3_utils import do_iteration_fn, warmstart_buffer, do_iteration_no_jit_fn, warmstart_buffer_no_jit
from baselines.qdax.utils.metrics import CSVLogger
from jax.flatten_util import ravel_pytree
from baselines.qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from dreamerv3.embodied.core.space import AngularSpace


import wandb
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


def load_repertoire_ppga(folder_load: str):  # TODO
    import os
    from pathlib import Path

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
    # archive_path = 'experiments/ppga_humanoid_imp_var_ranker/1111/checkpoints/cp_00002000/cp_00002000/archive_df_00002000.pkl' # TODO
    # scheduler_path = 'experiments/ppga_humanoid_imp_var_ranker/1111/checkpoints/cp_00002000/cp_00002000/scheduler_00002000.pkl'
    with open(str(archive_path), 'rb') as f:
        archive_df = pickle.load(f)
    with open(str(scheduler_path), 'rb') as f:
        scheduler = pickle.load(f)

    archive = scheduler.archive

    return archive


@dataclass
class Config:
    seed: int
    algo_name: str
    num_iterations: int
    log_period: int
    env_batch_size: int
    action_repeat: int
    path: str


def get_meta_env_ours(config_hydra, run_path):
    run_path = Path(run_path)
    try:
        config_path = list((run_path / "wandb").iterdir())[0] / "files" / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except:
        config_path = run_path / "wandb" / "latest-run" / "files" / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

    argv = [
    "--task={}".format(config["task"]["value"]),
    "--feat={}".format(config["feat"]["value"]),
    "--backend={}".format(config["backend"]["value"]),

    "--run.from_checkpoint={}".format(str(run_path / "checkpoint.ckpt")),
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
    brax.envs.register_environment("antwall", antwall.AntWall)
    env = brax.envs.create(env_name=config.task + "wall",
                           episode_length=config.episode_length,
                           action_repeat=config_hydra.action_repeat,
                           auto_reset=True,
                           batch_size=config_hydra.env_batch_size,
                           backend=config.backend,
                           debug=True)
    env = feat_wrappers.VelocityWrapper(env, config.task)

    # Create agent
    env_space = from_brax.FromBraxVec(env, obs_key="vector", seed=config.seed, n_envs=config_hydra.env_batch_size)
    env_space = dreamerv3.wrap_env(env_space, config)
    agent = dreamerv3.Agent(env_space.obs_space, env_space.act_space, env_space.feat_space, step, config)
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=['agent'])
    policy = lambda *args: agent.policy(*args, mode='eval')

    class MetaEnv(envs.Env):
        def __init__(self, env):
            self.env = env
            self._state = None
            self._step_fn = jax.jit(self.env.step)
            self._reset_fn = jax.jit(self.env.reset)
            self._position_normalization = 50.

        def get_obs(self, state, action):
            return {
                "vector": state.obs[..., 1:],
                "is_first": state.info["steps"] == 0,
                "is_terminal": state.done == 1,
                "feat": state.info["feat"],
                "goal": action,
            }

        def reset(self, rng):
            state = self.env.reset(rng)
            return state.replace(obs=self._augment_obs(state))

        def step(self, state, action):
            obs = self.get_obs(state, self._scale_action(action))
            acts, self._state = policy(obs, self._state)
            next_state = self._step_fn(state.replace(obs=state.obs[..., 1:]), jnp.array(acts["action"]))
            return next_state.replace(
                obs=self._augment_obs(next_state),
                reward=next_state.reward + self._get_reward(state, next_state))

        def _scale_action(self, action):
            return 0.5 * (self.env.behavior_descriptor_limits[1] - self.env.behavior_descriptor_limits[0]) * action + \
                0.5 * (self.env.behavior_descriptor_limits[1] + self.env.behavior_descriptor_limits[0])

        def _augment_obs(self, state):
            return jnp.concatenate([state.pipeline_state.q[..., 0:1]/self._position_normalization, state.obs], axis=-1)

        def _get_reward(self, state, next_state):
            return 0.

        @property
        def action_size(self):
            return self.env.feat_size

        @property
        def observation_size(self):
            return self.env.observation_size + 1

        @property
        def backend(self):
            return self.env.backend

        @property
        def episode_length(self):
            return self.env.episode_length

    return MetaEnv(env)


def get_meta_env_smerl(config_hydra, run_path):
    run_path = Path(run_path)
    with open(run_path / ".hydra" / "config.yaml") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.load(run_path / ".hydra" / "config.yaml")

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = environments.create(config.task + "wall" + "_" + config.feat,
                              episode_length=config.algo.episode_length,
                              action_repeat=config_hydra.action_repeat,
                              batch_size=config_hydra.env_batch_size,
                              backend=config.algo.backend,)

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

    class MetaEnv(envs.Env):
        def __init__(self, env):
            self.env = env
            self._position_normalization = 50.

        def reset(self, rng):
            state = self.env.reset(rng)
            return state.replace(obs=self._augment_obs(state))

        def step(self, state, action):
            latent_skill, _ = smerl._discriminator.apply(discriminator_params, self._scale_action(action))
            action, _ = smerl.select_action(
                obs=jnp.concatenate([state.obs[..., 1:], latent_skill], axis=-1),
                policy_params=actor_params,
                random_key=None,
                deterministic=True,)
            next_state = self.env.step(state.replace(obs=state.obs[..., 1:]), action)
            return next_state.replace(
                obs=self._augment_obs(next_state),
                reward=next_state.reward + self._get_reward(state, next_state))

        def _scale_action(self, action):
            return 0.5 * (self.env.behavior_descriptor_limits[1] - self.env.behavior_descriptor_limits[0]) * action + \
                0.5 * (self.env.behavior_descriptor_limits[1] + self.env.behavior_descriptor_limits[0])

        def _augment_obs(self, state):
            return jnp.concatenate([state.pipeline_state.q[..., 0:1]/self._position_normalization, state.obs], axis=-1)

        def _get_reward(self, state, next_state):
            return 0.

        @property
        def action_size(self):
            return self.env.feat_size

        @property
        def observation_size(self):
            return self.env.observation_size + 1

        @property
        def backend(self):
            return self.env.backend

        @property
        def episode_length(self):
            return self.env.episode_length

    return MetaEnv(env)


def get_meta_env_smerl_full_state(config_hydra, run_path, reverse=False):
    run_path = Path(run_path)
    with open(run_path / ".hydra" / "config.yaml") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.load(run_path / ".hydra" / "config.yaml")

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = environments.create(config.task + "wall" + "_" + config.feat,
                              episode_length=config.algo.episode_length,
                              action_repeat=config_hydra.action_repeat,
                              batch_size=config_hydra.env_batch_size,
                              backend=config.algo.backend,)

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
        reverse=reverse,
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

    with open(run_path / "actor/actor.pickle", "rb") as params_file:
        state_dict = pickle.load(params_file)
    actor_params = serialization.from_state_dict(fake_actor_params, state_dict)

    class MetaEnv(envs.Env):
        def __init__(self, env):
            self.env = env
            self._position_normalization = 50.

        def reset(self, rng):
            state = self.env.reset(rng)
            return state.replace(obs=self._augment_obs(state))

        def step(self, state, action):
            # action should be discrete here
            # assert jnp.abs(jnp.mean(action) - (1. / self.action_size)) < 1e-6
            # print(action)
            # override = jnp.zeros_like(action)
            # override = override.at[..., 0].set(1.)
            # action = override
            action, _ = smerl.select_action(
                obs=jnp.concatenate([state.obs[..., 1:], action], axis=-1),
                policy_params=actor_params,
                random_key=None,
                deterministic=True,)
            next_state = self.env.step(state.replace(obs=state.obs[..., 1:]), action)
            return next_state.replace(
                obs=self._augment_obs(next_state),
                reward=next_state.reward + self._get_reward(state, next_state))

        def _scale_action(self, action):
            raise NotImplementedError

        def _augment_obs(self, state):
            return jnp.concatenate([state.pipeline_state.q[..., 0:1]/self._position_normalization, state.obs], axis=-1)

        def _get_reward(self, state, next_state):
            return 0.

        @property
        def action_size(self):
            return config.algo.num_skills

        @property
        def observation_size(self):
            return self.env.observation_size + 1

        @property
        def backend(self):
            return self.env.backend

        @property
        def episode_length(self):
            return self.env.episode_length

    return MetaEnv(env)


def get_meta_env_smerl_reverse(config_hydra, run_path,):
    run_path = Path(run_path)
    with open(run_path / ".hydra" / "config.yaml") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.load(run_path / ".hydra" / "config.yaml")

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = environments.create(config.task + "wall" + "_" + config.feat,
                              episode_length=config.algo.episode_length,
                              action_repeat=config_hydra.action_repeat,
                              batch_size=config_hydra.env_batch_size,
                              backend=config.algo.backend,)

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

    class MetaEnv(envs.Env):
        def __init__(self, env):
            self.env = env
            self._position_normalization = 50.

        def reset(self, rng):
            state = self.env.reset(rng)
            return state.replace(obs=self._augment_obs(state))

        def step(self, state, action):
            latent_skill, _ = smerl._discriminator.apply(discriminator_params, self._scale_action(action))
            action, _ = smerl.select_action(
                obs=jnp.concatenate([state.obs[..., 1:], latent_skill], axis=-1),
                policy_params=actor_params,
                random_key=None,
                deterministic=True,)
            next_state = self.env.step(state.replace(obs=state.obs[..., 1:]), action)
            return next_state.replace(
                obs=self._augment_obs(next_state),
                reward=next_state.reward + self._get_reward(state, next_state))

        def _scale_action(self, action):
            return 0.5 * (self.env.behavior_descriptor_limits[1] - self.env.behavior_descriptor_limits[0]) * action + \
                0.5 * (self.env.behavior_descriptor_limits[1] + self.env.behavior_descriptor_limits[0])

        def _augment_obs(self, state):
            return jnp.concatenate([state.pipeline_state.q[..., 0:1]/self._position_normalization, state.obs], axis=-1)

        def _get_reward(self, state, next_state):
            return 0.

        @property
        def action_size(self):
            return self.env.feat_size

        @property
        def observation_size(self):
            return self.env.observation_size + 1

        @property
        def backend(self):
            return self.env.backend

        @property
        def episode_length(self):
            return self.env.episode_length

    return MetaEnv(env)


def get_meta_env_uvfa(config_hydra, run_path):
    run_path = Path(run_path)
    try:
        config_path = list((run_path / "wandb").iterdir())[0] / "files" / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except:
        config_path = run_path / "wandb" / "latest-run" / "files" / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

    argv = [
    "--task={}".format(config["task"]["value"]),
    "--feat={}".format(config["feat"]["value"]),
    "--backend={}".format(config["backend"]["value"]),

    "--run.from_checkpoint={}".format(str(run_path / "checkpoint.ckpt")),
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
    brax.envs.register_environment("antwall", antwall.AntWall)
    env = brax.envs.create(env_name=config.task + "wall",
                           episode_length=config.episode_length,
                           action_repeat=config_hydra.action_repeat,
                           auto_reset=True,
                           batch_size=config_hydra.env_batch_size,
                           backend=config.backend,
                           debug=True)
    env = feat_wrappers.VelocityWrapper(env, config.task)

    # Create agent
    env_space = from_brax.FromBraxVec(env, obs_key="vector", seed=config.seed, n_envs=config_hydra.env_batch_size)
    env_space = dreamerv3.wrap_env(env_space, config)
    agent = dreamerv3.Agent(env_space.obs_space, env_space.act_space, env_space.feat_space, step, config)
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=['agent'])
    policy = lambda *args: agent.policy(*args, mode='eval')

    class MetaEnv(envs.Env):
        def __init__(self, env):
            self.env = env
            self._state = None
            self._step_fn = jax.jit(self.env.step)
            self._reset_fn = jax.jit(self.env.reset)
            self._position_normalization = 50.

        def get_obs(self, state, action):
            return {
                "vector": state.obs[..., 1:],
                "is_first": state.info["steps"] == 0,
                "is_terminal": state.done == 1,
                "feat": state.info["feat"],
                "goal": action,
            }

        def reset(self, rng):
            state = self.env.reset(rng)
            return state.replace(obs=self._augment_obs(state))

        def step(self, state, action):
            obs = self.get_obs(state, self._scale_action(action))
            acts, self._state = policy(obs, self._state)
            next_state = self._step_fn(state.replace(obs=state.obs[..., 1:]), jnp.array(acts["action"]))
            return next_state.replace(
                obs=self._augment_obs(next_state),
                reward=next_state.reward + self._get_reward(state, next_state))

        def _scale_action(self, action):
            return 0.5 * (self.env.behavior_descriptor_limits[1] - self.env.behavior_descriptor_limits[0]) * action + \
                0.5 * (self.env.behavior_descriptor_limits[1] + self.env.behavior_descriptor_limits[0])

        def _augment_obs(self, state):
            return jnp.concatenate([state.pipeline_state.q[..., 0:1]/self._position_normalization, state.obs], axis=-1)

        def _get_reward(self, state, next_state):
            return 0.

        @property
        def action_size(self):
            return self.env.feat_size

        @property
        def observation_size(self):
            return self.env.observation_size + 1

        @property
        def backend(self):
            return self.env.backend

        @property
        def episode_length(self):
            return self.env.episode_length

    return MetaEnv(env)


def get_meta_env_dcg_me(config_hydra, run_path):
    run_path = Path(run_path)
    with open(run_path / ".hydra" / "config.yaml") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.load(run_path / ".hydra" / "config.yaml")

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = environments.create(config.task + "wall" + "_" + config.feat,
                              episode_length=config.algo.episode_length,
                              action_repeat=config_hydra.action_repeat,
                              batch_size=config_hydra.env_batch_size,
                              backend=config.algo.backend,)

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

    class MetaEnv(envs.Env):
        def __init__(self, env):
            self.env = env
            self._position_normalization = 50.

        def reset(self, rng):
            state = self.env.reset(rng)
            return state.replace(obs=self._augment_obs(state))

        def step(self, state, action):
            action = actor_dc_network.apply(actor_params, state.obs[..., 1:], action / env.behavior_descriptor_limits[1][0])
            next_state = self.env.step(state.replace(obs=state.obs[..., 1:]), action)
            return next_state.replace(
                obs=self._augment_obs(next_state),
                reward=next_state.reward + self._get_reward(state, next_state))

        def _scale_action(self, action):
            return 0.5 * (self.env.behavior_descriptor_limits[1] - self.env.behavior_descriptor_limits[0]) * action + \
                0.5 * (self.env.behavior_descriptor_limits[1] + self.env.behavior_descriptor_limits[0])

        def _augment_obs(self, state):
            return jnp.concatenate([state.pipeline_state.q[..., 0:1]/self._position_normalization, state.obs], axis=-1)

        def _get_reward(self, state, next_state):
            return 0.

        @property
        def action_size(self):
            return self.env.feat_size

        @property
        def observation_size(self):
            return self.env.observation_size + 1

        @property
        def backend(self):
            return self.env.backend

        @property
        def episode_length(self):
            return self.env.episode_length

    return MetaEnv(env)


def get_meta_env_qd_pg(config_hydra, run_path):
    run_path = Path(run_path)
    with open(run_path / ".hydra" / "config.yaml") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.load(run_path / ".hydra" / "config.yaml")

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = environments.create(config.task + "wall" + "_" + config.feat,
                              episode_length=config.algo.episode_length,
                              action_repeat=config_hydra.action_repeat,
                              batch_size=config_hydra.env_batch_size,
                              backend=config.algo.backend,)

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

    class MetaEnv(envs.Env):
        def __init__(self, env):
            self.env = env
            self._position_normalization = 50.

        def reset(self, rng):
            state = self.env.reset(rng)
            return state.replace(obs=self._augment_obs(state))

        def step(self, state, action):
            descriptor = self._scale_action(action)
            distances = jax.vmap(lambda x, y: jnp.linalg.norm(x - y, axis=-1), in_axes=(None, 0))(repertoire.descriptors, descriptor)
            indices = jax.vmap(jnp.argmin)(distances)
            params = jax.tree_util.tree_map(lambda x: x[indices], repertoire.genotypes)

            action = jax.vmap(policy_network.apply)(params, state.obs[..., 1:])
            next_state = self.env.step(state.replace(obs=state.obs[..., 1:]), action)
            return next_state.replace(
                obs=self._augment_obs(next_state),
                reward=next_state.reward + self._get_reward(state, next_state))

        def _scale_action(self, action):
            return 0.5 * (self.env.behavior_descriptor_limits[1] - self.env.behavior_descriptor_limits[0]) * action + \
                0.5 * (self.env.behavior_descriptor_limits[1] + self.env.behavior_descriptor_limits[0])

        def _augment_obs(self, state):
            return jnp.concatenate([state.pipeline_state.q[..., 0:1]/self._position_normalization, state.obs], axis=-1)

        def _get_reward(self, state, next_state):
            return 0.

        @property
        def action_size(self):
            return self.env.feat_size

        @property
        def observation_size(self):
            return self.env.observation_size + 1

        @property
        def backend(self):
            return self.env.backend

        @property
        def episode_length(self):
            return self.env.episode_length

    return MetaEnv(env)



def _get_agents_and_descs_ppga(config, archive, obs_shape, action_shape):
    agents = []
    descriptors = []
    device = torch.device('cuda')
    for elite in archive:
        desc = elite.measures
        agent = Actor(obs_shape=obs_shape[0], action_shape=action_shape, normalize_obs=config.algo.normalize_obs).deserialize(elite.solution).to(device)
        if config.algo.normalize_obs:
            agent.obs_normalizer.load_state_dict(elite.metadata['obs_normalizer'])
        agents.append(agent)
        descriptors.append(desc)
    agents = np.array(agents)
    repertoire_descriptors = jnp.asarray(descriptors)

    return agents, repertoire_descriptors

def get_meta_env_ppga(config_hydra, run_path):
    device = torch.device('cuda')
    run_path = Path(run_path)
    with open(run_path / ".hydra" / "config.yaml") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.load(run_path / ".hydra" / "config.yaml")

    from main_ppga import PPGAConfig
    cfg = PPGAConfig.create(config)
    cfg = cfg.as_dot_dict()

    cfg.num_emitters = 1

    cfg.batch_size = int(cfg.env_batch_size * cfg.rollout_length)
    cfg.num_envs = int(cfg.env_batch_size)

    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    

    if config.feat == "angle_notrigo":
        print("Angular space - USE NOTRIGO")
        config.feat = "angle"

    random_key = jax.random.PRNGKey(config.seed)
    # reevaluator = ReEvaluator(task_info.scoring_fn, num_reevals)

    solution_batch_size = 1000
    cfg.env_batch_size = solution_batch_size

    def get_env(hydra_config, cfg):
        if hydra_config.feat == "angle_notrigo":
            feat = "angle"
        else:
            feat = hydra_config.feat
        vec_env = make_vec_env_brax_ppga(task_name=hydra_config.task, feat_name=feat, batch_size=cfg.env_batch_size,
                                         seed=cfg.seed, backend=cfg.backend, clip_obs_rew=cfg.clip_obs_rew,
                                         episode_length=hydra_config.algo.episode_length)
        return vec_env

    vec_env = get_env(config, cfg)
    cfg.obs_shape = vec_env.single_observation_space.shape
    cfg.action_shape = vec_env.single_action_space.shape
    del vec_env

    # Init environment
    env = environments.create(config.task + "wall" + "_" + config.feat,
                              episode_length=config.algo.episode_length,
                              action_repeat=config_hydra.action_repeat,
                              batch_size=config_hydra.env_batch_size,
                              backend=config.algo.backend,)

    # Build the repertoire
    repertoire_ppga = load_repertoire_ppga(str(run_path) + '/')
    agents, repertoire_descriptors = _get_agents_and_descs_ppga(config, repertoire_ppga, cfg.obs_shape, cfg.action_shape)

    repertoire_descriptors = jnp.asarray(repertoire_descriptors)

    vec_inference = VectorizedActor(agents,
                        Actor,
                        obs_shape=cfg.obs_shape,
                        action_shape=cfg.action_shape,
                        normalize_obs=config.algo.normalize_obs).to(device)

    num_agents = len(agents)
    zeros_obs = torch.zeros(num_agents, *cfg.obs_shape).to(device)

    if config.algo.normalize_obs:
        repeats = 1
        obs_mean = [normalizer.obs_rms.mean for normalizer in vec_inference.obs_normalizers]
        obs_mean = torch.vstack(obs_mean).to(device)
        obs_mean = torch.repeat_interleave(obs_mean, dim=0, repeats=repeats)
        obs_var = [normalizer.obs_rms.var for normalizer in vec_inference.obs_normalizers]
        obs_var = torch.vstack(obs_var).to(device)
        obs_var = torch.repeat_interleave(obs_var, dim=0, repeats=repeats)
    import copy
    import time

    class MetaEnv(envs.Env):
        NUM_COUNT = 0

        def __init__(self, env):
            self.env = env
            self._position_normalization = 50.
            self._step_fn = jax.jit(self.env.step)

        def reset(self, rng):
            state = self.env.reset(rng)
            return state.replace(obs=self._augment_obs(state))

        def step(self, state, action_jax):
            # print("Step", self.NUM_COUNT)
            self.NUM_COUNT += 1

            obs_jax = state.obs[..., 1:]
            time1 = time.time()
            descriptor = self._scale_action(action_jax)
            distances = jax.vmap(lambda x, y: jnp.linalg.norm(x - y, axis=-1), in_axes=(None, 0))(repertoire_descriptors, descriptor)
            indices = jax.vmap(jnp.argmin)(distances)
            indices = np.asarray(indices)
            indices_torch = torch.from_numpy(indices).to(device)
            # print(f"time2", time.time() - time1)

            # selected_agents = agents[indices]
            # selected_obs_mean = obs_mean[indices]
            # selected_obs_var = obs_var[indices]

            # Selecting torch agents
            obs_torch = jax_to_torch(obs_jax)
            # print(f"time3", time.time() - time1)

            # copy torch tensor
            full_obs = copy.deepcopy(zeros_obs)
            full_obs[indices_torch] = obs_torch

            if config.algo.normalize_obs:
                obs_torch = (full_obs - obs_mean) / torch.sqrt(obs_var + 1e-8)
            else:
                obs_torch = full_obs

            # res = [torch.unsqueeze(agent(obs_torch[i]), axis=0) for i, agent in enumerate(selected_agents)]
            # acts_torch = torch.cat(res)
            # print(f"time4", time.time() - time1)

            acts_torch = vec_inference(obs_torch)
            # print(f"time5", time.time() - time1)


            action_jax = torch_to_jax(acts_torch)
            action_jax = action_jax[indices]
            action_jax = jnp.asarray(action_jax, dtype=jnp.float32)

            # print(f"time6", time.time() - time1)

            next_state = self._step_fn(state.replace(obs=state.obs[..., 1:]), action_jax)
            # print(f"time7", time.time() - time1)
            return next_state.replace(
                obs=self._augment_obs(next_state),
                reward=next_state.reward + self._get_reward(state, next_state))

        def _scale_action(self, action):
            return 0.5 * (self.env.behavior_descriptor_limits[1] - self.env.behavior_descriptor_limits[0]) * action + \
                0.5 * (self.env.behavior_descriptor_limits[1] + self.env.behavior_descriptor_limits[0])

        def _augment_obs(self, state):
            return jnp.concatenate([state.pipeline_state.q[..., 0:1]/self._position_normalization, state.obs], axis=-1)

        def _get_reward(self, state, next_state):
            return 0.

        @property
        def action_size(self):
            return self.env.feat_size

        @property
        def observation_size(self):
            return self.env.observation_size + 1

        @property
        def backend(self):
            return self.env.backend

        @property
        def episode_length(self):
            return self.env.episode_length

    return MetaEnv(env)



def get_meta_env_scopa(config_hydra, run_path):
    run_path = Path(run_path)
    with open(run_path / ".hydra" / "config.yaml") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.load(run_path / ".hydra" / "config.yaml")

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = environments.create(config.task + "wall" + "_" + config.feat,
                              episode_length=config.algo.episode_length,
                              action_repeat=config_hydra.action_repeat,
                              batch_size=config_hydra.env_batch_size,
                              backend=config.algo.backend,)

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

    class MetaEnv(envs.Env):
        def __init__(self, env):
            self.env = env
            self._position_normalization = 50.

        def reset(self, rng):
            state = self.env.reset(rng)
            return state.replace(obs=self._augment_obs(state))

        def step(self, state, action):
            dist_params = actor_network.apply(actor_params, state.obs[..., 1:], action)
            action = jax.nn.tanh(dist_params[..., : dist_params.shape[-1] // 2])
            next_state = self.env.step(state.replace(obs=state.obs[..., 1:]), action)
            return next_state.replace(
                obs=self._augment_obs(next_state),
                reward=next_state.reward + self._get_reward(state, next_state))

        def _augment_obs(self, state):
            return jnp.concatenate([state.pipeline_state.q[..., 0:1]/self._position_normalization, state.obs], axis=-1)

        def _get_reward(self, state, next_state):
            return 0.

        @property
        def action_size(self):
            return self.env.feat_size

        @property
        def observation_size(self):
            return self.env.observation_size + 1

        @property
        def backend(self):
            return self.env.backend

        @property
        def episode_length(self):
            return self.env.episode_length

    return MetaEnv(env)


def get_meta_env_domino(config_hydra, run_path):
    run_path = Path(run_path)
    with open(run_path / ".hydra" / "config.yaml") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.load(run_path / ".hydra" / "config.yaml")

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = environments.create(config.task + "wall" + "_" + config.feat,
                              episode_length=config.algo.episode_length,
                              action_repeat=config_hydra.action_repeat,
                              batch_size=config_hydra.env_batch_size,
                              backend=config.algo.backend,)

    num_policies = config.algo.num_skills

    # Init policy network
    policy_layer_sizes = config.algo.hidden_layer_sizes + (env.action_size,)
    # actor_network = MLP(
    #     layer_sizes=policy_layer_sizes,
    #     kernel_init=jax.nn.initializers.lecun_uniform(),
    #     final_activation=jnp.tanh,
    # )

    # import make_domino_networks
    from baselines.qdax.baselines.domino_networks import make_domino_networks 

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
        print("state_dict", state_dict)
    actor_params = serialization.from_state_dict(fake_actor_params, state_dict)
    # print("actor_params", actor_params)
    # print("shapes", jax.tree_map(lambda x: x.shape, actor_params))
    

    class MetaEnv(envs.Env):
        def __init__(self, env):
            self.env = env
            self._position_normalization = 50.

        def reset(self, rng):
            state = self.env.reset(rng)
            return state.replace(obs=self._augment_obs(state))

        def step(self, state, action):

            # action is a discrete value of the form of a one-hot vector
            matrix_actions = jax.vmap(_policy.apply, in_axes=(0, None), out_axes=1)(actor_params, state.obs[..., 2:])
            matrix_actions = jax.nn.tanh(matrix_actions[..., : matrix_actions.shape[-1] // 2])
            print(matrix_actions.shape, action.shape)
            action = jax.vmap(lambda x,y: jnp.matmul(jnp.transpose(x), y.reshape(-1, 1)).ravel())(matrix_actions, action)

            # action = actor_dc_network.apply(actor_params, state.obs[..., 1:], action / env.behavior_descriptor_limits[1][0])
            next_state = self.env.step(state.replace(obs=state.obs[..., 2:]), action)
            return next_state.replace(
                obs=self._augment_obs(next_state),
                reward=next_state.reward + self._get_reward(state, next_state))

        def _augment_obs(self, state):
            return jnp.concatenate([state.pipeline_state.q[..., 0:2]/self._position_normalization, state.obs], axis=-1)

        def _get_reward(self, state, next_state):
            return 0.

        @property
        def action_size(self):
            return num_policies

        @property
        def observation_size(self):
            return self.env.observation_size + 2

        @property
        def backend(self):
            return self.env.backend

        @property
        def episode_length(self):
            return self.env.episode_length

    return MetaEnv(env)



@hydra.main(version_base="1.2", config_path="configs/", config_name="adaptation_wall")
def main(config: Config) -> None:
    wandb.init(
        config=OmegaConf.to_container(config, resolve=True),
        project="QDAC",
        name="adaptation_wall",
    )

    os.mkdir("./actor/")

    path = config.path

    if config.algo_name == "qdac_mb":
        env = get_meta_env_ours(config, path)
    elif config.algo_name == "smerl":
        env = get_meta_env_smerl(config, path)
    elif config.algo_name == "smerl_reverse":
        env = get_meta_env_smerl_reverse(config, path)
    elif config.algo_name == "uvfa":
        env = get_meta_env_uvfa(config, path)
    elif config.algo_name == "dcg_me":
        env = get_meta_env_dcg_me(config, path)
    elif config.algo_name == "qd_pg":
        env = get_meta_env_qd_pg(config, path)
    elif config.algo_name == "qdac":
        env = get_meta_env_scopa(config, path)
    elif config.algo_name == "domino":
        env = get_meta_env_domino(config, path)
    elif config.algo_name == "ppga":
        env = get_meta_env_ppga(config, path)
    else:
        raise NotImplementedError

    # env = environments.create("ant_velocity",
    #                           episode_length=1000,
    #                           batch_size=256,
    #                           backend="spring",)

    reset_fn = jax.jit(env.reset)

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init SAC config
    sac_config = SacConfig(
        batch_size=config.batch_size,
        episode_length=env.episode_length,
        tau=config.soft_tau_update,
        normalize_observations=config.normalize_observations,
        learning_rate=config.learning_rate,
        alpha_init=config.alpha_init,
        discount=config.discount,
        reward_scaling=config.reward_scaling,
        hidden_layer_sizes=config.hidden_layer_sizes,
        fix_alpha=config.fix_alpha,
    )

    # Init SAC
    if config.algo_name in ("domino", "domino_joint_state", "smerl_full_state", "smerl_reverse_full_state"):
        sac = SACDiscrete(config=sac_config, action_size=env.action_size)
    else:
        sac = SAC(config=sac_config, action_size=env.action_size)
    random_key, subkey = jax.random.split(random_key)
    training_state = sac.init(subkey, env.action_size, env.observation_size)

    # Play step functions
    if config.algo_name in ("qdac_mb", "uvfa", "ppga"):
        play_step = functools.partial(
            sac.play_step_no_jit_fn,
            env=env,
            deterministic=False,
        )
        play_warmup_step = functools.partial(
            sac.play_step_no_jit_fn,
            env=env,
            deterministic=False,
        )
        play_eval_step = functools.partial(
            sac.play_step_no_jit_fn,
            env=env,
            deterministic=True,
        )
        eval_policy = functools.partial(
            sac.eval_policy_no_jit_fn,
            play_step_fn=play_eval_step,
        )
    else:
        play_step = functools.partial(
            sac.play_step_fn,
            env=env,
            deterministic=False,
        )
        play_eval_step = functools.partial(
            sac.play_step_fn,
            env=env,
            deterministic=True,
        )
        eval_policy = functools.partial(
            sac.eval_policy_fn,
            play_step_fn=play_eval_step,
        )

    # Init replay buffer
    dummy_transition = QDTransition.init_dummy(
        observation_dim=env.observation_size,
        action_dim=env.action_size,
        descriptor_dim=env.action_size,
    )
    replay_buffer = ReplayBuffer.init(
        buffer_size=config.replay_buffer_size, transition=dummy_transition
    )

    # Iterations
    if config.algo_name == "qdac_mb" or config.algo_name == "uvfa" or config.algo_name == "ppga":
        random_key, subkey = jax.random.split(random_key)
        env_state = env.reset(subkey)
        replay_buffer, _, training_state = warmstart_buffer_no_jit(
            replay_buffer=replay_buffer,
            training_state=training_state,
            env_state=env_state,
            num_warmstart_steps=config.warmup_steps,
            env_batch_size=config.env_batch_size,
            play_step_fn=play_warmup_step,
        )
        do_iteration = functools.partial(
            do_iteration_no_jit_fn,
            env_batch_size=config.env_batch_size,
            grad_updates_per_step=config.grad_updates_per_step,
            play_step_fn=play_step,
            update_fn=sac.update,
        )
    else:
        random_key, subkey = jax.random.split(random_key)
        env_state = env.reset(subkey)
        replay_buffer, _, training_state = warmstart_buffer(
            replay_buffer=replay_buffer,
            training_state=training_state,
            env_state=env_state,
            num_warmstart_steps=config.warmup_steps,
            env_batch_size=config.env_batch_size,
            play_step_fn=play_step,
        )
        do_iteration = functools.partial(
            do_iteration_fn,
            env_batch_size=config.env_batch_size,
            grad_updates_per_step=config.grad_updates_per_step,
            play_step_fn=play_step,
            update_fn=sac.update,
        )

    metrics = dict.fromkeys(["iteration", "episode_score", "episode_length", "actor_loss", "critic_loss", "alpha_loss", "obs_std", "obs_mean"], jnp.array([]))
    csv_logger = CSVLogger(
        str(Path(path) / "log.csv"),
        header=list(metrics.keys())
    )

    for i in range(config.num_iterations):
        training_state, env_state, replay_buffer, metrics = do_iteration(
            training_state=training_state,
            env_state=env_state,
            replay_buffer=replay_buffer,
        )

        if i % config.log_period == 0:
            random_key, subkey = jax.random.split(random_key)
            env_state_eval = reset_fn(subkey)
            true_return, _, episode_length_mean = eval_policy(training_state, env_state_eval)

            metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), metrics)
            metrics["iteration"] = i
            metrics["episode_score"] = true_return
            metrics["episode_length"] = episode_length_mean
            csv_logger.log(metrics)

    # Actor
    state_dict = serialization.to_state_dict(training_state.policy_params)
    with open("./actor/actor.pickle", "wb") as params_file:
        pickle.dump(state_dict, params_file)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main", node=Config)
    main()
