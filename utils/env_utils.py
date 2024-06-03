import warnings
from dataclasses import dataclass
from collections.abc import MutableMapping

import dreamerv3
from dreamerv3.embodied.envs import from_brax, antwall, humanoidwall, humanoidhurdle, humanoidgravity, antgravity, walker2dgravity, walker2dfriction
from dreamerv3.embodied.core import feat_wrappers, wrappers

import brax.envs
warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


# Hydra
@dataclass
class Config:
    seed: int

def flatten(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def get_argv_from_config(config):
   return ['--' + str(k) + '=' + str(v) for k, v in flatten(config, separator='.').items()]


# Env
def apply_feat_wrapper(env, config):
  if config.feat == "feet_contact":
    return feat_wrappers.FeetContactWrapper(env, config.task)
  elif config.feat == "velocity":
    return feat_wrappers.VelocityWrapper(env, config.task)
  elif config.feat == "jump":
    return feat_wrappers.JumpWrapper(env, config.task)
  elif config.feat == "angle":
    return feat_wrappers.AngleWrapper(env, config.task)
  elif config.feat == "velocityjump":
    return feat_wrappers.VelocityJumpWrapper(env, config.task)
  else:
    raise NotImplementedError(f"Feature {config.feat} not implemented")

def get_env(config, mode, index=None, actuator_failure_idx=None, actuator_failure_strength=None, gravity_coef=None, friction_index=None, **kwargs):
  if config.feat in ["feet_contact", "jump", "velocityjump"]:
    debug = True
  else:
    debug = False

  brax.envs.register_environment("antwall", antwall.AntWall)
  brax.envs.register_environment("humanoidhurdle", humanoidhurdle.HumanoidHurdle)
  brax.envs.register_environment("humanoidgravity", humanoidgravity.HumanoidGravity)
  brax.envs.register_environment("antgravity", antgravity.AntGravity)
  brax.envs.register_environment("walker2dgravity", walker2dgravity.Walker2dGravity)
  brax.envs.register_environment("walker2dfriction", walker2dfriction.Walker2dFriction)
  if index is not None:
    env = brax.envs.create(env_name="humanoidhurdle",
                          episode_length=config.episode_length,
                          auto_reset=False,
                          backend=config.backend,
                          debug=debug,
                          index=index,
                          **kwargs)
  elif friction_index is not None:
    env = brax.envs.create(env_name="walker2dfriction",
                          episode_length=config.episode_length,
                          auto_reset=False,
                          backend=config.backend,
                          debug=debug,
                          friction_index=friction_index,
                          **kwargs)
  elif gravity_coef is not None:
    env = brax.envs.create(env_name=config.task + "gravity",
                          episode_length=config.episode_length,
                          auto_reset=False,
                          backend=config.backend,
                          debug=debug,
                          gravity_coef=gravity_coef,
                          **kwargs)
  else:
    env = brax.envs.create(env_name=config.task,
                          episode_length=config.episode_length,
                          auto_reset=False,
                          backend=config.backend,
                          debug=debug,
                          **kwargs)
  if actuator_failure_idx:
    env = wrappers.ActuatorFailure(env, config.task, actuator_failure_idx, actuator_failure_strength)
  env = apply_feat_wrapper(env, config)

  if mode == "train":
    activate_pipeline_state = False
  elif mode == "eval":
    activate_pipeline_state = True
    assert config.run.from_checkpoint, "Must specify checkpoint to load from"
  else:
    raise NotImplementedError(f"Mode {mode} not implemented")

  env = from_brax.FromBraxVec(env, obs_key="vector", seed=config.seed, n_envs=config.envs.amount, activate_pipeline_state=activate_pipeline_state)
  env = dreamerv3.wrap_env(env, config)

  print("env:", config.task)
  print("n_envs:", config.envs.amount)
  print("feat:", config.feat)
  print("feat_space:", env.feat_space["vector"])

  return env
