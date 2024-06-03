import functools
from typing import Any, Callable, List, Optional, Union

import brax
from brax.envs.base import Env
from brax.envs.wrappers import training

from baselines.qdax.environments.base_wrappers import QDEnv, StateDescriptorResetWrapper
from baselines.qdax.environments.bd_extractors import get_feat_mean
from dreamerv3.embodied.core.feat_wrappers import (
    FeetContactWrapper,
    VelocityWrapper,
    JumpWrapper,
    AngleWrapper,
    FullStateWrapper,
    JointStateWrapper,
)
from dreamerv3.embodied.envs import from_brax, antwall, humanoidwall, humanoidhurdle, humanoidgravity, antgravity, walker2dgravity, walker2dfriction
from dreamerv3.embodied.core.wrappers import ActuatorFailure
from baselines.qdax.environments.init_state_wrapper import FixedInitialStateWrapper
from baselines.qdax.environments.wrappers import CompletedEvalWrapper, TimeAwarenessWrapper, OffsetWrapper, ClipRewardWrapper, FeatToStateDescriptorWrapper


# experimentally determinated offset
# should be sufficient to have only positive rewards but no guarantee
reward_offset = {
    "hopper_uni": 0.9,
    "walker2d_uni": 1.413,
    "halfcheetah_uni": 9.231,
    "ant_uni": 3.24,
    "humanoid_uni": 0.0,
    "ant_omni": 3.0,
    "humanoid_omni": 0.0,
    "anttrap": 3.38,
}


brax.envs.register_environment("antwall", antwall.AntWall)
brax.envs.register_environment("humanoidhurdle", humanoidhurdle.HumanoidHurdle)
brax.envs.register_environment("humanoidgravity", humanoidgravity.HumanoidGravity)
brax.envs.register_environment("antgravity", antgravity.AntGravity)
brax.envs.register_environment("walker2dgravity", walker2dgravity.Walker2dGravity)
brax.envs.register_environment("walker2dfriction", walker2dfriction.Walker2dFriction)
_qdax_custom_envs = {
    # feet_contact
    "hopper_feet_contact": {
        "env": "hopper",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}],
    },
    "walker2d_feet_contact": {
        "env": "walker2d",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}],
    },
    "halfcheetah_feet_contact": {
        "env": "halfcheetah",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}],
    },
    "ant_feet_contact": {
        "env": "ant",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}]
    },
    "humanoid_feet_contact": {
        "env": "humanoid",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}],
    },

    # jump
    "hopper_jump": {
        "env": "hopper",
        "wrappers": [JumpWrapper],
        "kwargs": [{}],
    },
    "walker2d_jump": {
        "env": "walker2d",
        "wrappers": [JumpWrapper],
        "kwargs": [{}],
    },
    "halfcheetah_jump": {
        "env": "halfcheetah",
        "wrappers": [JumpWrapper],
        "kwargs": [{}],
    },
    "ant_jump": {
        "env": "ant",
        "wrappers": [JumpWrapper],
        "kwargs": [{}]
    },
    "humanoid_jump": {
        "env": "humanoid",
        "wrappers": [JumpWrapper],
        "kwargs": [{}],
    },

    # velocity
    "hopper_velocity": {
        "env": "hopper",
        "wrappers": [VelocityWrapper],
        "kwargs": [{}],
    },
    "walker2d_velocity": {
        "env": "walker2d",
        "wrappers": [VelocityWrapper],
        "kwargs": [{}],
    },
    "halfcheetah_velocity": {
        "env": "halfcheetah",
        "wrappers": [VelocityWrapper],
        "kwargs": [{}],
    },
    "ant_velocity": {
        "env": "ant",
        "wrappers": [VelocityWrapper],
        "kwargs": [{}]
    },
    "humanoid_velocity": {
        "env": "humanoid",
        "wrappers": [VelocityWrapper],
        "kwargs": [{}],
    },

    # angle
    "hopper_angle": {
        "env": "hopper",
        "wrappers": [AngleWrapper],
        "kwargs": [{}],
    },
    "walker2d_angle": {
        "env": "walker2d",
        "wrappers": [AngleWrapper],
        "kwargs": [{}],
    },
    "halfcheetah_angle": {
        "env": "halfcheetah",
        "wrappers": [AngleWrapper],
        "kwargs": [{}],
    },
    "ant_angle": {
        "env": "ant",
        "wrappers": [AngleWrapper],
        "kwargs": [{}]
    },
    "humanoid_angle": {
        "env": "humanoid",
        "wrappers": [AngleWrapper],
        "kwargs": [{}],
    },

    "hopper_angle_notrigo": {
        "env": "hopper",
        "wrappers": [AngleWrapper],
        "kwargs": [{"trigonometric": False}],
    },
    "walker2d_angle_notrigo": {
        "env": "walker2d",
        "wrappers": [AngleWrapper],
        "kwargs": [{"trigonometric": False}],
    },
    "halfcheetah_angle_notrigo": {
        "env": "halfcheetah",
        "wrappers": [AngleWrapper],
        "kwargs": [{"trigonometric": False}],
    },
    "ant_angle_notrigo": {
        "env": "ant",
        "wrappers": [AngleWrapper],
        "kwargs": [{"trigonometric": False}],
    },
    "humanoid_angle_notrigo": {
        "env": "humanoid",
        "wrappers": [AngleWrapper],
        "kwargs": [{"trigonometric": False}],
    },

    # Full State
    "hopper_full_state": {
        "env": "hopper",
        "wrappers": [FullStateWrapper],
        "kwargs": [{}],
    },
    "walker2d_full_state": {
        "env": "walker2d",
        "wrappers": [FullStateWrapper],
        "kwargs": [{}],
    },
    "halfcheetah_full_state": {
        "env": "halfcheetah",
        "wrappers": [FullStateWrapper],
        "kwargs": [{}],
    },
    "ant_full_state": {
        "env": "ant",
        "wrappers": [FullStateWrapper],
        "kwargs": [{}]
    },
    "humanoid_full_state": {
        "env": "humanoid",
        "wrappers": [FullStateWrapper],
        "kwargs": [{}],
    },

    # Joint State Wrapper
    "hopper_joint_state": {
        "env": "hopper",
        "wrappers": [JointStateWrapper],
        "kwargs": [{}],
    },
    "walker2d_joint_state": {
        "env": "walker2d",
        "wrappers": [JointStateWrapper],
        "kwargs": [{}],
    },
    "halfcheetah_joint_state": {
        "env": "halfcheetah",
        "wrappers": [JointStateWrapper],
        "kwargs": [{}],
    },
    "ant_joint_state": {
        "env": "ant",
        "wrappers": [JointStateWrapper],
        "kwargs": [{}]
    },
    "humanoid_joint_state": {
        "env": "humanoid",
        "wrappers": [JointStateWrapper],
        "kwargs": [{}],
    },



    # Failure
    "walker2dfailure_feet_contact": {
        "env": "walker2d",
        "wrappers": [FeetContactWrapper, ActuatorFailure],
        "kwargs": [{}, {}],
    },
    "antfailure_feet_contact": {
        "env": "ant",
        "wrappers": [FeetContactWrapper, ActuatorFailure],
        "kwargs": [{}, {}]
    },
    "humanoidfailure_feet_contact": {
        "env": "humanoid",
        "wrappers": [FeetContactWrapper, ActuatorFailure],
        "kwargs": [{}, {}],
    },

    # Failure Full State
    "walker2dfailure_full_state": {
        "env": "walker2d",
        "wrappers": [FullStateWrapper, ActuatorFailure],
        "kwargs": [{}, {}],
    },
    "antfailure_full_state": {
        "env": "ant",
        "wrappers": [FullStateWrapper, ActuatorFailure],
        "kwargs": [{}, {}]
    },
    "humanoidfailure_full_state": {
        "env": "humanoid",
        "wrappers": [FullStateWrapper, ActuatorFailure],
        "kwargs": [{}, {}],
    },

    # Failure Joint State Wrapper
    "walker2dfailure_joint_state": {
        "env": "walker2d",
        "wrappers": [JointStateWrapper, ActuatorFailure],
        "kwargs": [{}, {}],
    },
    "antfailure_joint_state": {
        "env": "ant",
        "wrappers": [JointStateWrapper, ActuatorFailure],
        "kwargs": [{}, {}]
    },
    "humanoidfailure_joint_state": {
        "env": "humanoid",
        "wrappers": [JointStateWrapper, ActuatorFailure],
        "kwargs": [{}, {}],
    },

    # Hurdles
    "humanoidhurdle_jump": {
        "env": "humanoidhurdle",
        "wrappers": [JumpWrapper],
        "kwargs": [{}],
    },
    ## Hurdles Full State
    "humanoidhurdle_full_state": {
        "env": "humanoidhurdle",
        "wrappers": [FullStateWrapper],
        "kwargs": [{}],
    },
    ## Hurdles Joint State Wrapper
    "humanoidhurdle_joint_state": {
        "env": "humanoidhurdle",
        "wrappers": [JointStateWrapper],
        "kwargs": [{}],
    },

    # Wall
    "antwall_velocity": {
        "env": "antwall",
        "wrappers": [VelocityWrapper],
        "kwargs": [{}]
    },
    ## Wall Full State
    "antwall_full_state": {
        "env": "antwall",
        "wrappers": [FullStateWrapper],
        "kwargs": [{}],
    },
    ## Wall Joint State Wrapper
    "antwall_joint_state": {
        "env": "antwall",
        "wrappers": [JointStateWrapper],
        "kwargs": [{}],
    },

    # Gravity
    "walker2dgravity_feet_contact": {
        "env": "walker2dgravity",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}],
    },
    "antgravity_feet_contact": {
        "env": "antgravity",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}]
    },
    "humanoidgravity_feet_contact": {
        "env": "humanoidgravity",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}],
    },
    "antgravity_velocity": {
        "env": "antgravity",
        "wrappers": [VelocityWrapper],
        "kwargs": [{}]
    },
    "humanoidgravity_jump": {
        "env": "humanoidgravity",
        "wrappers": [JumpWrapper],
        "kwargs": [{}],
    },

    ## Gravity Full State
    "walker2dgravity_full_state": {
        "env": "walker2dgravity",
        "wrappers": [FullStateWrapper],
        "kwargs": [{}],
    },
    "antgravity_full_state": {
        "env": "antgravity",
        "wrappers": [FullStateWrapper],
        "kwargs": [{}]
    },
    "humanoidgravity_full_state": {
        "env": "humanoidgravity",
        "wrappers": [FullStateWrapper],
        "kwargs": [{}],
    },

    ## Gravity Joint State Wrapper
    "walker2dgravity_joint_state": {
        "env": "walker2dgravity",
        "wrappers": [JointStateWrapper],
        "kwargs": [{}],
    },
    "antgravity_joint_state": {
        "env": "antgravity",
        "wrappers": [JointStateWrapper],
        "kwargs": [{}]
    },
    "humanoidgravity_joint_state": {
        "env": "humanoidgravity",
        "wrappers": [JointStateWrapper],
        "kwargs": [{}],
    },


    # Friction
    "walker2dfriction_feet_contact": {
        "env": "walker2dfriction",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}],
    },
    ## Friction Full State
    "walker2dfriction_full_state": {
        "env": "walker2dfriction",
        "wrappers": [FullStateWrapper],
        "kwargs": [{}],
    },
    ## Friction Joint State Wrapper
    "walker2dfriction_joint_state": {
        "env": "walker2dfriction",
        "wrappers": [JointStateWrapper],
        "kwargs": [{}],
    },
}


def create(
    env_name: str,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    eval_metrics: bool = False,
    fixed_init_state: bool = False,
    clip_reward: bool = False,
    qdax_wrappers_kwargs: Optional[List] = None,
    **kwargs: Any,
) -> Union[Env, QDEnv]:
    """Creates an Env with a specified brax system.
    Please use namespace to avoid confusion between this function and
    brax.envs.create.
    """
    if env_name in brax.envs._envs.keys():
        base_env_name = env_name
        env = brax.envs._envs[env_name](debug=True, **kwargs)
    elif env_name in _qdax_custom_envs.keys():
        # Create env
        base_env_name = _qdax_custom_envs[env_name]["env"]
        env = brax.envs._envs[base_env_name](debug=True, **kwargs)

        # Apply wrappers
        wrappers = _qdax_custom_envs[env_name]["wrappers"]
        if qdax_wrappers_kwargs is None:
            kwargs_list = _qdax_custom_envs[env_name]["kwargs"]
        else:
            kwargs_list = qdax_wrappers_kwargs
        for wrapper, kwargs in zip(wrappers, kwargs_list):  # type: ignore
            env = wrapper(env, base_env_name, **kwargs)  # type: ignore
    else:
        raise NotImplementedError("This environment name does not exist!")
    
    env = FeatToStateDescriptorWrapper(env)

    if episode_length is not None:
        env = training.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = training.VmapWrapper(env, batch_size)
    if auto_reset:
        env = training.AutoResetWrapper(env)
        if env_name in _qdax_custom_envs.keys():
            env = StateDescriptorResetWrapper(env)
    if fixed_init_state:
        env = FixedInitialStateWrapper(env, base_env_name)  # type: ignore
    if eval_metrics:
        env = training.EvalWrapper(env)
        env = CompletedEvalWrapper(env)
    if clip_reward:
        env = OffsetWrapper(env, base_env_name)
        env = ClipRewardWrapper(env)
    return env


def create_fn(env_name: str, **kwargs: Any) -> Callable[..., brax.envs.Env]:
    """Returns a function that when called, creates an Env.
    Please use namespace to avoid confusion between this function and
    brax.envs.create_fn.
    """
    return functools.partial(create, env_name, **kwargs)
