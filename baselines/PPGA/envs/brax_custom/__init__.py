# Copyright 2022 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import
"""Some example environments to help get started quickly with brax_custom."""

import functools
from typing import Callable, Optional, Type, Union, overload

import brax
from brax.envs import ant
from brax.envs import fast
from brax.envs import half_cheetah
from brax.envs import hopper
from brax.envs import humanoid
from brax.envs import inverted_double_pendulum
from brax.envs import inverted_pendulum
from brax.envs import pusher
from brax.envs import reacher
from brax.envs import walker2d
from brax.envs import wrappers
from brax.envs.base import Env, State, Wrapper, PipelineEnv
from brax.envs.wrappers import training

import gym

from baselines.PPGA.envs.brax_custom.custom_wrappers.reward_wrappers import TotalReward
from baselines.PPGA.envs.brax_custom.custom_wrappers.clip_wrappers import ActionClipWrapper, RewardClipWrapper, ObservationClipWrapper

# From QDax: experimentally determinated offset (except for antmaze)
# should be sufficient to have only positive rewards but no guarantee
reward_offset = {
    "ant": 3.24,
    "humanoid": 0.0,
    "halfcheetah": 9.231,
    "hopper": 0.9,
    "walker2d": 1.413,
}

_envs = {
    'ant': functools.partial(ant.Ant, use_contact_forces=True),
    'fast': fast.Fast,
    'halfcheetah': half_cheetah.Halfcheetah,
    'hopper': hopper.Hopper,
    'humanoid': humanoid.Humanoid,
    'inverted_pendulum': inverted_pendulum.InvertedPendulum,
    'inverted_double_pendulum': inverted_double_pendulum.InvertedDoublePendulum,
    'pusher': pusher.Pusher,
    'reacher': reacher.Reacher,
    'walker2d': walker2d.Walker2d,
}


def get_environment(env_name, **kwargs) -> Env:
    return _envs[env_name](**kwargs)


def register_environment(env_name: str, env_class: Type[Env]):
    _envs[env_name] = env_class


