import gym
import numpy as np
import torch

from brax.envs.base import Env, State, Wrapper
import jax
import jax.numpy as jnp


class TotalReward(Wrapper):
    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        state.info['total_reward'] = jnp.zeros(self.env.batch_size)
        state.info['traj_length'] = jnp.zeros(self.env.batch_size)
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        nstate = self.env.step(state, action)
        if 'total_reward' in nstate.info:
            total_rew = nstate.info['total_reward']
            total_rew += nstate.reward
            state.info.update(total_reward=total_rew)
        if 'traj_length' in nstate.info:
            t = nstate.info['traj_length']
            t += jnp.ones(self.env.batch_size)
            state.info.update(traj_length=t)
        return nstate

