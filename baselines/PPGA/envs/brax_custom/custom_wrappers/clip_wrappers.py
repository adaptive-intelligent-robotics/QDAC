from brax.envs.base import Env, State, Wrapper
import jax.numpy as jnp


class ActionClipWrapper(Wrapper):
    def __init__(self, env: Env, a_min: float, a_max: float):
        super().__init__(env)
        self.a_min = jnp.array(a_min)
        self.a_max = jnp.array(a_max)

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        action = jnp.clip(action, self.a_min, self.a_max)
        nstate = self.env.step(state, action)
        return nstate


class ObservationClipWrapper(Wrapper):
    def __init__(self, env: Env, obs_min, obs_max):
        super().__init__(env)
        self.obs_min = jnp.array(obs_min)
        self.obs_max = jnp.array(obs_max)

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        obs = state.obs
        clipped_obs = jnp.clip(obs, self.obs_min, self.obs_max)
        state = state.replace(obs=clipped_obs)
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        nstate = self.env.step(state, action)
        obs = nstate.obs
        clipped_obs = jnp.clip(obs, self.obs_min, self.obs_max)
        nstate = nstate.replace(obs=clipped_obs)
        return nstate


class RewardClipWrapper(Wrapper):
    def __init__(self, env: Env, rew_min, rew_max):
        super().__init__(env)
        self.rew_min = jnp.array(rew_min)
        self.rew_max = jnp.array(rew_max)

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        rew = state.reward
        clipped_rew = jnp.clip(rew, self.rew_min, self.rew_max)
        state = state.replace(reward=clipped_rew)
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        nstate = self.env.step(state, action)
        rew = nstate.reward
        clipped_rew = jnp.clip(rew, self.rew_min, self.rew_max)
        nstate = nstate.replace(reward=clipped_rew)
        return nstate
