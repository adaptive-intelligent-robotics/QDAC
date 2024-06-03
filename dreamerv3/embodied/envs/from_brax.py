from functools import partial, cached_property

import embodied
import numpy as np
import jax
import jax.numpy as jnp


class FromBrax(embodied.Env):

  def __init__(self, env, obs_key="vector", act_key="action", seed: int = 0):
    self._env = env
    self._obs_key = obs_key
    self._act_key = act_key

    self._random_key = jax.random.PRNGKey(seed)
    self._random_key, random_key = jax.random.split(self._random_key)
    self._state = self._env.reset(random_key)
    self._done = jnp.ones(())

  @property
  def info(self):
    return {**self._state.metrics, **self._state.info}

  @cached_property
  def obs_space(self):
    return {
      "vector": embodied.Space(np.float32, (self._env.observation_size)),
      "reward": embodied.Space(np.float32),
      "is_first": embodied.Space(bool),
      "is_last": embodied.Space(bool),
      "is_terminal": embodied.Space(bool),
    }

  @cached_property
  def act_space(self):
    return {
      "action": embodied.Space(np.float32, (self._env.action_size,), low=-1., high=1.),
      "reset": embodied.Space(bool)
    }

  @partial(jax.jit, static_argnums=0)
  def _step_jit(self, action, action_reset, random_key, state, done):
    print("Tracing _step function.")

    pred = jnp.logical_or(action_reset, done)

    def true_fun():
      return self._env.reset(random_key)

    def false_fun():
      return self._env.step(state, action)

    return jax.lax.cond(pred, true_fun, false_fun)

  def step(self, action):
    self._random_key, random_key = jax.random.split(self._random_key)
    self._state = self._step_jit(action["action"], np.asarray(action["reset"], dtype=np.float32), random_key, self._state, self._done)
    self._done = self._state.done
    return self._obs()

  def _obs(self):
    obs = {self._obs_key: self._state.obs}
    obs = self._flatten(obs)
    obs = {k: np.asarray(v) for k, v in obs.items()}
    if "feat" in self._state.info:
      obs.update(
          reward=np.asarray(self._state.reward, dtype=np.float32),
          is_first=np.asarray(self._state.info["steps"] == 0, dtype=bool),
          is_last=np.asarray(self._state.done, dtype=bool),
          is_terminal=np.asarray(self._state.done, dtype=bool),
          feat=np.asarray(self._state.info["feat"], dtype=np.float32),)
    else:
      obs.update(
          reward=np.asarray(self._state.reward, dtype=np.float32),
          is_first=np.asarray(self._state.info["steps"] == 0, dtype=bool),
          is_last=np.asarray(self._state.done, dtype=bool),
          is_terminal=np.asarray(self._state.done, dtype=bool),)
    return obs

  def render(self):
    raise NotImplementedError

  def close(self):
    pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + "/" + key if prefix else key
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split("/")
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, "n"):
      return embodied.Space(np.int32, (), 0, space.n)
    return embodied.Space(space.dtype, space.shape, space.low, space.high)


class FromBraxVec(embodied.Env):

  def __init__(self, env, obs_key="vector", act_key="action", seed: int = 0, n_envs: int = 1, activate_pipeline_state: bool = False):
    assert n_envs > 0
    self._env = env
    self._obs_key = obs_key
    self._act_key = act_key
    self._n_envs = n_envs

    self._random_key = jax.random.PRNGKey(seed)
    self._random_key, random_key = jax.random.split(self._random_key)
    random_keys = jax.random.split(random_key, n_envs)
    self._state = jax.vmap(self._env.reset)(random_keys)
    self._done = jnp.ones((self._n_envs,))

    self._state_jit_vmap_fn = jax.jit(jax.vmap(self._step))

    self._activate_pipeline_state = activate_pipeline_state

  @property
  def sys(self):
    return self._env.sys
  
  @property
  def dt(self):
    return self._env.dt

  @property
  def info(self):
    return {**self._state.metrics, **self._state.info}

  @cached_property
  def obs_space(self):
    return {
      "vector": embodied.Space(np.float32, (self._env.observation_size)),
      "reward": embodied.Space(np.float32),
      "is_first": embodied.Space(bool),
      "is_last": embodied.Space(bool),
      "is_terminal": embodied.Space(bool),
      "feat": self.feat_space["vector"],
    }

  @cached_property
  def act_space(self):
    return {
      "action": embodied.Space(np.float32, (self._env.action_size,), low=-1., high=1.),
      "reset": embodied.Space(bool)
    }
  
  @cached_property
  def num_features(self):
    return self.feat_space["vector"].shape[0]
  
  @cached_property
  def feat_space(self):
    return self._env.feat_space

  def __len__(self):
    return self._n_envs

  def _step(self, action, action_reset, random_key, state, done):
    print("Tracing _step function.")

    pred = jnp.logical_or(action_reset, done)

    def true_fun():
      return self._env.reset(random_key)

    def false_fun():
      return self._env.step(state, action)

    return jax.lax.cond(pred, true_fun, false_fun)

  def step(self, action):
    self._random_key, random_key = jax.random.split(self._random_key)
    random_keys = jax.random.split(random_key, self._n_envs)
    self._state = self._state_jit_vmap_fn(action["action"], np.asarray(action["reset"], dtype=np.float32), random_keys, self._state, self._done)
    self._done = self._state.done
    return self._obs()

  def _obs(self):
    obs = {self._obs_key: self._state.obs}
    obs = self._flatten(obs)
    obs = {k: np.asarray(v) for k, v in obs.items()}
    if "feat" in self._state.info:
      obs.update(
          reward=np.asarray(self._state.reward, dtype=np.float32),
          is_first=np.asarray(self._state.info["steps"] == 0, dtype=bool),
          is_last=np.asarray(self._state.done, dtype=bool),
          is_terminal=np.asarray(self._state.done, dtype=bool),
          feat=np.asarray(self._state.info["feat"], dtype=np.float32),
          )
    else:
      obs.update(
          reward=np.asarray(self._state.reward, dtype=np.float32),
          is_first=np.asarray(self._state.info["steps"] == 0, dtype=bool),
          is_last=np.asarray(self._state.done, dtype=bool),
          is_terminal=np.asarray(self._state.done, dtype=bool),)
    if self._activate_pipeline_state:
      obs.update(
          pipeline_state=self._state.pipeline_state,
          )
    return obs

  def render(self):
    raise NotImplementedError

  def close(self):
    pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + "/" + key if prefix else key
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split("/")
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, "n"):
      return embodied.Space(np.int32, (), 0, space.n)
    return embodied.Space(space.dtype, space.shape, space.low, space.high)
