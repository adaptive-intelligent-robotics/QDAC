import collections

import numpy as np
import jax


from .basics import convert


class Driver:

  _CONVERSION = {
      np.floating: np.float32,
      np.signedinteger: np.int32,
      np.uint8: np.uint8,
      bool: bool,
  }

  def __init__(self, env, goal_sampler, period_sample_goals, **kwargs):
    assert len(env) > 0
    self._env = env
    self._goal_sampler = goal_sampler
    self._period_sample_goals = period_sample_goals
    self._kwargs = kwargs
    self._on_steps = []
    self._on_episodes = []
    self._goals = None
    self.reset()

  def reset(self):
    self._acts = {
        k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
        for k, v in self._env.act_space.items()}
    self._acts['reset'] = np.ones(len(self._env), bool)
    self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
    self._state = None

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    assert all(len(x) == len(self._env) for x in self._acts.values())
    acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}
    old_obs = self._env.step(acts)
    obs = {k: convert(v) for k, v in old_obs.items() if k != 'pipeline_state'}
    
    # Sample goals
    eps_length = [len(ep['reward']) for ep in self._eps]
    is_goal_sampled = [eps_length[i] % self._period_sample_goals == 0 or obs['is_first'][i] for i in range(len(self._env))]
    self._goals = np.array([self._goal_sampler.sample() if is_goal_sampled[i] else self._goals[i] for i in range(len(self._env))])
    obs['goal'] = self._goals

    assert all(len(x) == len(self._env) for key, x in obs.items()), obs
    acts, self._state = policy(obs, self._state, **self._kwargs)
    acts = {k: convert(v) for k, v in acts.items()}
    if obs['is_last'].any():
      mask = 1 - obs['is_last']
      acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}
    acts['reset'] = obs['is_last'].copy()
    self._acts = acts
    trns = {**obs, **acts}
    if 'pipeline_state' in old_obs.keys():
      trns['pipeline_state'] = old_obs['pipeline_state']
    if obs['is_first'].any():
      for i, first in enumerate(obs['is_first']):
        if first:
          self._eps[i].clear()
    for i in range(len(self._env)):
      trn = {k: v[i] for k, v in trns.items() if k != 'pipeline_state'}
      if 'pipeline_state' in trns.keys():
        trn['pipeline_state'] = jax.tree_map(lambda x: x[i], trns['pipeline_state'])

      [self._eps[i][k].append(v) for k, v in trn.items()]
      [fn(trn, i, **self._kwargs) for fn in self._on_steps]
      step += 1
    if obs['is_last'].any():
      for i, done in enumerate(obs['is_last']):
        if done:
          ep = {k: convert(v) for k, v in self._eps[i].items() if k != 'pipeline_state'}
          if 'pipeline_state' in self._eps[i].keys():
            ep['pipeline_state'] = self._eps[i]['pipeline_state']
          [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]
          episode += 1
    return step, episode

  def _expand(self, value, dims):
    while len(value.shape) < dims:
      value = value[..., None]
    return value
