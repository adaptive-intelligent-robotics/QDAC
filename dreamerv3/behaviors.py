import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from . import agent
from . import expl
from . import ninjax as nj
from . import jaxutils


class Greedy(nj.Module):

  def __init__(self, wm, act_space, feat_space, config):
    rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
    feat_fn = lambda s: wm.heads['feat'](s).mean()[1:]

    resolution = agent.ImagActorCritic.get_resolution(feat_space, config)
    delta_constraint = agent.ImagActorCritic.calculate_delta_constraint(resolution=resolution, feat_space=feat_space)

    beta, environment_offset = agent.ImagActorCritic.get_beta_and_offset(feat_space=feat_space, resolution=resolution)

    rew_constraint_fn = lambda s: (environment_offset - jnp.linalg.norm(wm.heads['feat'](s).mean() - s["goal"], axis=-1)[1:]) / beta

    num_feats = wm.num_feats

    def _rewfn_uvfa(s):
      print("wm.heads['feat'](s).mean()", wm.heads['feat'](s).mean().shape)
      print("s['goal']", s['goal'].shape)
      print("wm.heads['reward'](s).mean()[1:]", wm.heads['reward'](s).mean()[1:].shape)
      print("jnp.linalg.norm(wm.heads['feat'](s).mean() - s['goal'], axis=-1)[1:]", jnp.linalg.norm(wm.heads['feat'](s).mean() - s["goal"], axis=-1)[1:].shape)

      distance = jnp.linalg.norm(wm.heads['feat'](s).mean() - s["goal"], axis=-1)[1:]
      rewards = wm.heads['reward'](s).mean()[1:]
      lambda_value_uvfa = config.goal.fixed_lagrangian_coeff
      print("lambda_value_uvfa", lambda_value_uvfa)
      return (1. - lambda_value_uvfa) * rewards - lambda_value_uvfa * distance

    if config.critic_type == 'sf_v_function':
      critics = {'extr': agent.VFunction(rewfn, config, name='critic'),
                 'sf': agent.SFFunction(feat_fn, num_feats, config, name='sf')}
      scales = {'extr': 1.0, 'sf': 1.0}  # TODO: should scales be 0.5 here?
    elif config.critic_type == 'constraint_v_function':
      critics = {'extr': agent.VFunction(rewfn, config, name='critic'),
                 'constraint': agent.VFunction(rew_constraint_fn, config, name='constraint')}
      scales = {'extr': 1.0, 'constraint': 1.0}  # TODO: add sf for logging purposes only
    elif config.critic_type == 'uvfa_critic_type':
      lambda_value_uvfa = config.goal.fixed_lagrangian_coeff
      # rewfn_uvfa = lambda s: (1. - lambda_value_uvfa) * wm.heads['reward'](s).mean()[1:] - lambda_value_uvfa * jnp.linalg.norm(wm.heads['feat'](s).mean() - s["goal"], axis=-1)[1:]
      critics = {'uvfa': agent.VFunction(_rewfn_uvfa, config, name='critic'),
                 'sf': agent.SFFunction(feat_fn, num_feats, config, name='sf'),  # SF is there for logging purposes only
                 }

      scales = {'uvfa': 1.0, 'sf': 0.0}
    else:
      raise NotImplementedError(config.critic_type)
    self.ac = agent.ImagActorCritic(
        critics, scales, act_space, feat_space, config, name='ac')

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    return self.ac.train(imagine, start, data)

  def report(self, data):
    return {}


class Random(nj.Module):

  def __init__(self, wm, act_space, config):
    self.config = config
    self.act_space = act_space

  def initial(self, batch_size):
    return jnp.zeros(batch_size)

  def policy(self, latent, state):
    batch_size = len(state)
    shape = (batch_size,) + self.act_space.shape
    if self.act_space.discrete:
      dist = jaxutils.OneHotDist(jnp.zeros(shape))
    else:
      dist = tfd.Uniform(-jnp.ones(shape), jnp.ones(shape))
      dist = tfd.Independent(dist, 1)
    return {'action': dist}, state

  def train(self, imagine, start, data):
    return None, {}

  def report(self, data):
    return {}


class Explore(nj.Module):

  REWARDS = {
      'disag': expl.Disag,
  }

  def __init__(self, wm, act_space, config):
    self.config = config
    self.rewards = {}
    critics = {}
    for key, scale in config.expl_rewards.items():
      if not scale:
        continue
      if key == 'extr':
        rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
        critics[key] = agent.VFunction(rewfn, config, name=key)
      else:
        rewfn = self.REWARDS[key](
            wm, act_space, config, name=key + '_reward')
        critics[key] = agent.VFunction(rewfn, config, name=key)
        self.rewards[key] = rewfn
    scales = {k: v for k, v in config.expl_rewards.items() if v}
    self.ac = agent.ImagActorCritic(
        critics, scales, act_space, config, name='ac')

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    metrics = {}
    for key, rewfn in self.rewards.items():
      mets = rewfn.train(data)
      metrics.update({f'{key}_k': v for k, v in mets.items()})
    traj, mets = self.ac.train(imagine, start, data)
    metrics.update(mets)
    return traj, metrics

  def report(self, data):
    return {}
