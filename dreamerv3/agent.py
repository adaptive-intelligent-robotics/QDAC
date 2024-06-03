import numpy as np
import wandb

import embodied
import jax
import jax.numpy as jnp
import ruamel.yaml as yaml
import embodied.core.goal_sampler

tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
  def filter(self, record):
    return 'check_types' not in record.getMessage()
logger.addFilter(CheckTypesFilter())

from . import behaviors
from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj


@jaxagent.Wrapper
class Agent(nj.Module):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, feat_space, step, config):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.feat_space = feat_space
    self.step = step
    num_feats = feat_space['vector'].shape[0]
    self.wm = WorldModel(obs_space, act_space, num_feats, config, name='wm')
    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.wm, self.act_space, self.feat_space, self.config, name='task_behavior')
    if config.expl_behavior == 'None':
      self.expl_behavior = self.task_behavior
    else:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.wm, self.act_space, self.feat_space, self.config, name='expl_behavior')

  def policy_initial(self, batch_size):
    return (
        self.wm.initial(batch_size),
        self.task_behavior.initial(batch_size),
        self.expl_behavior.initial(batch_size))

  def train_initial(self, batch_size):
    return self.wm.initial(batch_size)

  def policy(self, obs, state, mode='train'):
    self.config.jax.jit and print('Tracing policy function.')
    obs = self.preprocess(obs)
    (prev_latent, prev_action), task_state, expl_state = state
    embed = self.wm.encoder(obs)
    latent, _ = self.wm.rssm.obs_step(
        prev_latent, prev_action, embed, obs['is_first'])
    latent_with_goal = {**latent, 'goal': obs['goal']}
    self.expl_behavior.policy(latent_with_goal, expl_state)
    task_outs, task_state = self.task_behavior.policy(latent_with_goal, task_state)
    expl_outs, expl_state = self.expl_behavior.policy(latent_with_goal, expl_state)
    if mode == 'eval':
      outs = task_outs
      outs['action'] = outs['action'].sample(seed=nj.rng())
      outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
    elif mode == 'explore':
      outs = expl_outs
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng())
    elif mode == 'train':
      outs = task_outs
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng())
    state = ((latent, outs['action']), task_state, expl_state)
    return outs, state

  def train(self, data, state):
    self.config.jax.jit and print('Tracing train function.')
    metrics = {}
    data = self.preprocess(data)
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)
    context = {**data, **wm_outs['post']}
    start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
    _, mets = self.task_behavior.train(self.wm.imagine, start, context)
    metrics.update(mets)
    if self.config.expl_behavior != 'None':
      _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    outs = {}
    return outs, state, metrics

  def report(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    report.update(self.wm.report(data))
    mets = self.task_behavior.report(data)
    report.update({f'task_{k}': v for k, v in mets.items()})
    if self.expl_behavior is not self.task_behavior:
      mets = self.expl_behavior.report(data)
      report.update({f'expl_{k}': v for k, v in mets.items()})
    return report

  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      else:
        value = value.astype(jnp.float32)
      obs[key] = value
    obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
    return obs


class WorldModel(nj.Module):

  def __init__(self, obs_space, act_space, num_feats, config):
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.config = config
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
    self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')
    self.rssm = nets.RSSM(**config.rssm, name='rssm')
    self.num_feats = num_feats
    self.heads = {
        'decoder': nets.MultiDecoder(shapes, **config.decoder, name='dec'),
        'reward': nets.MLP((), **config.reward_head, name='rew'),
        'cont': nets.MLP((), **config.cont_head, name='cont'),
        'feat': nets.MLP((num_feats,), **config.feat_head, name='feat'),
    }
    self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt)
    scales = self.config.loss_scales.copy()
    image, vector = scales.pop('image'), scales.pop('vector')
    scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
    scales.update({k: vector for k in self.heads['decoder'].mlp_shapes})
    self.scales = scales

  def initial(self, batch_size):
    prev_latent = self.rssm.initial(batch_size)
    prev_action = jnp.zeros((batch_size, *self.act_space.shape))
    return prev_latent, prev_action

  def train(self, data, state):
    modules = [self.encoder, self.rssm, *self.heads.values()]
    mets, (state, outs, metrics) = self.opt(
        modules, self.loss, data, state, has_aux=True)
    metrics.update(mets)
    return state, outs, metrics

  def loss(self, data, state):
    embed = self.encoder(data)
    prev_latent, prev_action = state
    prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1)
    post, prior = self.rssm.observe(
        embed, prev_actions, data['is_first'], prev_latent)
    dists = {}
    feats = {**post, 'embed': embed}
    for name, head in self.heads.items():
      out = head(feats if name in self.config.grad_heads else sg(feats))
      out = out if isinstance(out, dict) else {name: out}
      dists.update(out)
    losses = {}
    losses['dyn'] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
    losses['rep'] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
    for key, dist in dists.items():
      loss = -dist.log_prob(data[key].astype(jnp.float32))
      assert loss.shape == embed.shape[:2], (key, loss.shape)
      losses[key] = loss
    scaled = {k: v * self.scales[k] for k, v in losses.items()}
    model_loss = sum(scaled.values())
    out = {'embed':  embed, 'post': post, 'prior': prior}
    out.update({f'{k}_loss': v for k, v in losses.items()})
    last_latent = {k: v[:, -1] for k, v in post.items()}
    last_action = data['action'][:, -1]
    state = last_latent, last_action
    metrics = self._metrics(data, dists, post, prior, losses, model_loss)
    return model_loss.mean(), (state, out, metrics)

  def imagine(self, policy, start, horizon):
    first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
    keys = list(self.rssm.initial(1).keys())
    print("start keys", start.keys())
    start = {k: v for k, v in start.items() if k in (*keys, 'goal')}
    start['action'] = policy(start)
    def step(prev, _):
      prev = prev.copy()
      state = self.rssm.img_step(prev, prev.pop('action'))
      state["goal"] = start["goal"]
      return {**state, 'action': policy(state), 'goal': start['goal']}
    traj = jaxutils.scan(
        step, jnp.arange(horizon), start, self.config.imag_unroll)
    traj = {
        k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
    cont = self.heads['cont'](traj).mode()
    traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
    discount = 1 - 1 / self.config.horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
    return traj

  def report(self, data):
    state = self.initial(len(data['is_first']))
    report = {}
    report.update(self.loss(data, state)[-1][-1])
    context, _ = self.rssm.observe(
        self.encoder(data)[:6, :5], data['action'][:6, :5],
        data['is_first'][:6, :5])
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine(data['action'][:6, 5:], start))
    for key in self.heads['decoder'].cnn_shapes.keys():
      truth = data[key][:6].astype(jnp.float32)
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      error = (model - truth + 1) / 2
      video = jnp.concatenate([truth, model, error], 2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)
    return report

  def _metrics(self, data, dists, post, prior, losses, model_loss):
    entropy = lambda feat: self.rssm.get_dist(feat).entropy()
    metrics = {}
    metrics.update(jaxutils.tensorstats(entropy(prior), 'prior_ent'))
    metrics.update(jaxutils.tensorstats(entropy(post), 'post_ent'))
    metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics['model_loss_mean'] = model_loss.mean()
    metrics['model_loss_std'] = model_loss.std()
    metrics['reward_max_data'] = jnp.abs(data['reward']).max()
    metrics['feat_max_data'] = jnp.abs(data['feat']).max()
    metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()
    metrics['feat_max_pred'] = jnp.abs(dists['feat'].mean()).max()
    if 'reward' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
      metrics.update({f'reward_{k}': v for k, v in stats.items()})

    # if 'feat' in dists and not self.config.jax.debug_nans:
    #   print(dists['feat'], data['feat'].shape)
    #   stats = jaxutils.balance_stats(dists['feat'], data['feat'], 0.1)
    #   metrics.update({f'feat_{k}': v for k, v in stats.items()})

    if 'cont' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
      metrics.update({f'cont_{k}': v for k, v in stats.items()})
    return metrics


class ImagActorCritic(nj.Module):

  def __init__(self, critics, scales, act_space, feat_space, config):
    critics = {k: v for k, v in critics.items() if (scales[k] or k == "sf")}
    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if (scales[k] or k == "sf")}
    self.scales = scales
    self.act_space = act_space
    self.config = config
    disc = act_space.discrete
    self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
    self.actor = nets.MLP(
        name='actor', dims='deter', shape=act_space.shape, **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont)
    self.retnorms = {
        k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
        for k in critics}
    self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)

    if config.goal.use_fixed_lagrangian:
      fixed_lagrangian_coeff = config.goal.fixed_lagrangian_coeff
      assert 0.0 <= fixed_lagrangian_coeff <= 1.0
      self.fixed_lagrangian_coeff = fixed_lagrangian_coeff
    else:
      self.lagrangian = nets.MLP(shape=(), name='lagrangian', **config.lagrangian)
      self.opt_lagrangian = jaxutils.Optimizer(name='lagrangian_opt', **config.lagrangian_opt)

    self.goal_sampler = embodied.core.goal_sampler.GoalSampler(feat_space)

    resolution = self.get_resolution(feat_space, config)
    self.delta_constraint = self.calculate_delta_constraint(resolution=resolution, feat_space=feat_space)
    self.resolution = resolution
    self.feat_space = feat_space

    self.lagrangian_multiplicator = config.goal.lagrangian_multiplicator


  @staticmethod
  def get_beta_and_offset(resolution, feat_space: embodied.core.Space):
      delta_constraint = ImagActorCritic.calculate_delta_constraint(resolution=resolution, feat_space=feat_space)
      # beta = (feat_space['vector'].high[0] - feat_space['vector'].low[0]) / 2.
      # environment_offset = 2. * beta * delta_constraint

      beta = (feat_space['vector'].high[0] - feat_space['vector'].low[0]) / 4.
      environment_offset = (feat_space['vector'].high[0] - feat_space['vector'].low[0]) / 2.
      return beta, environment_offset

  @staticmethod
  def get_resolution(feat_space, config):

    # if resolution override is set, use it
    if config.goal.resolution > 0:
      return config.goal.resolution

    feat_space = feat_space["vector"]
    env_name = config.task
    feat_name = config.feat

    # Overriden resolutions
    if feat_name == "velocity":
      target_delta_ant_velocity = 0.1
      range = feat_space.high[0] - feat_space.low[0]
      resolution = int(range / (2 * target_delta_ant_velocity))
      return resolution
    elif feat_name == "jump":
      return 50

    # Default resolutions
    if feat_space.shape[0] == 1:
      return 50
    elif feat_space.shape[0] == 2:
      return 50
    elif feat_space.shape[0] == 3:
      return 10
    elif feat_space.shape[0] == 4:
      return 5
    elif feat_space.shape[0] == 5:
      return 4
    elif feat_space.shape[0] == 6:
      return 4
    else:
      raise NotImplementedError(f"feat_space.shape[0]={feat_space.shape[0]} not implemented")

  @staticmethod
  def calculate_delta_constraint(resolution, feat_space: embodied.core.Space):
    low = feat_space["vector"].low
    high = feat_space["vector"].high
    delta_array = (high - low) / (2. * resolution)
    delta_max = np.max(delta_array)
    return delta_max

  def initial(self, batch_size):
    return {}

  def policy(self, state, carry):
    return {'action': self.actor(state)}, carry

  def calculate_goal_coefficients(self, goals, sigmoid=True):
    if self.config.goal.use_fixed_lagrangian or self.config.critic_type == "uvfa_critic_type":
      return jnp.full(shape=goals.shape[:-1], fill_value=self.fixed_lagrangian_coeff)
    else:
      if not sigmoid:
        return self.lagrangian_multiplicator * self.lagrangian(goals).mean()
      else:
        return jax.nn.sigmoid(self.lagrangian_multiplicator * self.lagrangian(goals).mean())

  def train(self, imagine, start, context):
    batch_size = len(start['feat'])
    batch_goals = jax.vmap(self.goal_sampler.sample)(nj.rng(amount=batch_size))
    print("batch_goals", batch_goals)
    start = {**start, 'goal': batch_goals}

    def loss(start):
      policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
      traj = imagine(policy, start, self.config.imag_horizon)
      loss, metrics = self.loss(traj)
      return loss, (traj, metrics)

    def loss_lagrangian(start):
      policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
      traj = imagine(policy, start, self.config.imag_horizon)
      loss, metrics = self.loss_lagrangian(traj)
      return loss, (traj, metrics)

    metrics = {}
    mets, (traj, _new_metrics) = self.opt(self.actor, loss, start, has_aux=True)
    metrics.update(mets)
    metrics.update(_new_metrics)

    # Optimize lagrangian network
    if not self.config.goal.use_fixed_lagrangian \
        and not self.config.critic_type == "uvfa_critic_type":
      mets, (_, _new_metrics) = self.opt_lagrangian(self.lagrangian, loss_lagrangian, start, has_aux=True)
      metrics.update(mets)
      metrics.update(_new_metrics)

    array_goals = self.goal_sampler.get_goals_evaluation_metric()
    goals = array_goals.reshape(-1, array_goals.shape[-1])
    goal_coefficients = self.calculate_goal_coefficients(goals)
    goal_coefficients = goal_coefficients.reshape(*array_goals.shape[:-1])

    # goal_coefficients = jax.vmap(jax.vmap(lambda x,y: self.lagrangian(jnp.asarray([x,y])), in_axes=(0, None)), in_axes=(None, 0))(goal_x, goal_y)
    metrics['lagrangian_grid_coefficients'] = goal_coefficients

    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    return traj, metrics

  def loss_lagrangian(self, traj):
    metrics = {}
    discount = 1 - 1 / self.config.horizon
    delta = self.delta_constraint

    # Add the loss for the feature function
    if self.config.critic_type == 'sf_v_function':
      _, sf, _ = self.critics["sf"].score(sg(traj))
      rescaled_sf = sf * (1 - discount)
      distance_to_goal = jnp.linalg.norm(rescaled_sf - traj["goal"][:-1], axis=-1)
      is_constrain_satisfied = distance_to_goal < delta

    elif self.config.critic_type == 'constraint_v_function':
      _, v_constraint, _ = self.critics["constraint"].score(sg(traj))
      beta, environment_offset = ImagActorCritic.get_beta_and_offset(feat_space=self.feat_space, resolution=self.resolution)
      rescaled_v_constraint = v_constraint * (1 - discount)
      average_distance = environment_offset - beta * rescaled_v_constraint
      is_constrain_satisfied = average_distance < delta

    else:
      raise NotImplementedError

    is_constrain_satisfied = is_constrain_satisfied.astype(jnp.float32)

    labels = jax.lax.stop_gradient(is_constrain_satisfied)

    # Using cross-entropy loss

    coeff_no_sigmoid = self.calculate_goal_coefficients(traj["goal"][:-1], sigmoid=False)

    # loss = -1. * (jax.scipy.special.xlogy(1. - is_constrain_satisfied, scaled_coeff)
    #               + jax.scipy.special.xlogy(is_constrain_satisfied, 1. - scaled_coeff))

    logits = coeff_no_sigmoid
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)

    # Using cross-entropy loss
    loss = -1. * ((1. - labels) * log_p
                  + labels * log_not_p)

    loss *= sg(traj['weight'])[:-1]
    loss *= self.config.loss_scales.lagrangian

    metrics.update({'lagrangian_loss': loss.mean()})
    return loss.mean(), metrics

  def loss(self, traj):
    metrics = {}
    advs = []

    scaled_coeff = self.calculate_goal_coefficients(traj["goal"][:-1])

    if self.config.critic_type == 'sf_v_function':
      additional_scale = {key: 1.0 for key in self.critics.keys()}
    elif self.config.critic_type == 'constraint_v_function':
      additional_scale = {
        "constraint": scaled_coeff,
        "extr": 1.0 - scaled_coeff,
      }
    elif self.config.critic_type == 'uvfa_critic_type':
      additional_scale = {key: 1.0 for key in self.critics.keys()}
    else:
      raise NotImplementedError("Unknown critic type")
    
    total = sum(self.scales[k] * additional_scale[k] for k in self.critics)

    for key, critic in self.critics.items():
      if key in ('sf',):
        continue
      rew, ret, base = critic.score(traj, self.actor)
      offset, invscale = self.retnorms[key](ret)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      advs.append((normed_ret - normed_base) * self.scales[key] * additional_scale[key] / total)
      metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
    adv = jnp.stack(advs).sum(0)
    policy = self.actor(sg(traj))
    logpi = policy.log_prob(sg(traj['action']))[:-1]
    loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]

    # Add the loss for the feature function
    if self.config.critic_type == 'sf_v_function':
      _, sf, _ = self.critics["sf"].score(traj, self.actor)
      discount = 1 - 1 / self.config.horizon
      rescaled_sf = sf * (1 - discount)

      # distance_to_goal = jnp.linalg.norm(rescaled_sf - traj["goal"][:-1], axis=-1)
      # distance_to_goal = jnp.sum(jnp.square(rescaled_sf - traj["goal"][:-1]), axis=-1)
      delta_pseudo_huber_loss = 0.001
      distance_to_goal = delta_pseudo_huber_loss * (jnp.sqrt(1 + jnp.sum(jnp.square(rescaled_sf - traj["goal"][:-1]), axis=-1)/(delta_pseudo_huber_loss**2)) - 1)
      loss = (1. - scaled_coeff) * loss \
             + scaled_coeff * distance_to_goal
      # loss = loss + lagrangian_coeff * (distance_to_goal - delta)
      metrics.update({'mean_scaled_lagrangian_coeff': scaled_coeff.mean(),
                      'scaled_lagrangian_coeff_dist': scaled_coeff.ravel(),
                      'min_scaled_lagrangian_coeff': scaled_coeff.min(),
                      'max_scaled_lagrangian_coeff': scaled_coeff.max(),
                      'lagrangian_distance_to_goal': distance_to_goal.mean(),
                      'min_lagrangian_distance_to_goal': distance_to_goal.min(),
                      'max_lagrangian_distance_to_goal': distance_to_goal.max(),
                      'lagrangian_distance_to_goal_dist': distance_to_goal.ravel(),
                      })
    elif self.config.critic_type == 'constraint_v_function':
      _, v_constraint, _ = self.critics["constraint"].score(traj, self.actor)
      discount = 1 - 1 / self.config.horizon
      rescaled_v_constraint = v_constraint * (1 - discount)
      
      beta, environment_offset = ImagActorCritic.get_beta_and_offset(feat_space=self.feat_space, resolution=self.resolution)
      distance_to_goal = environment_offset - beta * rescaled_v_constraint
      # constraint_loss = distance_to_goal

      # Loss is already modified above.
      # loss = (1. - scaled_coeff) * loss \
            #  + scaled_coeff * constraint_loss
      metrics.update({'mean_scaled_lagrangian_coeff': scaled_coeff.mean(),
                      'scaled_lagrangian_coeff_dist': scaled_coeff.ravel(),
                      'min_scaled_lagrangian_coeff': scaled_coeff.min(),
                      'max_scaled_lagrangian_coeff': scaled_coeff.max(),
                      'lagrangian_distance_to_goal': distance_to_goal.mean(),
                      'min_lagrangian_distance_to_goal': distance_to_goal.min(),
                      'max_lagrangian_distance_to_goal': distance_to_goal.max(),
                      'lagrangian_distance_to_goal_dist': distance_to_goal.ravel(),
                      })
    elif self.config.critic_type == 'uvfa_critic_type':
      _, sf, _ = self.critics["sf"].score(traj, self.actor)
      discount = 1 - 1 / self.config.horizon
      rescaled_sf = sf * (1 - discount)

      distance_to_goal = jnp.linalg.norm(rescaled_sf - traj["goal"][:-1], axis=-1)

      # WARNING: HERE WE DON'T MODIFY THE LOSS, we just use the SF for logging purposes
      # loss = (1. - scaled_coeff) * loss \
      #        + scaled_coeff * distance_to_goal

      metrics.update({'mean_scaled_lagrangian_coeff': scaled_coeff.mean(),
                      'scaled_lagrangian_coeff_dist': scaled_coeff.ravel(),
                      'min_scaled_lagrangian_coeff': scaled_coeff.min(),
                      'max_scaled_lagrangian_coeff': scaled_coeff.max(),
                      'lagrangian_distance_to_goal': distance_to_goal.mean(),
                      'min_lagrangian_distance_to_goal': distance_to_goal.min(),
                      'max_lagrangian_distance_to_goal': distance_to_goal.max(),
                      'lagrangian_distance_to_goal_dist': distance_to_goal.ravel(),
                      })
    else:
      raise NotImplementedError

    ent = policy.entropy()[:-1]
    loss -= self.config.actent * ent

    loss *= sg(traj['weight'])[:-1]
    loss *= self.config.loss_scales.actor
    metrics.update(self._metrics(traj, policy, logpi, ent, adv))
    return loss.mean(), metrics

  def _metrics(self, traj, policy, logpi, ent, adv):
    metrics = {}
    ent = policy.entropy()[:-1]
    rand = (ent - policy.minent) / (policy.maxent - policy.minent)
    rand = rand.mean(range(2, len(rand.shape)))
    act = traj['action']
    act = jnp.argmax(act, -1) if self.act_space.discrete else act
    metrics.update(jaxutils.tensorstats(act, 'action'))
    metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
    metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
    metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return metrics


class VFunction(nj.Module):

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), name='net', dims='deter', **self.config.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.config.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.critic_opt)

  def train(self, traj, actor):
    target = sg(self.score(traj)[1])
    mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics

  def loss(self, traj, target):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(self.slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None):
    rew = self.rewfn(traj)
    assert len(rew) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    value = self.net(traj).mean()
    vals = [value[-1]]
    interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    return rew, ret, value[:-1]


class SFFunction(nj.Module):
  def __init__(self, feat_fn, num_feats, config):
    self.feat_fn = feat_fn
    self.config = config
    self.net = nets.MLP((num_feats,), name='net', dims='deter', **self.config.sf)
    self.slow = nets.MLP((num_feats,), name='slow', dims='deter', **self.config.sf)
    self.updater = jaxutils.SlowUpdater(
      self.net, self.slow,
      self.config.slow_sf_fraction,
      self.config.slow_sf_update)
    self.opt = jaxutils.Optimizer(name='sf_opt', **self.config.sf_opt)

  def train(self, traj, actor):
    target = sg(self.score(traj)[1])
    args = traj, target
    mets, metrics = self.opt(self.net, self.loss, *args, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics

  def loss(self, traj, target):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
        '...i,...i->...',
        sg(self.slow(traj).probs),
        jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.sf
    metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None):
    feature = self.feat_fn(traj)
    assert len(feature) == len(traj['action']) - 1, (
      'should provide rewards for all but last action')
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    disc = disc[..., None]
    sf = self.net(traj).mean()
    vals = [sf[-1]]
    interm = feature + disc * sf[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    return feature, ret, sf[:-1]
