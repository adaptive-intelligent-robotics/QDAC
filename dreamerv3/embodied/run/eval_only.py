import pickle
from datetime import datetime
import re

import flax.struct
import jax

import embodied
import numpy as np

import jax.numpy as jnp

from brax.io import html

from dreamerv3.embodied.core.goal_sampler import GoalSamplerCyclic


def save_rollout_html(env, pipeline_env_list, file_name: str):
  rollout_html = html.render(env.sys.replace(dt=env.dt), pipeline_env_list)
  with open(file_name, 'w') as f:
      f.write(rollout_html)

@flax.struct.dataclass
class CollectionResults:
  goal: jnp.ndarray
  feat_mean: jnp.ndarray
  reward_episode: jnp.ndarray

  @classmethod
  def create(cls, goal, feat_mean, reward_episode):
    return cls(goal, feat_mean, reward_episode)

  @classmethod
  def load(cls, path):
    with open(path, "rb") as f:
      results_tree = pickle.load(f)
    return cls.create(
      goal=results_tree.goal,
      feat_mean=results_tree.feat_mean,
      reward_episode=results_tree.reward_episode,
    )


def eval_only(agent, env, goal_sampler: GoalSamplerCyclic, period_sample_goals, logger, args):
  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  print('Observation space:', env.obs_space)
  print('Action space:', env.act_space)

  results = []
  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    logger.add({'length': length, 'score': score}, prefix='episode')
    feat_mean = np.mean(ep['feat'], axis=0)
    goal = ep['goal'][0]
    # print(f'Episode has {length} steps, return {score:.1f}, and mean feat {feat_mean} for goal {goal}.')

    results.append(
      CollectionResults.create(goal=goal, reward_episode=score, feat_mean=feat_mean)
    )
    goal_sampler.log_goal(goal)

  driver = embodied.Driver(env, goal_sampler, period_sample_goals)
  driver.on_episode(lambda ep, worker: per_episode(ep))

  checkpoint = embodied.Checkpoint()
  checkpoint.agent = agent
  checkpoint.load(args.from_checkpoint, keys=['agent'])

  print('Start evaluation loop.')
  policy = lambda *args: agent.policy(*args, mode='eval')
  while not goal_sampler.all_goals_visited():
    # print('Number of goals to visit:', goal_sampler.number_goals_to_visit())
    driver(policy, episodes=1)
  logger.write()

  path_saving_results = str(logdir / "results_dreamer.pkl")

  print("Saving results tree to", path_saving_results)
  results_tree = jax.tree_map(
    lambda *res: jnp.asarray(res),
    *results
  )

  # use pickle to save the results tree
  with open(path_saving_results, "wb") as f:
    pickle.dump(results_tree, f)
    print("Saved results tree to", path_saving_results)
