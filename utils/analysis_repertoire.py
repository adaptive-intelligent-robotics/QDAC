import pickle
from collections import OrderedDict

import jax
import jax.numpy as jnp


import flax.struct

from baselines.qdax.types import Genotype, Fitness, Descriptor, Centroid
from dreamerv3.embodied.run.eval_only import CollectionResults

NUM_BINS = 21

def dist(a, b):
  return jnp.linalg.norm(a - b)

v_dist = jax.vmap(dist, in_axes=(None, 0))
vv_dist = jax.vmap(v_dist, in_axes=(0, None))

class AnalysisRepertoire(flax.struct.PyTreeNode):
  """Class for the repertoire in Map Elites.

  Args:
      genotypes: a PyTree containing all the genotypes in the repertoire ordered
          by the centroids. Each leaf has a shape (num_centroids, num_features). The
          PyTree can be a simple Jax array or a more complex nested structure such
          as to represent parameters of neural network in Flax.
      fitnesses: an array that contains the fitness of solutions in each cell of the
          repertoire, ordered by centroids. The array shape is (num_centroids,).
      descriptors: an array that contains the descriptors of solutions in each cell
          of the repertoire, ordered by centroids. The array shape
          is (num_centroids, num_descriptors).
      centroids: an array the contains the centroids of the tesselation. The array
          shape is (num_centroids, num_descriptors).
  """

  fitnesses: Fitness
  descriptors: Descriptor
  centroids: Centroid

  @classmethod
  def create(cls, fitnesses, descriptors, centroids):
    return cls(fitnesses=fitnesses, descriptors=descriptors, centroids=centroids)

  @classmethod
  def create_from_path_collection_results(cls, path_collection_results: str):
    collection_results = CollectionResults.load(path_collection_results)
    return cls.create_from_collection_results(collection_results)

  @classmethod
  def create_from_path_repertoire(cls, path_repertoire: str):
    with open(path_repertoire, "rb") as f:
      repertoire: AnalysisLatentRepertoire = pickle.load(f)
    return cls.create(
      fitnesses=repertoire.fitnesses,
      descriptors=repertoire.descriptors,
      centroids=repertoire.centroids,
    )

  @classmethod
  def create_from_collection_results(cls, collection_results: CollectionResults):
    goals_array = collection_results.goal
    feat_mean_array = collection_results.feat_mean
    reward_episode_array = collection_results.reward_episode

    dict_goal_per_index_goal = OrderedDict()

    # goal_size = goals_array.shape[1]
    print("goals_array", goals_array.shape)
    array_visited_goals = jnp.full_like(goals_array, fill_value=jnp.nan)
    threshold = 1e-4

    v_dist_jit = jax.jit(v_dist)

    # Grouping goals together
    for index, goal in enumerate(goals_array):
      # print("index", index)
      distance_to_visited_goals = v_dist_jit(goal, array_visited_goals)
      if array_visited_goals is None:
        array_visited_goals = array_visited_goals.at[index].set(goal)
        # list_visited_goals.append(goal)
        dict_goal_per_index_goal[index] = [index]
      elif not jnp.nanmin(distance_to_visited_goals) < threshold:
        array_visited_goals = array_visited_goals.at[index].set(goal)
        dict_goal_per_index_goal[index] = [index]
      else:
        index_goal = int(jnp.nanargmin(distance_to_visited_goals).item())
        assert distance_to_visited_goals[index_goal] < threshold
        dict_goal_per_index_goal[index_goal].append(index)

    list_grouped_goals = [
      goals_array[index_goal]
      for index_goal in dict_goal_per_index_goal.keys()
    ]

    dict_feat_per_index_goal = OrderedDict()
    dict_reward_per_index_goal = OrderedDict()

    number_replicates_goals = min([
      len(indexes)
      for indexes in dict_goal_per_index_goal.values()
    ])

    for index_goal in dict_goal_per_index_goal.keys():
      indexes = jnp.asarray(dict_goal_per_index_goal[index_goal])
      dict_feat_per_index_goal[index_goal] = jnp.asarray(feat_mean_array[indexes])[:number_replicates_goals]
      dict_reward_per_index_goal[index_goal] = jnp.asarray(reward_episode_array[indexes])[:number_replicates_goals]

    feat_mean_array = jnp.asarray(list(dict_feat_per_index_goal.values()))
    reward_episode_array = jnp.asarray(list(dict_reward_per_index_goal.values()))

    return cls(
      fitnesses=reward_episode_array,
      descriptors=feat_mean_array,
      centroids=jnp.asarray(list_grouped_goals),
    )

  def mean_fitnesses(self):
    return jax.vmap(jnp.mean)(self.fitnesses)

  def mean_descriptors(self):
    return jax.vmap(lambda x: jnp.mean(x, axis=0))(self.descriptors)

  def mean_distance_to_goal(self, angles=False):
    if angles:
      return jax.vmap(
        lambda array_descs, goal: jnp.mean(jnp.pi - jnp.abs(v_dist(goal, array_descs) - jnp.pi))
      )(self.descriptors, self.centroids)
    else:
      return jax.vmap(
        lambda array_descs, goal: jnp.mean(v_dist(goal, array_descs))
      )(self.descriptors, self.centroids)

  def filter_out_high_distances_to_goal(self, threshold_distance: float):
    fitnesses = self.mean_fitnesses()
    distance_to_goal = self.mean_distance_to_goal()
    new_fitnesses = fitnesses.at[distance_to_goal > threshold_distance].set(-jnp.inf)
    return AnalysisRepertoire(fitnesses=new_fitnesses, descriptors=self.mean_descriptors(), centroids=self.centroids)

  def filter_out_low_fitnesses(self, threshold_fitness):
    fitnesses = self.mean_fitnesses()
    new_fitnesses = fitnesses.at[fitnesses < threshold_fitness].set(-jnp.inf)
    return AnalysisRepertoire(fitnesses=new_fitnesses, descriptors=self.mean_descriptors(), centroids=self.centroids)

  def qd_score(self, min_fitness, max_fitness, threshold_distance):
    repertoire = self.filter_out_high_distances_to_goal(threshold_distance)
    fitnesses = repertoire.mean_fitnesses()
    fitnesses = (fitnesses - min_fitness) / (max_fitness - min_fitness)
    fitnesses = fitnesses.at[fitnesses < 0.0].set(0.0)
    fitnesses = fitnesses.at[fitnesses > 1.0].set(1.0)
    return jnp.sum(fitnesses)

  def neg_distance_score(self, min_neg_distance):
    distances_to_goal = self.mean_distance_to_goal()
    negated_distances = -distances_to_goal
    negated_distances = (negated_distances - min_neg_distance) / (0. - min_neg_distance)
    negated_distances = negated_distances.at[negated_distances < 0.0].set(0.0)
    negated_distances = negated_distances.at[negated_distances > 1.0].set(1.0)
    return jnp.sum(negated_distances)

  def returns_profile(self, min_fitness, max_fitness, threshold_distance, num_bins=NUM_BINS):
    repertoire = self.filter_out_high_distances_to_goal(threshold_distance)

    fitnesses = repertoire.mean_fitnesses()

    def _percentage_satisfy_constraint(_threshold_fitness):
      return jnp.sum(fitnesses > _threshold_fitness) / len(fitnesses)

    percentage_satisfy_constraint = jax.vmap(_percentage_satisfy_constraint)(jnp.linspace(min_fitness, max_fitness, num_bins))
    return percentage_satisfy_constraint

  def neg_distance_profile(self, min_neg_distance, num_bins=NUM_BINS, angles=False):
    max_neg_distance = 0.0

    neg_mean_distances = -1. * self.mean_distance_to_goal(angles)

    def _percentage_satisfy_constraint(_threshold_distance):
      return jnp.sum(neg_mean_distances > _threshold_distance) / len(neg_mean_distances)

    percentage_satisfy_constraint = jax.vmap(_percentage_satisfy_constraint)(
      jnp.linspace(min_neg_distance, max_neg_distance, num_bins))
    return percentage_satisfy_constraint


class AnalysisLatentRepertoire(AnalysisRepertoire):
  latent_goals: Descriptor
