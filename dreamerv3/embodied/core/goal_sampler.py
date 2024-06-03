import jax
import jax.numpy as jnp


class GoalSampler:
  def __init__(self, feat_space):
    super().__init__()
    self._feat_space = feat_space
    self._low = self._feat_space["vector"].low
    self._high = self._feat_space["vector"].high

  def sample(self, random_key=None):
    if random_key is None:
      return self._feat_space["vector"].sample()
    else:
      random_key, subkey = jax.random.split(random_key)
      values = jax.random.uniform(
        subkey,
        shape=self._feat_space["vector"].shape_no_transform,
        minval=self._low,
        maxval=self._high,)
      return self._feat_space["vector"].transform(values)

  def get_goals_evaluation_metric(self, num_goals_evaluate=50):
    if len(self._low.ravel()) == 1:
      goal_x = jnp.linspace(self._low[0], self._high[0], num_goals_evaluate + 1, endpoint=True)
      goals = jnp.asarray(goal_x).reshape(1, -1)
      goals = jnp.tile(goals, (num_goals_evaluate + 1, 1))
      goals = goals.reshape((num_goals_evaluate + 1, num_goals_evaluate + 1, 1))

      return self._feat_space["vector"].transform(goals)
    elif len(self._low.ravel()) == 2:
      goal_x = jnp.linspace(self._low[0], self._high[0], num_goals_evaluate + 1, endpoint=True)
      goal_y = jnp.linspace(self._low[1], self._high[1], num_goals_evaluate + 1, endpoint=True)
      goal_x_v, goal_y_v = jnp.meshgrid(goal_x, goal_y)

      goals = jnp.stack([goal_x_v, goal_y_v], axis=-1)

      return self._feat_space["vector"].transform(goals)
    elif len(self._low.ravel()) >= 3:
      goal_x = jnp.linspace(self._low[0], self._high[0], num_goals_evaluate + 1, endpoint=True)
      goal_y = jnp.linspace(self._low[1], self._high[1], num_goals_evaluate + 1, endpoint=True)
      goal_x_v, goal_y_v = jnp.meshgrid(goal_x, goal_y)

      goals = jnp.stack([goal_x_v, goal_y_v], axis=-1)

      mid_last_dimensions = (self._low[2:] + self._high[2:]) / 2.
      mid_last_dimensions = mid_last_dimensions.reshape(1, -1)

      goals = jax.vmap(jax.vmap(lambda x: jnp.concatenate([x, mid_last_dimensions.ravel()], axis=-1).ravel()))(goals)

      goals.reshape((num_goals_evaluate + 1, num_goals_evaluate + 1, -1))

      return self._feat_space["vector"].transform(goals)


class GoalSamplerCyclic(GoalSampler):
  def __init__(self, feat_space, goal_list, number_visits_per_goal):
    super().__init__(feat_space)
    self.goal_list = goal_list
    self.array_goals = jnp.asarray(self.goal_list)

    self.index_goal = 0
    self.count_visited_indexes = {
      _index_goal: 0
      for _index_goal in range(len(goal_list))
    }
    self.number_visits_per_goal = number_visits_per_goal

  @staticmethod
  @jax.jit
  def get_min_distance(goal, goal_array):
    distances = jax.vmap(lambda x: jnp.linalg.norm(x - goal))(goal_array)
    min_distance = jnp.min(distances)
    return jnp.argmin(distances), min_distance

  def log_goal(self, goal):
    # Keep track of the goals that have been visited
    index_min, min_distance = self.get_min_distance(goal, self.array_goals)
    if min_distance > 1e-3:
      raise ValueError("The goal is not in the goal list")
    index_min = int(index_min)

    self.count_visited_indexes[index_min] += 1

  def number_goals_to_visit(self):
    return sum(
      max(0, self.number_visits_per_goal - self.count_visited_indexes[index])
      for index in self.count_visited_indexes.keys()
    )

  def all_goals_visited(self):
    return all([
      self.count_visited_indexes[index] >= self.number_visits_per_goal
      for index in self.count_visited_indexes.keys()
    ])

  def sample(self, random_key=None):
    if random_key is None:
      new_goal = self.goal_list[self.index_goal]
      self.index_goal += 1
      if self.index_goal == len(self.goal_list):
        self.index_goal = 0
      return new_goal
    else:
      raise ValueError("This function should not be called with a random_key, and cannot be jitted")
