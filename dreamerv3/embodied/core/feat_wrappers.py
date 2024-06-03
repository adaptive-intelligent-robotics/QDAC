from abc import abstractmethod
from typing import List, Tuple
import warnings

import numpy as np
import jax
import jax.numpy as jnp
from brax.envs.base import Env, State

import embodied
from . import base


class FeatWrapper(base.Wrapper):

  def __init__(self, env, env_name):
    super().__init__(env)
    self._env_name = env_name

  @property
  @abstractmethod
  def feat_space(self) -> dict:
    ...

  @property
  def feat_size(self) -> int:
    return self.feat_space["vector"].shape[0]

  # QDax compatibility
  @property
  def state_descriptor_length(self) -> int:
    return self.feat_space["vector"].shape[0]

  @property
  def state_descriptor_name(self) -> str:
    ...

  @property
  def state_descriptor_limits(self) -> Tuple[List, List]:
    return (jnp.asarray(self.feat_space["vector"].low), jnp.asarray(self.feat_space["vector"].high))

  @property
  def behavior_descriptor_length(self) -> int:
    return self.state_descriptor_length

  @property
  def behavior_descriptor_limits(self) -> Tuple[List, List]:
    return self.state_descriptor_limits

  @property
  def name(self) -> str:
    return self._env_name


# Feet Contact Wrapper
FEET_NAMES = {
    "hopper": ["foot"],
    "walker2d": ["foot", "foot_left"],
    "halfcheetah": ["bfoot", "ffoot"],
    "ant": ["", "", "", ""],
    "humanoid": ["left_shin", "right_shin"],
    "antwall": ["", "", "", ""],
    "humanoidwall": ["left_shin", "right_shin"],
    "humanoidgravity": ["left_shin", "right_shin"],
    "walker2dgravity": ["foot", "foot_left"],
    "antgravity": ["", "", "", ""],
    "walker2dfriction": ["foot", "foot_left"],
    "antfriction": ["", "", "", ""],
    "humanoidfriction": ["left_shin", "right_shin"],
}

class FeetContactWrapper(FeatWrapper):

  def __init__(self, env: Env, env_name: str):
    super().__init__(env, env_name)

    if env_name not in FEET_NAMES:
      raise NotImplementedError(f"FeetContactWrapper does not support {env_name}.")

    if hasattr(self.env, "sys"):
      self._feet_idx = jnp.array(
        [i for i, feet_name in enumerate(self.env.sys.link_names) if feet_name in FEET_NAMES[env_name]]
      )
    else:
      raise NotImplementedError(f"FeetContactWrapper does not support {env_name}.")

  @property
  def feat_space(self) -> dict:
    return {'vector': embodied.Space(np.float32, (len(self._feet_idx),), low=0., high=1.)}

  @property
  def state_descriptor_name(self) -> str:
    return "feet_contact"

  def reset(self, rng: jnp.ndarray) -> State:
    state = self.env.reset(rng)
    state.info["feat"] = self._get_feat(state)
    return state

  def step(self, state: State, action: jnp.ndarray) -> State:
    state = self.env.step(state, action)
    state.info["feat"] = self._get_feat(state)
    return state

  def _get_feat(self, state: State) -> jnp.ndarray:
    return jnp.any(jax.vmap(lambda x: (state.pipeline_state.contact.link_idx[0] == x) & \
                                      (state.pipeline_state.contact.penetration >= 0))(self._feet_idx), axis=-1).astype(jnp.float32)


COG_NAMES = {
    "hopper": "torso",
    "walker2d": "torso",
    "halfcheetah": "torso",
    "ant": "torso",
    "humanoid": "torso",
    "antwall": "torso",
    "humanoidwall": "torso",
    "humanoidgravity": "torso",
    "antgravity": "torso",
    "walker2dgravity": "torso",
}

VELOCITY_BOUNDS = {
    "hopper": (np.array([-5.]), np.array([5.])),
    "walker2d": (np.array([-5.]), np.array([5.])),
    "halfcheetah": (np.array([-5.]), np.array([5.])),
    "ant": (np.array([-5., -5.]), np.array([5., 5.])),
    "humanoid": (np.array([-5., -5.]), np.array([5., 5.])),
    "antwall": (np.array([-5., -5.]), np.array([5., 5.])),
    "humanoidwall": (np.array([-5., -5.]), np.array([5., 5.])),
    "humanoidgravity": (np.array([-5., -5.]), np.array([5., 5.])),
    "antgravity": (np.array([-5., -5.]), np.array([5., 5.])),
    "walker2dgravity": (np.array([-5.]), np.array([5.])),
}

class VelocityWrapper(FeatWrapper):

  def __init__(self, env: Env, env_name: str):
    super().__init__(env, env_name)

    if env_name not in COG_NAMES or env_name not in VELOCITY_BOUNDS:
      raise NotImplementedError(f"VelocityWrapper does not support {env_name}.")

    if hasattr(self.env, "sys"):
      self._cog_idx = self.env.sys.link_names.index(COG_NAMES[env_name])
      self._bounds = VELOCITY_BOUNDS[env_name]
      self._dim = self._bounds[0].size
    else:
      raise NotImplementedError(f"VelocityWrapper does not support {env_name}.")

  @property
  def feat_space(self) -> dict:
    return {'vector': embodied.Space(np.float32, (self._dim,), low=self._bounds[0], high=self._bounds[1])}

  @property
  def state_descriptor_name(self) -> str:
    return "velocity"

  def reset(self, rng: jnp.ndarray) -> State:
    state = self.env.reset(rng)
    state.info["feat"] = self._get_feat(state)
    return state

  def step(self, state: State, action: jnp.ndarray) -> State:
    state = self.env.step(state, action)
    state.info["feat"] = self._get_feat(state)
    return state

  def _get_feat(self, state: State) -> jnp.ndarray:
    return state.pipeline_state.xd.vel[..., self._cog_idx, :self._dim]


class JumpWrapper(FeatWrapper):

  def __init__(self, env: Env, env_name: str):
    super().__init__(env, env_name)

    if env_name not in ["ant", "humanoid"]:
      warnings.warn(f"JumpWrapper does not support {env_name}.")

  @property
  def feat_space(self) -> dict:
    return {'vector': embodied.Space(np.float32, (1,), low=0., high=0.25)}

  @property
  def state_descriptor_name(self) -> str:
    return "jump"

  def reset(self, rng: jnp.ndarray) -> State:
    state = self.env.reset(rng)
    state.info["feat"] = self._get_feat(state)
    return state

  def step(self, state: State, action: jnp.ndarray) -> State:
    state = self.env.step(state, action)
    state.info["feat"] = self._get_feat(state)
    return state

  def _get_feat(self, state: State) -> jnp.ndarray:
    return jnp.min(state.pipeline_state.contact.pos[..., 2], axis=-1, keepdims=True)


class AngleWrapper(FeatWrapper):

  def __init__(self, env: Env, env_name: str, trigonometric: bool = True):
    super().__init__(env, env_name)

    if env_name not in COG_NAMES:
      raise NotImplementedError(f"AngleWrapper does not support {env_name}.")

    if hasattr(self.env, "sys"):
      self._cog_idx = self.env.sys.link_names.index("pelvis" if env_name == "humanoid" else COG_NAMES[env_name])
    else:
      raise NotImplementedError(f"AngleWrapper does not support {env_name}.")

    self._trigonometric = trigonometric

  @property
  def feat_space(self) -> dict:
    if self._trigonometric:
      return {'vector': embodied.AngularSpace(np.float32, (1,))}
    else:
      return {'vector': embodied.Space(np.float32, (1,), low=-np.pi, high=np.pi)}

  def reset(self, rng: jnp.ndarray) -> State:
    state = self.env.reset(rng)
    state.info["feat"] = self._get_feat(state)
    return state

  def step(self, state: State, action: jnp.ndarray) -> State:
    state = self.env.step(state, action)
    state.info["feat"] = self._get_feat(state)
    return state

  def _inverse_quaternion(self, q):
    return jnp.array([q[0], -q[1], -q[2], -q[3]])/jnp.linalg.norm(q)

  def _multiply_quaternions(self, q_1, q_2):
    w1, x1, y1, z1 = q_1
    w2, x2, y2, z2 = q_2
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

  def _rotate_vector_with_quaternion(self, v, q):
    v = self._multiply_quaternions(q, jnp.array([0., v[0], v[1], v[2]]))
    vector = self._multiply_quaternions(v, self._inverse_quaternion(q))
    return vector[1:]

  def _angle_between(self, v_1, v_2):
    return jnp.arctan2(jnp.cross(v_1, v_2), jnp.dot(v_1, v_2))

  def _get_feat(self, state: State) -> jnp.ndarray:
    rotate_x = self._rotate_vector_with_quaternion(jnp.array([1., 0., 0.]), state.pipeline_state.x.rot[self._cog_idx])
    angle = self._angle_between(jnp.array([1., 0.]), rotate_x[:-1])
    angle = angle[None]
    if self._trigonometric:
      return jnp.concatenate([jnp.cos(angle), jnp.sin(angle)], axis=-1)
    else:
      return angle


class FullStateWrapper(FeatWrapper):

  def __init__(self, env: Env, env_name: str):
    super().__init__(env, env_name)

    # if env_name not in COG_NAMES or env_name not in VELOCITY_BOUNDS:
    #   raise NotImplementedError(f"FullStateWrapper does not support {env_name}.")

    if hasattr(self.env, "sys"):
      fake_state = self.env.reset(jax.random.PRNGKey(0))
      self._dim = fake_state.obs.shape[0]
      self._bounds = (np.full(self._dim, -np.inf), np.full(self._dim, np.inf))
    else:
      raise NotImplementedError(f"FullStateWrapper does not support {env_name}.")

  @property
  def feat_space(self) -> dict:
    return {'vector': embodied.Space(np.float32, (self._dim,), low=self._bounds[0], high=self._bounds[1])}

  @property
  def state_descriptor_name(self) -> str:
    return "full_state"

  def reset(self, rng: jnp.ndarray) -> State:
    state = self.env.reset(rng)
    state.info["feat"] = self._get_feat(state)
    return state

  def step(self, state: State, action: jnp.ndarray) -> State:
    state = self.env.step(state, action)
    state.info["feat"] = self._get_feat(state)
    return state

  def _get_feat(self, state: State) -> jnp.ndarray:
    return state.obs


class JointStateWrapper(FeatWrapper):

  def __init__(self, env: Env, env_name: str):
    super().__init__(env, env_name)

    if env_name.startswith("ant"):
      self.joint_angles_indexes = list(range(5, 13))  # Angles are from index 5 to 12 inclusive
      self.joint_velocities_indexes = list(range(19, 27))  # Velocities are from index 19 to 26 inclusive
    elif env_name.startswith("humanoid"):
      self.joint_angles_indexes = list(range(5, 22))
      self.joint_velocities_indexes = list(range(28, 45))
    elif env_name.startswith("walker2d"):
      self.joint_angles_indexes = list(range(2, 8))
      self.joint_velocities_indexes = list(range(10, 17))
    else:
      raise NotImplementedError(f"JointStateWrapper does not support {env_name}.")

    # if env_name not in COG_NAMES or env_name not in VELOCITY_BOUNDS:
    #   raise NotImplementedError(f"FullStateyWrapper does not support {env_name}.")

    if hasattr(self.env, "sys"):
      self._dim = len(self.joint_angles_indexes) + len(self.joint_velocities_indexes)
      self._bounds = (np.full(self._dim, -np.inf), np.full(self._dim, np.inf))
    else:
      raise NotImplementedError(f"FullStateyWrapper does not support {env_name}.")

  @property
  def feat_space(self) -> dict:
    return {'vector': embodied.Space(np.float32, (self._dim,), low=self._bounds[0], high=self._bounds[1])}

  @property
  def state_descriptor_name(self) -> str:
    return "joint_state"

  def reset(self, rng: jnp.ndarray) -> State:
    state = self.env.reset(rng)
    state.info["feat"] = self._get_feat(state)
    return state

  def step(self, state: State, action: jnp.ndarray) -> State:
    state = self.env.step(state, action)
    state.info["feat"] = self._get_feat(state)
    return state

  def _get_feat(self, state: State) -> jnp.ndarray:
    return state.obs[..., self.joint_angles_indexes + self.joint_velocities_indexes]
