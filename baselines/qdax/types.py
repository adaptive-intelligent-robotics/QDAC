"""Defines types used in QDax"""

from typing import Dict, Generic, TypeVar, Union

from brax.envs.base import State
import jax
import jax.numpy as jnp
from chex import ArrayTree
from typing_extensions import TypeAlias


# MDP types
Observation: TypeAlias = jnp.ndarray
Action: TypeAlias = jnp.ndarray
Reward: TypeAlias = jnp.ndarray
Done: TypeAlias = jnp.ndarray
EnvState: TypeAlias = State
Params: TypeAlias = ArrayTree

# Evolution types
Genotype: TypeAlias = ArrayTree
Centroid: TypeAlias = jnp.ndarray
Fitness: TypeAlias = jnp.ndarray
Descriptor: TypeAlias = jnp.ndarray
StateDescriptor: TypeAlias = jnp.ndarray
Gradient: TypeAlias = jnp.ndarray

Skill: TypeAlias = jnp.ndarray

ExtraScores: TypeAlias = Dict[str, ArrayTree]

# Pareto fronts
T = TypeVar("T", bound=Union[Fitness, Genotype, Descriptor, jnp.ndarray])


class ParetoFront(Generic[T]):
    def __init__(self) -> None:
        super().__init__()


# Others
Mask: TypeAlias = jnp.ndarray
RNGKey: TypeAlias = jax.Array
Metrics: TypeAlias = Dict[str, jnp.ndarray]
