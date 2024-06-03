from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from baselines.qdax.types import Action, Observation


def make_diayn_networks(
    skill_type: str,
    num_skills: int,
    action_size: int,
    hidden_layer_sizes: Tuple[int, ...] = (256, 256),
) -> Tuple[hk.Transformed, hk.Transformed, hk.Transformed]:
    """Creates networks used in DIAYN.

    Args:
        action_size: the size of the environment's action space
        num_skills: the number of skills set
        hidden_layer_sizes: the number of neurons for hidden layers.
            Defaults to (256, 256).

    Returns:
        the policy network
        the critic network
        the discriminator network
    """

    def _actor_fn(obs: Observation) -> jnp.ndarray:
        network = hk.Sequential(
            [
                hk.nets.MLP(
                    list(hidden_layer_sizes) + [2 * action_size],
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                ),
            ]
        )
        return network(obs)

    def _critic_fn(obs: Observation, action: Action) -> jnp.ndarray:
        network1 = hk.Sequential(
            [
                hk.nets.MLP(
                    list(hidden_layer_sizes) + [1],
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                ),
            ]
        )
        network2 = hk.Sequential(
            [
                hk.nets.MLP(
                    list(hidden_layer_sizes) + [1],
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                ),
            ]
        )
        input_ = jnp.concatenate([obs, action], axis=-1)
        value1 = network1(input_)
        value2 = network2(input_)
        return jnp.concatenate([value1, value2], axis=-1)

    def _discriminator_categorical_fn(obs: Observation) -> jnp.ndarray:
        network = hk.Sequential(
            [
                hk.nets.MLP(
                    list(hidden_layer_sizes) + [num_skills],
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                ),
            ]
        )
        return network(obs)

    def _discriminator_normal_fn(obs: Observation) -> jnp.ndarray:
        network = hk.Sequential(
            [
                hk.nets.MLP(
                    list(hidden_layer_sizes),
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu,
                ),
            ]
        )
        mean = hk.Linear(num_skills)(network(obs))
        log_std = hk.Linear(num_skills)(network(obs))
        return mean, log_std

    policy = hk.without_apply_rng(hk.transform(_actor_fn))
    critic = hk.without_apply_rng(hk.transform(_critic_fn))
    if skill_type == "categorical":
        discriminator = hk.without_apply_rng(hk.transform(_discriminator_categorical_fn))
    elif skill_type == "normal":
        discriminator = hk.without_apply_rng(hk.transform(_discriminator_normal_fn))
    else:
        raise NotImplementedError(f"Not implemented skill type {skill_type}.")

    return policy, critic, discriminator
