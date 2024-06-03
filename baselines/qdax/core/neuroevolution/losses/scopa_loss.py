from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from brax.training.distribution import ParametricDistribution

from baselines.qdax.core.neuroevolution.buffers.buffer import Transition
from baselines.qdax.types import Action, Observation, Params, RNGKey, Skill


def make_scopa_loss_fn(
    actor_fn: Callable[[Params, Observation, Skill], jnp.ndarray],
    critic_fn: Callable[[Params, Observation, Action, Skill], jnp.ndarray],
    successor_features_fn: Callable[[Params, Observation, Action, Skill], jnp.ndarray],
    lagrange_fn: Callable[[Params, Observation, Skill], jnp.ndarray],
    parametric_action_distribution: ParametricDistribution,
    reward_scaling: float,
    discount: float,
    action_size: int,
    delta: float,
    delta_pseudo_huber_loss: float,
    lambda_: float,
) -> Tuple[
    Callable[[RNGKey, Params, Params, Params, Params, jnp.ndarray, Transition], jnp.ndarray],
    Callable[[RNGKey, Params, Params, Params, jnp.ndarray, Transition], jnp.ndarray],
    Callable[[RNGKey, Params, Params, Params, Transition], jnp.ndarray],
    Callable[[RNGKey, Params, Params, jnp.ndarray, Transition], jnp.ndarray],
    Callable[[RNGKey, Params, jnp.ndarray, Transition], jnp.ndarray],
]:
    """Creates the loss used in SCOPA.

    Args:
        actor_fn: the apply function of the actor
        critic_fn: the apply function of the critic
        successor_features_fn: the apply function of the successor features
        lagrange_fn: the apply function of the Lagrange
        parametric_action_distribution: the distribution over actions
        reward_scaling: a multiplicative factor to the reward
        discount: the discount factor
        action_size: the size of the environment's action space
        delta: the distance to skill threshold

    Returns:
        the loss of the actor
        the loss of the critic
        the loss of the successor features
        the loss of the lagrange
        the loss of alpha
    """
    def _euclidean_distance(x1, x2):
        return jnp.sqrt(jnp.sum(jnp.square(x1 - x2), axis=-1))


    def _pseudo_huber_loss(x1, x2):
        return delta_pseudo_huber_loss * (jnp.sqrt(1 + jnp.sum(jnp.square(x1 - x2), axis=-1)/(delta_pseudo_huber_loss**2)) - 1)


    def _actor_loss_fn(
        actor_params: Params,
        critic_params: Params,
        successor_features_params: Params,
        lagrange_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        random_key: RNGKey,
    ) -> jnp.ndarray:
        """
        Creates the actor loss used in SAC.

        Args:
            actor_params: parameters of the actor
            critic_params: parameters of the critic
            successor_features_params: parameters of the successor features
            lagrange_params: parameters of the lagrange
            alpha: entropy coefficient value
            transitions: transitions collected by the agent
            random_key: random key

        Returns:
            Loss of the actor
        """
        dist_params = actor_fn(actor_params, transitions.obs, transitions.desc_prime)
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, random_key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)

        q_action = critic_fn(critic_params, transitions.obs, action, transitions.desc_prime)
        value = jnp.mean(q_action, axis=-1)

        sf_action = successor_features_fn(successor_features_params, transitions.obs, action, transitions.desc_prime)
        scaled_sf = (1 - discount) * jnp.mean(sf_action, axis=-1)
        distance_to_skill = _pseudo_huber_loss(scaled_sf, transitions.desc_prime)

        lagrange = jax.nn.sigmoid(jnp.squeeze(lagrange_fn(lagrange_params, transitions.obs, transitions.desc_prime)))

        actor_loss = alpha * log_prob - (1 - lagrange) * value + lambda_ * lagrange * distance_to_skill
        return jnp.mean(actor_loss)


    def _critic_loss_fn(
        critic_params: Params,
        target_critic_params: Params,
        actor_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        random_key: RNGKey,
    ) -> jnp.ndarray:
        """
        Creates the critic loss used in SAC.

        Args:
            critic_params: parameters of the critic
            target_critic_params: parameters of the target critic
            actor_params: parameters of the actor
            alpha: entropy coefficient value
            transitions: transitions collected by the agent
            random_key: random key

        Returns:
            Loss of the critic
        """
        q_old_action = critic_fn(critic_params, transitions.obs, transitions.actions, transitions.desc_prime)

        next_dist_params = actor_fn(actor_params, transitions.next_obs, transitions.desc_prime)
        next_action = parametric_action_distribution.sample_no_postprocessing(
            next_dist_params, random_key
        )
        next_log_prob = parametric_action_distribution.log_prob(
            next_dist_params, next_action
        )
        next_action = parametric_action_distribution.postprocess(next_action)

        next_q = critic_fn(target_critic_params, transitions.next_obs, next_action, transitions.desc_prime)
        next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob

        target_q = jax.lax.stop_gradient(
            transitions.rewards * reward_scaling
            + (1.0 - transitions.dones) * discount * next_v
        )

        q_error = q_old_action - target_q[..., None]
        q_error *= (1 - transitions.truncations)[..., None]
        q_loss = 0.5 * jnp.mean(jnp.square(q_error))

        return q_loss


    def _successor_features_loss_fn(
        successor_features_params: Params,
        target_successor_features_params: Params,
        actor_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        random_key: RNGKey,
    ) -> jnp.ndarray:
        """
        Creates the successor_features loss used in SAC.

        Args:
            target_successor_features_params: parameters of the target successor_features
            successor_features_params: parameters of the successor_features
            actor_params: parameters of the actor
            transitions: transitions collected by the agent
            random_key: random key

        Returns:
            Loss of the successor_features
        """
        sf_old_action = successor_features_fn(successor_features_params, transitions.obs, transitions.actions, transitions.desc_prime)

        next_dist_params = actor_fn(actor_params, transitions.next_obs, transitions.desc_prime)
        next_action = parametric_action_distribution.sample_no_postprocessing(
            next_dist_params, random_key
        )
        next_log_prob = parametric_action_distribution.log_prob(
            next_dist_params, next_action
        )
        next_action = parametric_action_distribution.postprocess(next_action)

        next_sf = successor_features_fn(target_successor_features_params, transitions.next_obs, next_action, transitions.desc_prime)
        next_sf = jnp.mean(next_sf, axis=-1) # - alpha * next_log_prob[..., None]  # TODO: check if this is correct

        target_sf = jax.lax.stop_gradient(
            transitions.state_desc
            + (1.0 - transitions.dones[..., None]) * discount * next_sf
        )

        sf_error = sf_old_action - target_sf[..., None]
        sf_error *= (1 - transitions.truncations[..., None, None])
        sf_loss = 0.5 * jnp.mean(jnp.square(sf_error))

        return sf_loss


    def _lagrange_loss_fn(
        lagrange_params: Params,
        actor_params: Params,
        successor_features_params: Params,
        transitions: Transition,
        random_key: RNGKey,
    ) -> jnp.ndarray:
        """
        Creates the actor loss used in SAC.

        Args:
            lagrange_params: parameters of the Lagrange
            actor_params: parameters of the actor
            successor_features_params: parameters of the successor features
            transitions: transitions collected by the agent
            random_key: random key

        Returns:
            Loss of the lagrange
        """
        dist_params = actor_fn(actor_params, transitions.obs, transitions.desc_prime)
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, random_key
        )
        action = parametric_action_distribution.postprocess(action)

        sf_action = successor_features_fn(successor_features_params, transitions.obs, action, transitions.desc_prime)
        scaled_sf = (1 - discount) * jnp.mean(sf_action, axis=-1)
        distance_to_skill = _euclidean_distance(scaled_sf, transitions.desc_prime)

        labels = jax.lax.stop_gradient(distance_to_skill >= delta).astype(jnp.float32)
        logits = jnp.squeeze(lagrange_fn(lagrange_params, transitions.obs, transitions.desc_prime))

        # Compute binary cross-entropy loss
        log_prob = jax.nn.log_sigmoid(logits)
        log_not_prob = jax.nn.log_sigmoid(-logits)
        return -jnp.mean(labels * log_prob + (1 - labels) * log_not_prob)


    def _alpha_loss_fn(
        alpha_params: jnp.ndarray,
        actor_params: Params,
        transitions: Transition,
        random_key: RNGKey,
    ) -> jnp.ndarray:
        """
        Creates the alpha loss used in SAC.
        Eq 18 from https://arxiv.org/pdf/1812.05905.pdf.

        Args:
            alpha_params: entropy coefficient log value
            actor_params: parameters of the actor
            transitions: transitions collected by the agent
            random_key: random key

        Returns:
            the loss of the entropy parameter auto-tuning
        """

        target_entropy = -0.5 * action_size

        dist_params = actor_fn(actor_params, transitions.obs, transitions.desc_prime)
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, random_key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        alpha = jnp.exp(alpha_params)
        alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)

        loss = jnp.mean(alpha_loss)
        return loss

    return _actor_loss_fn, _critic_loss_fn, _successor_features_loss_fn, _lagrange_loss_fn, _alpha_loss_fn
