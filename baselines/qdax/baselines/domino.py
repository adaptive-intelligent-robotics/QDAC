"""
A collection of functions and classes that define the algorithm Diversity Is All You
Need (DIAYN), ref: https://arxiv.org/abs/1802.06070.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import optax

from brax.envs import State as EnvState
from brax.training.distribution import NormalTanhDistribution, ParametricDistribution

from baselines.qdax.baselines.domino_networks import make_domino_networks
from baselines.qdax.baselines.sac import SAC, SacConfig
from baselines.qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer, Transition
from baselines.qdax.core.neuroevolution.mdp_utils import TrainingState, get_first_episode
from baselines.qdax.core.neuroevolution.sac_td3_utils import generate_unroll
from baselines.qdax.environments import QDEnv
from baselines.qdax.types import Metrics, Params, Reward, RNGKey, Skill, StateDescriptor, Observation, Action




class DOMINOTransition(QDTransition):

    @classmethod
    def init_dummy(  # type: ignore
        cls, observation_dim: int, action_dim: int, descriptor_dim: int, num_skills) -> QDTransition:
        """
        Initialize a dummy transition that then can be passed to constructors to get
        all shapes right.
        Args:
            observation_dim: observation dimension
            action_dim: action dimension
        Returns:
            a dummy transition
        """
        dummy_transition = cls(
            obs=jnp.zeros(shape=(1, observation_dim)),
            next_obs=jnp.zeros(shape=(1, observation_dim)),
            rewards=jnp.zeros(shape=(1,)),
            dones=jnp.zeros(shape=(1,)),
            truncations=jnp.zeros(shape=(1,)),
            actions=jnp.zeros(shape=(1, action_dim)),
            state_desc=jnp.zeros(shape=(1, descriptor_dim)),
            next_state_desc=jnp.zeros(shape=(1, descriptor_dim)),
            desc=jnp.zeros(shape=(1, descriptor_dim)),
            desc_prime=jnp.zeros(shape=(1, descriptor_dim)),
        )
        return dummy_transition

    @property
    def flatten_dim(self) -> int:
        """
        Returns:
            the dimension of the transition once flattened.
        """
        flatten_dim = (
            2 * self.observation_dim
            + self.action_dim
            + 3
            + 2 * self.state_descriptor_dim
            + 2 * self.descriptor_dim
        )
        return flatten_dim

    def flatten(self) -> jnp.ndarray:
        """
        Returns:
            a jnp.ndarray that corresponds to the flattened transition.
        """
        flatten_transition = jnp.concatenate(
            [
                self.obs,
                self.next_obs,
                jnp.expand_dims(self.rewards, axis=-1),
                jnp.expand_dims(self.dones, axis=-1),
                jnp.expand_dims(self.truncations, axis=-1),
                self.actions,
                self.state_desc,
                self.next_state_desc,
                self.desc,
                self.desc_prime,
            ],
            axis=-1,
        )
        return flatten_transition

    @classmethod
    def from_flatten(
        cls,
        flattened_transition: jnp.ndarray,
        transition: DOMINOTransition,
    ) -> DOMINOTransition:
        """
        Creates a transition from a flattened transition in a jnp.ndarray.
        Args:
            flattened_transition: flattened transition in a jnp.ndarray of shape
                (batch_size, flatten_dim)
            transition: a transition object (might be a dummy one) to
                get the dimensions right
        Returns:
            a Transition object
        """
        obs_dim = transition.observation_dim
        action_dim = transition.action_dim
        state_desc_dim = transition.state_descriptor_dim
        desc_dim = transition.descriptor_dim

        obs = flattened_transition[:, :obs_dim]
        next_obs = flattened_transition[:, obs_dim : (2 * obs_dim)]
        rewards = jnp.ravel(flattened_transition[:, (2 * obs_dim) : (2 * obs_dim + 1)])
        dones = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 1) : (2 * obs_dim + 2)]
        )
        truncations = jnp.ravel(
            flattened_transition[:, (2 * obs_dim + 2) : (2 * obs_dim + 3)]
        )
        actions = flattened_transition[
            :, (2 * obs_dim + 3) : (2 * obs_dim + 3 + action_dim)
        ]
        state_desc = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim) : (2 * obs_dim + 3 + action_dim + state_desc_dim),
        ]
        next_state_desc = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim + state_desc_dim) : (
                2 * obs_dim + 3 + action_dim + 2 * state_desc_dim
            ),
        ]
        desc = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim + 2 * state_desc_dim) : (
                2 * obs_dim + 3 + action_dim + 2 * state_desc_dim + desc_dim
            ),
        ]
        desc_prime = flattened_transition[
            :,
            (2 * obs_dim + 3 + action_dim + 2 * state_desc_dim + desc_dim) : (
                2 * obs_dim + 3 + action_dim + 2 * state_desc_dim + 2 * desc_dim
            ),
        ]
        return cls(
            obs=obs,
            next_obs=next_obs,
            rewards=rewards,
            dones=dones,
            truncations=truncations,
            actions=actions,
            state_desc=state_desc,
            next_state_desc=next_state_desc,
            desc=desc,
            desc_prime=desc_prime,
        )


class DOMINOTrainingState(TrainingState):
    """Training state for the DIAYN algorithm"""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    critic_optimizer_state: optax.OptState
    critic_params: Params
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    target_critic_params: Params
    random_key: RNGKey
    steps: jnp.ndarray
    avg_values: jnp.ndarray  # shape: (num_skills, dim_feats)
    avg_sfs: jnp.ndarray  # shape: (num_skills, dim_feats)
    lagrange_optimizer_state: optax.OptState
    lagrange_params: Params


    @classmethod
    def create(cls,
               policy_optimizer_state: optax.OptState,
               policy_params: Params,
               critic_optimizer_state: optax.OptState,
               critic_params: Params,
               alpha_optimizer_state: optax.OptState,
               alpha_params: Params,
               target_critic_params: Params,
               random_key: RNGKey,
               steps: jnp.ndarray,
               avg_values: jnp.ndarray,
               avg_sfs: jnp.ndarray,
                lagrange_optimizer_state: optax.OptState,
                lagrange_params: Params,
               ):
        return cls(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            target_critic_params=target_critic_params,
            random_key=random_key,
            steps=steps,
            avg_values=avg_values,
            avg_sfs=avg_sfs,
            lagrange_optimizer_state=lagrange_optimizer_state,
            lagrange_params=lagrange_params,
        )

@dataclass
class DOMINOConfig(SacConfig):
    """Configuration for the DOMINO algorithm"""

    skill_type: str = "categorical"
    num_skills: int = 5
    descriptor_full_state: bool = False

    # Those values are taken from the DOMINO paper for DMControl environments
    optimality_ratio: float = 0.5
    alpha_d_v_avg: float = 0.9
    alpha_d_sfs_avg: float = 0.99

    learning_rate_lagrange: float = 1e-3


class DOMINO(SAC):
    """Implements DIAYN algorithm https://arxiv.org/abs/1802.06070.

    Note that the functions select_action, _update_alpha, _update_critic and
    _update_actor are inherited from SAC algorithm.

    In the current implementation, we suppose that the skills are fixed one
    hot vectors, and do not support continuous skills at the moment.

    Also, we suppose that the skills are evaluated in parallel in a fixed
    manner: a batch of environments, containing a multiple of the number
    of skills, is used to evaluate the skills in the environment and hence
    to generate transitions. The sampling is hence fixed and perfectly uniform.

    Since we are using categorical skills, the current loss function used
    to train the discriminator is the categorical cross entropy loss.

    We plan to add continous skill as an option in the future. We also plan
    to release the current constraint on the number of batched environments
    by sampling from the skills rather than having this fixed setting.
    """

    @staticmethod
    def domino_policy_loss_fn(
        policy_params: Params,
        policy_fn: Callable[[Params, Observation], jnp.ndarray],
        critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
        parametric_action_distribution: ParametricDistribution,
        critic_params: Params,
        alpha: jnp.ndarray,
        lagrange_params: jnp.ndarray,
        transitions: DOMINOTransition,
        random_key: RNGKey,
        skill: Skill,
    ) -> jnp.ndarray:
        """
        Creates the policy loss used in SAC.

        Args:
            policy_params: parameters of the policy
            policy_fn: the apply function of the policy
            critic_fn: the apply function of the critic
            parametric_action_distribution: the distribution over actions
            critic_params: parameters of the critic
            alpha: entropy coefficient value
            lagrange_params: parameters of the lagrange multiplier
            transitions: transitions collected by the agent
            random_key: random key

        Returns:
            the loss of the policy
        """

        dist_params = policy_fn(policy_params, transitions.obs)
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, random_key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)
        q_action_e, q_action_d = jax.vmap(critic_fn, in_axes=(0, None, None))(critic_params, transitions.obs, action)

        min_q_e = jnp.min(q_action_e, axis=-1)  # shape: (batch_size,)
        min_q_d = jnp.min(q_action_d, axis=-1)  # shape: (batch_size,)
        # lagrange_params # shape: (num_skills,)
        # make skill of shape (batch_size, num_skills)
        num_skills = skill.shape[0]
        
        skill = jnp.expand_dims(skill, axis=0)
        skill = jnp.repeat(skill, repeats=transitions.obs.shape[0], axis=0)

        lagrange_params_repeat = jax.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), 
                                              repeats=num_skills, axis=0), lagrange_params)
        min_q = DOMINO.weight_cumulants(lagrange_params_repeat, skill, min_q_e, min_q_d)
        # min_q = min_q_e

        actor_loss = alpha * log_prob - min_q

        loss = jnp.mean(actor_loss)

        return loss

    @staticmethod
    def domino_critic_loss_fn(
        critic_params: Params,
        policy_fn: Callable[[Params, Observation], jnp.ndarray],
        critic_fn: Callable[[Params, Observation, Action], jnp.ndarray],
        parametric_action_distribution: ParametricDistribution,
        reward_scaling: float,
        discount: float,
        policy_params: Params,
        target_critic_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        diversity_rewards: jnp.ndarray,
        random_key: RNGKey,
    ) -> jnp.ndarray:
        """
        Creates the critic loss used in SAC.

        Args:
            critic_params: parameters of the critic
            policy_fn: the apply function of the policy
            critic_fn: the apply function of the critic
            parametric_action_distribution: the distribution over actions
            policy_params: parameters of the policy
            target_critic_params: parameters of the target critic
            alpha: entropy coefficient value
            transitions: transitions collected by the agent
            diversity_rewards: diversity rewards
            random_key: random key
            reward_scaling: a multiplicative factor to the reward
            discount: the discount factor

        Returns:
            the loss of the critic
        """

        q_old_action_e, q_old_action_d = jax.vmap(critic_fn, in_axes=(0, None, None))(critic_params, transitions.obs, transitions.actions)
        next_dist_params = policy_fn(policy_params, transitions.next_obs)
        next_action = parametric_action_distribution.sample_no_postprocessing(
            next_dist_params, random_key
        )
        next_log_prob = parametric_action_distribution.log_prob(
            next_dist_params, next_action
        )
        next_action = parametric_action_distribution.postprocess(next_action)
        next_q_e, next_q_d = jax.vmap(critic_fn, in_axes=(0, None, None))(target_critic_params, transitions.next_obs, next_action)

        next_v_e = jnp.min(next_q_e, axis=-1) - alpha * next_log_prob
        next_v_d = jnp.min(next_q_d, axis=-1) - alpha * next_log_prob

        target_q_e = jax.lax.stop_gradient(
            transitions.rewards * reward_scaling
            + (1.0 - transitions.dones) * discount * next_v_e
        )
        target_q_d = jax.lax.stop_gradient(
            diversity_rewards * reward_scaling
            + (1.0 - transitions.dones) * discount * next_v_d
        )

        q_error_e = q_old_action_e - jnp.expand_dims(target_q_e, -1)
        q_error_e *= jnp.expand_dims(1 - transitions.truncations, -1)

        q_error_d = q_old_action_d - jnp.expand_dims(target_q_d, -1)
        q_error_d *= jnp.expand_dims(1 - transitions.truncations, -1)

        q_loss = 0.5 * jnp.mean(jnp.square(q_error_e)) + 0.5 * jnp.mean(jnp.square(q_error_d))

        return q_loss

    @staticmethod
    def intrinsic_reward(phi, sfs, latents, attractive_power=3.,
                         repulsive_power=0., attractive_coeff=0., target_d=1.):
        """Computes a diversity reward using successor features.

        Args:
            phi: features [tbf].
            sfs: avg successor features [lf] or
                predicted, discounted successor features [tbfl].
            latents: [tbl].
            attractive_power: the power of the attractive force.
            repulsive_power: the power of the repulsive force.
            attractive_coeff: convex mixing of attractive & repulsive forces
            target_d(\ell_0): desired target distance between the sfs.
            When attractive_coeff=0.5, target_d is the minimizer of the
            objective, i.e., the gradient (the reward) is zero.

        Returns:
            intrinsic_reward.
        """

        # If sfs are predicted we have 2 extra leading dims.
        if jnp.ndim(sfs) == 4:
            sfs = jnp.swapaxes(sfs, 2, 3)  # tbfl -> tblf (to match lf shape of avg sf)
            compute_dist_fn = jax.vmap(jax.vmap(DOMINO.compute_distances))
            matmul_fn = lambda x, y: jnp.einsum('tbl,tblf->tbf', x, y)
        elif jnp.ndim(sfs) == 2:
            compute_dist_fn = DOMINO.compute_distances
            matmul_fn = jnp.matmul
        else:
            raise ValueError('Invalid shape for argument ‘sfs‘.')

        l, f = sfs.shape[-2:]

        # Computes an tb lxl matrix where each row, corresponding to a latent,
        # is a 1 hot vector indicating the index of the latent with the closest sfs
        dists = compute_dist_fn(sfs, sfs)
        dists += jnp.eye(l) * jnp.max(dists)
        nearest_latents_matrix = jax.nn.one_hot(jnp.argmin(dists, axis=-2), num_classes=l)

        # Computes a [tbl] vector with the nearest latent to each latent in latents
        nearest_latents = matmul_fn(latents, nearest_latents_matrix)

        # Compute psi_i-psi_j
        psi_diff = matmul_fn(latents - nearest_latents, sfs)  # tbf
        norm_diff = jnp.sqrt(jnp.sum(jnp.square(psi_diff), axis=-1)) / target_d
        c = (1. - attractive_coeff) * norm_diff ** repulsive_power
        c -= attractive_coeff * norm_diff ** attractive_power
        reward = c * jnp.sum(phi * psi_diff, axis=-1) / f

        return reward

    @staticmethod
    def l2dist(x, y):
        """Returns the L2 distance between a pair of inputs."""
        return jnp.sqrt(jnp.sum(jnp.square(x - y)))

    @staticmethod
    def compute_distances(x, y, dist_fn=None):
        """Returns the distance between each pair of the two collections of inputs."""

        if dist_fn is None:
            dist_fn = DOMINO.l2dist

        return jax.vmap(jax.vmap(dist_fn, (None, 0)), (0, None))(x, y)


    @staticmethod
    def weight_cumulants(lagrange, latents, extrinsic_cumulants, intrinsic_cumulants):
        """Weights cumulants using the Lagrange multiplier.

        Args:
            lagrange: lagrange [l].
            latents: latents [bl].  # TODO: changed from [tbl] to [bl], is this correct?
            extrinsic_cumulants: [b].
            intrinsic_cumulants: [b].

        Returns:
            extrinsic reward r_e
            intrinsic_reward r_d.
        """
        sig_lagrange = jax.nn.sigmoid(lagrange) 
        latent_sig_lagrange = jnp.matmul(latents, sig_lagrange)  # b
        # No diversity rewards for latent 0, only maximize extrinsic reward
        intrinsic_cumulants *= (1 - latents[:, 0])

        first_latent_mask = latents[:, 0]
        # Put 1 in first_latent_mask else put latent_sig_lagrange
        latent_sig_lagrange = jnp.where(first_latent_mask.ravel(),
                                        jnp.ones_like(latent_sig_lagrange.ravel()),
                                        latent_sig_lagrange.ravel())

        return (1 - latent_sig_lagrange) * intrinsic_cumulants + latent_sig_lagrange * extrinsic_cumulants

    @staticmethod
    def lagrangian_loss(lagrange_params, r_avg, optimality_ratio):
        """Loss function for the Lagrange multiplier.

        Args:
            lagrange_params: lagrange [l].
            r: moving averages of reward [l].
            optimality_ratio: [1].
        """
        l_ = jax.nn.sigmoid(lagrange_params["params"])
        loss = jnp.sum(l_[1:] * (r_avg[1:] - r_avg[0] * optimality_ratio))
        return loss

    def __init__(self, config: DOMINOConfig, action_size: int):
        self._config: DOMINOConfig = config
        if self._config.normalize_observations:
            raise NotImplementedError("Normalization is not implemented for DIAYN yet")

        # define the networks
        self._policy, self._critic = make_domino_networks(
            action_size=action_size,
            hidden_layer_sizes=self._config.hidden_layer_sizes,
        )

        # define the action distribution
        self._action_size = action_size
        self._parametric_action_distribution = NormalTanhDistribution(
            event_size=action_size
        )
        self._sample_action_fn = self._parametric_action_distribution.sample

        # define the optimizers
        self._policy_optimizer = optax.adam(learning_rate=self._config.learning_rate)
        self._critic_optimizer = optax.adam(learning_rate=self._config.learning_rate)
        self._alpha_optimizer = optax.adam(learning_rate=self._config.learning_rate)
        self._lagrange_optimizer = optax.adam(
            learning_rate=self._config.learning_rate_lagrange
        )

    def init(  # type: ignore
        self,
        random_key: RNGKey,
        action_size: int,
        observation_size: int,
        descriptor_size: int,
    ) -> DOMINOTrainingState:
        """Initialise the training state of the algorithm.

        Args:
            random_key: a jax random key
            action_size: the size of the environment's action space
            observation_size: the size of the environment's observation space
            descriptor_size: the size of the environment's descriptor space (i.e. the
                dimension of the discriminator's input)

        Returns:
            the initial training state of DIAYN
        """

        # define policy and critic params
        dummy_obs = jnp.zeros((1, observation_size))
        dummy_action = jnp.zeros((1, action_size))

        random_key, subkey = jax.random.split(random_key)
        policy_params = self._policy.init(subkey, dummy_obs)

        random_key, subkey = jax.random.split(random_key)
        subkeys_critic = jax.random.split(subkey, 2)
        critic_params = jax.vmap(self._critic.init, in_axes=(0, None, None))(subkeys_critic, dummy_obs, dummy_action)

        target_critic_params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x.copy()), critic_params
        )

        random_key, subkey = jax.random.split(random_key)
        lagrange_coefficients = {"params": jnp.array(0.)}

        policy_optimizer_state = self._policy_optimizer.init(policy_params)
        critic_optimizer_state = jax.vmap(self._critic_optimizer.init)(critic_params)
        lagrange_optimizer_state = self._lagrange_optimizer.init(
            lagrange_coefficients
        )

        log_alpha = jnp.asarray(jnp.log(self._config.alpha_init), dtype=jnp.float32)
        alpha_optimizer_state = self._alpha_optimizer.init(log_alpha)

        return DOMINOTrainingState.create(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=log_alpha,

            target_critic_params=target_critic_params,
            random_key=random_key,
            steps=jnp.array(0),
            avg_values=jnp.array(0.),
            avg_sfs=jnp.zeros(descriptor_size,),

            lagrange_optimizer_state=lagrange_optimizer_state,
            lagrange_params=lagrange_coefficients,
        )

    @partial(jax.jit, static_argnames=("self",))
    def _sample_z_from_prior(self, random_key):
        index = jax.random.randint(random_key, shape=(), minval=0, maxval=self._config.num_skills)
        return jax.nn.one_hot(index, num_classes=self._config.num_skills)

    @partial(jax.jit, static_argnames=("self",))
    def _compute_diversity_reward(
        self,
        transition: DOMINOTransition,
        avg_sfs: jnp.ndarray,
        skill,
    ) -> Reward:
        """Computes the diversity of reward of DOMINO.

        Args:
            transition: a batch of transitions from the replay buffer
            training_state: the current training state

        Returns:
            the diversity reward
        """

        skills = jnp.repeat(jnp.expand_dims(skill, axis=0), repeats=transition.state_desc.shape[0], axis=0)
        # repeat skill transition.state_desc.shape[0] times

        rewards = self.intrinsic_reward(
            phi=transition.state_desc,
            sfs=avg_sfs,
            latents=skills,
        )

        return rewards

    @partial(jax.jit, static_argnames=("self", "env", "deterministic"))
    def play_step_fn(
        self,
        env_state: EnvState,
        training_state: DOMINOTrainingState,
        env: QDEnv,
        deterministic: bool = False,
    ) -> Tuple[EnvState, DOMINOTrainingState, QDTransition]:
        """Plays a step in the environment. Concatenates skills to the observation
        vector, selects an action according to SAC rule and performs the environment
        step.

        Args:
            env_state: the current environment state
            training_state: the DIAYN training state
            skills: the skills concatenated to the observation vector
            env: the environment
            deterministic: whether or not to select action in a deterministic way.
                Defaults to False.

        Returns:
            the new environment state
            the new DIAYN training state
            the played transition
        """

        random_key = training_state.random_key
        policy_params = training_state.policy_params

        # Sample skill from prior when new episode starts
        random_keys = jax.random.split(random_key, env_state.obs.shape[0]+1)
        random_key, random_keys = random_keys[0], random_keys[1:]

        obs = env_state.obs

        # If the env does not support state descriptor, we set it to (0,0)
        if "state_descriptor" in env_state.info:
            state_desc = env_state.info["state_descriptor"]
        else:
            state_desc = jnp.zeros((env_state.obs.shape[0], 2))

        actions, random_key = self.select_action(
            obs=obs,
            policy_params=policy_params,
            random_key=random_key,
            deterministic=deterministic,
        )

        next_env_state = env.step(env_state, actions)
        next_obs = next_env_state.obs
        if "state_descriptor" in next_env_state.info:
            next_state_desc = next_env_state.info["state_descriptor"]
        else:
            next_state_desc = jnp.zeros((next_env_state.obs.shape[0], 2))
        truncations = next_env_state.info["truncation"]
        transition = DOMINOTransition(
            obs=obs,
            next_obs=next_obs,
            state_desc=state_desc,
            next_state_desc=next_state_desc,
            rewards=next_env_state.reward,
            dones=next_env_state.done,
            actions=actions,
            truncations=truncations,
            desc=jnp.zeros((env_state.obs.shape[0], env.behavior_descriptor_length,)) * jnp.nan,
            desc_prime=jnp.zeros((env_state.obs.shape[0], env.behavior_descriptor_length,)) * jnp.nan,
        )
        training_state = training_state.replace(random_key=random_key)

        return next_env_state, training_state, transition

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "play_step_fn",
            "env",
        ),
    )
    def eval_policy_2_fn(
        self,
        training_state: DOMINOTrainingState,
        play_step_fn: Callable[
            [EnvState, Params, RNGKey],
            Tuple[EnvState, Params, RNGKey, QDTransition],
        ],
        env,
    ) -> Tuple[Reward, Reward, Reward, StateDescriptor]:
        """Evaluates one agent's policy over an entire episode, across all batched
        environments.


        Args:
            training_state: the DIAYN training state
            play_step_fn: the play_step function used to collect the evaluation episode

        Returns:
            true return averaged over batch dimension, shape: (1,)
            true return per environment, shape: (env_batch_size,)
            diversity return per environment, shape: (env_batch_size,)
            state descriptors, shape: (episode_length, env_batch_size, descriptor_size)

        """
        # Init state
        random_key = training_state.random_key
        random_key, random_subkey = jax.random.split(random_key)
        init_state = env.reset(random_subkey)

        # Sample skills
        # init_state.info["skills"] = jnp.eye(self._config.num_skills)

        random_key, subkey = jax.random.split(random_key)
        training_state = training_state.replace(random_key=subkey)

        # Rollout
        state, training_state, transitions = generate_unroll(
            init_state=init_state,
            training_state=training_state,
            episode_length=self._config.episode_length,
            play_step_fn=play_step_fn,
        )
        training_state: DOMINOTrainingState

        transitions = get_first_episode(transitions)
        true_returns = jnp.nansum(transitions.rewards, axis=0)
        #  true_return = jnp.mean(true_returns, axis=-1)
        true_return = true_returns

        reshaped_transitions = jax.tree_util.tree_map(
            lambda x: x.reshape((self._config.episode_length, -1)),
            transitions,
        )

        if self._config.descriptor_full_state:
            state_desc = reshaped_transitions.obs[:, :-self._config.num_skills]
            next_state_desc = reshaped_transitions.next_obs[
                :, : -self._config.num_skills
            ]
            reshaped_transitions = reshaped_transitions.replace(
                state_desc=state_desc, next_state_desc=next_state_desc
            )

        return (
            true_return,
            true_returns,
            transitions.state_desc,
        )

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "play_step_fn",
            "env",
        ),
    )
    def eval_policy_fn(
        self,
        training_state: DOMINOTrainingState,
        skill,
        avg_sfs,
        play_step_fn: Callable[
            [EnvState, Params, RNGKey],
            Tuple[EnvState, Params, RNGKey, QDTransition],
        ],
        env,
    ) -> Tuple[Reward, Reward, Reward, StateDescriptor]:
        """Evaluates the agent's policy over an entire episode, across all batched
        environments.


        Args:
            training_state: the DIAYN training state
            play_step_fn: the play_step function used to collect the evaluation episode

        Returns:
            true return averaged over batch dimension, shape: (1,)
            true return per environment, shape: (env_batch_size,)
            diversity return per environment, shape: (env_batch_size,)
            state descriptors, shape: (episode_length, env_batch_size, descriptor_size)

        """
        # Init state
        random_key = training_state.random_key
        random_key, random_subkey = jax.random.split(random_key)
        init_state = env.reset(random_subkey)

        # Sample skills
        # init_state.info["skills"] = jnp.eye(self._config.num_skills)

        random_key, subkey = jax.random.split(random_key)
        training_state = training_state.replace(random_key=subkey)

        # Rollout
        state, training_state, transitions = generate_unroll(
            init_state=init_state,
            training_state=training_state,
            episode_length=self._config.episode_length,
            play_step_fn=play_step_fn,
        )
        training_state: DOMINOTrainingState

        transitions = get_first_episode(transitions)
        true_returns = jnp.nansum(transitions.rewards, axis=0)
        true_return = jnp.mean(true_returns, axis=-1)

        reshaped_transitions = jax.tree_util.tree_map(
            lambda x: x.reshape((self._config.episode_length * env.batch_size, -1)),
            transitions,
        )

        if self._config.descriptor_full_state:
            state_desc = reshaped_transitions.obs[:, :-self._config.num_skills]
            next_state_desc = reshaped_transitions.next_obs[
                :, : -self._config.num_skills
            ]
            reshaped_transitions = reshaped_transitions.replace(
                state_desc=state_desc, next_state_desc=next_state_desc
            )

        diversity_rewards = self._compute_diversity_reward(
            reshaped_transitions,
            avg_sfs,
            skill,
        ).reshape((self._config.episode_length, env.batch_size))

        diversity_returns = jnp.nansum(diversity_rewards, axis=0)

        return (
            true_return,
            true_returns,
            diversity_returns,
            transitions.state_desc,
        )

    @staticmethod
    def v_update_avg(previous_avg, array_elements, alpha):
        num_elements = len(array_elements)
        powers = jnp.power(alpha, jnp.arange(num_elements))[::-1]

        if len(array_elements.shape) == 1:
            new_average = jnp.sum((1 - alpha) * powers * array_elements, axis=0) + jnp.power(alpha,
                                                                                             num_elements) * previous_avg
        else:
            new_average = jax.vmap(lambda _x: jnp.sum((1 - alpha) * powers * _x, axis=0), in_axes=1)(
                array_elements) + jnp.power(alpha, num_elements) * previous_avg
        return new_average


    @partial(jax.jit, static_argnames=("self",))
    def _update_networks(
        self,
        training_state_tree: DOMINOTrainingState,
        transitions_tree: QDTransition,
    ) -> Tuple[DOMINOTrainingState, Metrics]:
        """Updates all the networks of the training state.

        Args:
            training_state: the current training state.
            transitions: transitions sampled from the replay buffer.
            random_key: a random key to handle stochastic operations.

        Returns:
            The update training state, metrics and a new random key.
        """
        random_key = training_state_tree.random_key

        # Compute discriminator loss and gradients
        # NO VMAP HERE todo: check that this is correct
        lagrange_loss, lagrange_gradient = jax.value_and_grad(
            self.lagrangian_loss
        )(
            training_state_tree.lagrange_params,
            r_avg=training_state_tree.avg_values,
            optimality_ratio=self._config.optimality_ratio,
        )

        # update discriminator
        (
            lagrange_updates,
            updated_lagrangian_optimizer_state,
        ) = self._lagrange_optimizer.update(
            lagrange_gradient, training_state_tree.lagrange_optimizer_state
        )
        updated_lagrange_params = optax.apply_updates(
            training_state_tree.lagrange_params, lagrange_updates
        )

        updated_lagrange_params = jax.tree_map(lambda x: jnp.clip(x, -15., 15.), updated_lagrange_params)

        # update alpha
        (
            alpha_params,
            alpha_optimizer_state,
            alpha_loss,
            random_key,
        ) = jax.vmap(self._update_alpha, in_axes=(None, 0, 0, 0))(
            self._config.learning_rate,
            training_state_tree,
            transitions_tree,
            random_key,
        )

        # update critic
        all_skills = jnp.eye(self._config.num_skills)
        (
            critic_params,
            target_critic_params,
            critic_optimizer_state,
            critic_loss,
            critic_norm_gradient,
            random_key,
        ) = jax.vmap(self._update_critic, in_axes=(None, None, None, None, 0, 0, 0, 0))(
            self._config.learning_rate,
            self._config.reward_scaling,
            self._config.discount,
            training_state_tree.avg_sfs,  # vmap does not apply to this.
            training_state_tree,
            transitions_tree,
            all_skills,
            random_key,
        )

        # update actor
        (
            policy_params,
            policy_optimizer_state,
            policy_loss,
            random_key,
        ) = jax.vmap(self._update_actor, in_axes=(None, 0, 0, 0, 0))(
            self._config.learning_rate,
            training_state_tree,
            transitions_tree,
            random_key,
            all_skills,
        )

        alpha_d_v_avg = self._config.alpha_d_v_avg
        alpha_d_sfs_avg = self._config.alpha_d_sfs_avg

        new_avg_values = alpha_d_v_avg * training_state_tree.avg_values + (1 - alpha_d_v_avg) * jnp.mean(transitions_tree.rewards, axis=1)
        new_avg_sfs = alpha_d_sfs_avg * training_state_tree.avg_sfs + (1 - alpha_d_sfs_avg) * jnp.mean(transitions_tree.state_desc, axis=1)


        # Create new training state
        new_training_state = DOMINOTrainingState.create(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            target_critic_params=target_critic_params,
            random_key=random_key,
            steps=training_state_tree.steps + 1,
            avg_values=new_avg_values,
            avg_sfs=new_avg_sfs,
            lagrange_optimizer_state=updated_lagrangian_optimizer_state,
            lagrange_params=updated_lagrange_params,
        )

        metrics = {
            "actor_loss": policy_loss,
            "critic_loss": critic_loss,
            "lagrange_loss": lagrange_loss,
            "alpha_loss": alpha_loss,
            "critic_norm_gradient": critic_norm_gradient,
        }

        return new_training_state, metrics

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        training_state_tree: DOMINOTrainingState,
        replay_buffer_tree: ReplayBuffer,
    ) -> Tuple[DOMINOTrainingState, ReplayBuffer, Metrics]:
        """Performs a training step to update the policy, the critic and the
        discriminator parameters.

        Args:
            training_state: the current DIAYN training state
            replay_buffer: the replay buffer

        Returns:
            the updated DIAYN training state
            the replay buffer
            the training metrics
        """
        # Sample a batch of transitions in the buffer
        random_key = training_state_tree.random_key
        transitions_tree, random_key = jax.vmap(ReplayBuffer.sample, in_axes=(0, 0, None))(
            replay_buffer_tree,
            random_key,
            self._config.batch_size,
        )

        # update params of networks in the training state
        new_training_state_tree, metrics = self._update_networks(
            training_state_tree, transitions_tree=transitions_tree,
        )

        # TODO: deal with metrics tree

        return new_training_state_tree, replay_buffer_tree, metrics

    @partial(jax.jit, static_argnames=("self",))
    def _update_critic(
        self,
        critic_lr: float,
        reward_scaling: float,
        discount: float,
        avg_sfs: jnp.ndarray,
        training_state: DOMINOTrainingState,
        transitions: Transition,
        skill,
        random_key: RNGKey,
    ) -> Tuple[Params, Params, optax.OptState, jnp.ndarray, RNGKey]:
        """Updates the critic following the method described in the
        Soft Actor Critic paper.

        Args:
            critic_lr: critic learning rate
            reward_scaling: coefficient to scale rewards
            discount: discount factor
            training_state: the current training state.
            transitions: a batch of transitions sampled from the replay buffer.
            random_key: a random key to handle stochastic operations.

        Returns:
            New parameters of the critic and its target. New optimizer state,
            loss and a new random key.
        """

        # Compute the rewards and replace transitions
        diversity_rewards = self._compute_diversity_reward(transitions, avg_sfs, skill)
        # update critic
        random_key, subkey = jax.random.split(random_key)
        critic_loss, critic_gradient = jax.value_and_grad(self.domino_critic_loss_fn)(
            training_state.critic_params,
            policy_fn=self._policy.apply,
            critic_fn=self._critic.apply,
            parametric_action_distribution=self._parametric_action_distribution,
            reward_scaling=reward_scaling,
            discount=discount,
            policy_params=training_state.policy_params,
            target_critic_params=training_state.target_critic_params,
            alpha=jnp.exp(training_state.alpha_params),
            transitions=transitions,
            diversity_rewards=diversity_rewards,
            random_key=subkey,
        )
        # critic_optimizer = optax.adam(learning_rate=critic_lr)
        flattened_gradient, _ = ravel_pytree(critic_gradient)
        critic_norm_gradient = jnp.linalg.norm(flattened_gradient)
        # critic_optimizer = optax.chain(optax.clip(1.0), optax.adam(learning_rate=critic_lr))
        (critic_updates, critic_optimizer_state,) = jax.vmap(self._critic_optimizer.update)(
            critic_gradient, training_state.critic_optimizer_state
        )
        critic_params = optax.apply_updates(
            training_state.critic_params, critic_updates
        )
        target_critic_params = jax.tree_util.tree_map(
            lambda x1, x2: (1.0 - self._config.tau) * x1 + self._config.tau * x2,
            training_state.target_critic_params,
            critic_params,
        )

        return (
            critic_params,
            target_critic_params,
            critic_optimizer_state,
            critic_loss,
            critic_norm_gradient,
            random_key,
        )

    @partial(jax.jit, static_argnames=("self",))
    def _update_actor(
        self,
        policy_lr: float,
        training_state: DOMINOTrainingState,
        transitions: Transition,
        random_key: RNGKey,
        skill: jnp.ndarray,
    ) -> Tuple[Params, optax.OptState, jnp.ndarray, RNGKey]:
        """Updates the actor parameters following the stochastic
        policy gradient theorem with the method introduced in SAC.

        Args:
            policy_lr: policy learning rate
            training_state: the current training state.
            transitions: a batch of transitions sampled from the replay
                buffer.
            random_key: a random key to handle stochastic operations.

        Returns:
            New params and optimizer state. Current loss. New random key.
        """

        random_key, subkey = jax.random.split(random_key)
        policy_loss, policy_gradient = jax.value_and_grad(self.domino_policy_loss_fn)(
            training_state.policy_params,
            policy_fn=self._policy.apply,
            critic_fn=self._critic.apply,
            parametric_action_distribution=self._parametric_action_distribution,
            critic_params=training_state.critic_params,
            alpha=jnp.exp(training_state.alpha_params),
            lagrange_params=training_state.lagrange_params["params"],
            transitions=transitions,
            random_key=subkey,
            skill=skill,
        )
        policy_optimizer = self._policy_optimizer
        (policy_updates, policy_optimizer_state,) = policy_optimizer.update(
            policy_gradient, training_state.policy_optimizer_state
        )
        policy_params = optax.apply_updates(
            training_state.policy_params, policy_updates
        )

        return policy_params, policy_optimizer_state, policy_loss, random_key
