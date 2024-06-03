"""
A collection of functions and classes that define the algorithm Diversity Is All You
Need (DIAYN), ref: https://arxiv.org/abs/1802.06070.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from brax.envs import Env
from brax.envs import State as EnvState
from brax.training.distribution import NormalTanhDistribution

from baselines.qdax.baselines.sac import SAC, SacConfig
from baselines.qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from baselines.qdax.core.neuroevolution.losses.diayn_loss import make_diayn_loss_fn
from baselines.qdax.core.neuroevolution.mdp_utils import TrainingState, get_first_episode
from baselines.qdax.core.neuroevolution.networks.diayn_networks import make_diayn_networks
from baselines.qdax.core.neuroevolution.sac_td3_utils import generate_unroll
from baselines.qdax.types import Metrics, Params, Reward, RNGKey, Skill, StateDescriptor


class DiaynTrainingState(TrainingState):
    """Training state for the DIAYN algorithm"""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    critic_optimizer_state: optax.OptState
    critic_params: Params
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    target_critic_params: Params
    discriminator_optimizer_state: optax.OptState
    discriminator_params: Params
    random_key: RNGKey
    steps: jnp.ndarray


@dataclass
class DiaynConfig(SacConfig):
    """Configuration for the DIAYN algorithm"""

    skill_type: str = "categorical"
    num_skills: int = 5
    descriptor_full_state: bool = False
    extrinsic_reward: bool = False
    beta: float = 0.


class DIAYN(SAC):
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

    def __init__(self, config: DiaynConfig, action_size: int):
        self._config: DiaynConfig = config
        if self._config.normalize_observations:
            raise NotImplementedError("Normalization is not implemented for DIAYN yet")

        # define the networks
        self._policy, self._critic, self._discriminator = make_diayn_networks(
            skill_type=self._config.skill_type,
            num_skills=self._config.num_skills,
            action_size=action_size,
            hidden_layer_sizes=self._config.hidden_layer_sizes,
        )

        # define the action distribution
        self._action_size = action_size
        self._parametric_action_distribution = NormalTanhDistribution(
            event_size=action_size
        )
        self._sample_action_fn = self._parametric_action_distribution.sample

        # define the losses
        (
            self._alpha_loss_fn,
            self._policy_loss_fn,
            self._critic_loss_fn,
            self._discriminator_loss_fn,
        ) = make_diayn_loss_fn(
            policy_fn=self._policy.apply,
            critic_fn=self._critic.apply,
            discriminator_fn=self._discriminator.apply,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            action_size=action_size,
            skill_type=self._config.skill_type,
            num_skills=self._config.num_skills,
            parametric_action_distribution=self._parametric_action_distribution,
        )

        # define the optimizers
        self._policy_optimizer = optax.adam(learning_rate=self._config.learning_rate)
        self._critic_optimizer = optax.adam(learning_rate=self._config.learning_rate)
        self._alpha_optimizer = optax.adam(learning_rate=self._config.learning_rate)
        self._discriminator_optimizer = optax.adam(
            learning_rate=self._config.learning_rate
        )

    def init(  # type: ignore
        self,
        random_key: RNGKey,
        action_size: int,
        observation_size: int,
        descriptor_size: int,
    ) -> DiaynTrainingState:
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
        dummy_obs = jnp.zeros((1, observation_size + self._config.num_skills))
        dummy_action = jnp.zeros((1, action_size))
        dummy_discriminator_obs = jnp.zeros((1, descriptor_size))

        random_key, subkey = jax.random.split(random_key)
        policy_params = self._policy.init(subkey, dummy_obs)

        random_key, subkey = jax.random.split(random_key)
        critic_params = self._critic.init(subkey, dummy_obs, dummy_action)

        target_critic_params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x.copy()), critic_params
        )

        random_key, subkey = jax.random.split(random_key)
        discriminator_params = self._discriminator.init(
            subkey, obs=dummy_discriminator_obs
        )

        policy_optimizer_state = self._policy_optimizer.init(policy_params)
        critic_optimizer_state = self._critic_optimizer.init(critic_params)
        discriminator_optimizer_state = self._discriminator_optimizer.init(
            discriminator_params
        )

        log_alpha = jnp.asarray(jnp.log(self._config.alpha_init), dtype=jnp.float32)
        alpha_optimizer_state = self._alpha_optimizer.init(log_alpha)

        return DiaynTrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=log_alpha,
            target_critic_params=target_critic_params,
            discriminator_optimizer_state=discriminator_optimizer_state,
            discriminator_params=discriminator_params,
            random_key=random_key,
            steps=jnp.array(0),
        )

    @partial(jax.jit, static_argnames=("self",))
    def _sample_z_from_prior(self, random_key):
        if self._config.skill_type == "categorical":
            index = jax.random.randint(random_key, shape=(), minval=0, maxval=self._config.num_skills)
            return jax.nn.one_hot(index, num_classes=self._config.num_skills)
        elif self._config.skill_type == "normal":
            return jax.random.uniform(random_key, shape=(self._config.num_skills,), minval=-1., maxval=1.)

    @partial(jax.jit, static_argnames=("self", "add_log_p_z"))
    def _compute_diversity_reward(
        self,
        transition: QDTransition,
        discriminator_params: Params,
        add_log_p_z: bool = True,
    ) -> Reward:
        """Computes the diversity reward of DIAYN.

        Args:
            transition: a batch of transitions from the replay buffer
            discriminator_params: the parameters of the discriminator
            add_log_p_z: whether or not to add (minus) the probability of the skills'
                prior distribution. Defaults to True.

        Returns:
            the diversity reward
        """

        next_state_desc = transition.next_state_desc
        skills = transition.next_obs[:, -self._config.num_skills:]
        if self._config.skill_type == "categorical":
            reward = jnp.sum(
                jax.nn.log_softmax(self._discriminator.apply(discriminator_params, next_state_desc)) * skills,
                axis=1,
            )
            if add_log_p_z:
                reward += jnp.log(self._config.num_skills)
        elif self._config.skill_type == "normal":
            mean, log_std = self._discriminator.apply(discriminator_params, next_state_desc)
            cov = jax.vmap(jnp.diag)(jnp.exp(2 * log_std))
            reward = jax.vmap(jax.scipy.stats.multivariate_normal.logpdf)(skills, mean, cov)
            if add_log_p_z:
                reward += self._config.num_skills * jnp.log(2)
        else:
            raise NotImplementedError(f"Not implemented skill type {self._config.skill_type}.")

        return reward

    @partial(jax.jit, static_argnames=("self", "env", "deterministic"))
    def play_step_fn(
        self,
        env_state: EnvState,
        training_state: DiaynTrainingState,
        env: Env,
        deterministic: bool = False,
    ) -> Tuple[EnvState, DiaynTrainingState, QDTransition]:
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
        new_skills = jax.vmap(self._sample_z_from_prior)(random_keys)
        old_skills = env_state.info["skills"]
        condition = (env_state.done == 1.) | (env_state.info["steps"] == 0.)
        skills = jnp.where(jnp.repeat(condition[:, None], repeats=self._config.num_skills, axis=1), new_skills, old_skills)
        env_state.info["skills"] = skills
        obs = jnp.concatenate([env_state.obs, skills], axis=1)

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
        next_obs = jnp.concatenate([next_env_state.obs, skills], axis=1)
        if "state_descriptor" in next_env_state.info:
            next_state_desc = next_env_state.info["state_descriptor"]
        else:
            next_state_desc = jnp.zeros((next_env_state.obs.shape[0], 2))
        truncations = next_env_state.info["truncation"]
        transition = QDTransition(
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
    def eval_policy_fn(
        self,
        training_state: DiaynTrainingState,
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
            eval_env_first_state: the initial state for evaluation
            play_step_fn: the play_step function used to collect the evaluation episode
            env_batch_size: the number of environments we play simultaneously

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
        random_keys = jax.random.split(random_key, env.batch_size+1)
        random_keys, random_key = random_keys[:-1], random_keys[-1]
        init_state.info["skills"] = jax.vmap(self._sample_z_from_prior)(random_keys)
        training_state = training_state.replace(random_key=random_key)

        # Rollout
        state, training_state, transitions = generate_unroll(
            init_state=init_state,
            training_state=training_state,
            episode_length=self._config.episode_length,
            play_step_fn=play_step_fn,
        )

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
            transition=reshaped_transitions,
            discriminator_params=training_state.discriminator_params,
        ).reshape((self._config.episode_length, env.batch_size))

        diversity_returns = jnp.nansum(diversity_rewards, axis=0)

        return (
            true_return,
            true_returns,
            diversity_returns,
            transitions.state_desc,
        )

    @partial(jax.jit, static_argnames=("self",))
    def _compute_reward(
        self, transition: QDTransition, training_state: DiaynTrainingState
    ) -> Reward:
        """Computes the reward to train the networks.

        Args:
            transition: a batch of transitions from the replay buffer
            training_state: the current training state

        Returns:
            the DIAYN diversity reward
        """
        diversity_reward = self._compute_diversity_reward(
            transition=transition,
            discriminator_params=training_state.discriminator_params,
            add_log_p_z=True,
        )
        if self._config.extrinsic_reward:
            return transition.rewards + self._config.beta * diversity_reward
        else:
            return self._config.beta * diversity_reward

    @partial(jax.jit, static_argnames=("self",))
    def _update_networks(
        self,
        training_state: DiaynTrainingState,
        transitions: QDTransition,
    ) -> Tuple[DiaynTrainingState, Metrics]:
        """Updates all the networks of the training state.

        Args:
            training_state: the current training state.
            transitions: transitions sampled from the replay buffer.
            random_key: a random key to handle stochastic operations.

        Returns:
            The update training state, metrics and a new random key.
        """
        random_key = training_state.random_key

        # Compute discriminator loss and gradients
        discriminator_loss, discriminator_gradient = jax.value_and_grad(
            self._discriminator_loss_fn
        )(
            training_state.discriminator_params,
            transitions=transitions,
        )

        # update discriminator
        (
            discriminator_updates,
            discriminator_optimizer_state,
        ) = self._discriminator_optimizer.update(
            discriminator_gradient, training_state.discriminator_optimizer_state
        )
        discriminator_params = optax.apply_updates(
            training_state.discriminator_params, discriminator_updates
        )

        # udpate alpha
        (
            alpha_params,
            alpha_optimizer_state,
            alpha_loss,
            random_key,
        ) = self._update_alpha(
            alpha_lr=self._config.learning_rate,
            training_state=training_state,
            transitions=transitions,
            random_key=random_key,
        )

        # update critic
        (
            critic_params,
            target_critic_params,
            critic_optimizer_state,
            critic_loss,
            random_key,
        ) = self._update_critic(
            critic_lr=self._config.learning_rate,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            training_state=training_state,
            transitions=transitions,
            random_key=random_key,
        )

        # update actor
        (
            policy_params,
            policy_optimizer_state,
            policy_loss,
            random_key,
        ) = self._update_actor(
            policy_lr=self._config.learning_rate,
            training_state=training_state,
            transitions=transitions,
            random_key=random_key,
        )

        # Create new training state
        new_training_state = DiaynTrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            target_critic_params=target_critic_params,
            discriminator_optimizer_state=discriminator_optimizer_state,
            discriminator_params=discriminator_params,
            random_key=random_key,
            steps=training_state.steps + 1,
        )
        metrics = {
            "actor_loss": policy_loss,
            "critic_loss": critic_loss,
            "discriminator_loss": discriminator_loss,
            "alpha_loss": alpha_loss,
        }

        return new_training_state, metrics

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        training_state: DiaynTrainingState,
        replay_buffer: ReplayBuffer,
    ) -> Tuple[DiaynTrainingState, ReplayBuffer, Metrics]:
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
        random_key = training_state.random_key
        transitions, random_key = replay_buffer.sample(
            random_key,
            sample_size=self._config.batch_size,
        )

        # Optionally replace the state descriptor by the observation
        if self._config.descriptor_full_state:
            state_desc = transitions.obs[:, : -self._config.num_skills]
            next_state_desc = transitions.next_obs[:, : -self._config.num_skills]
            transitions = transitions.replace(
                state_desc=state_desc, next_state_desc=next_state_desc
            )

        # Compute the rewards and replace transitions
        rewards = self._compute_reward(transitions, training_state)
        transitions = transitions.replace(rewards=rewards)

        # update params of networks in the training state
        new_training_state, metrics = self._update_networks(
            training_state, transitions=transitions
        )

        return new_training_state, replay_buffer, metrics