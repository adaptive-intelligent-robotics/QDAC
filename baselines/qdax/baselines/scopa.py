"""
A collection of functions and classes that define the algorithm SCOPA.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.envs import Env
from brax.envs import State as EnvState
from brax.training.distribution import NormalTanhDistribution

from baselines.qdax.baselines.sac import SAC, SacConfig
from baselines.qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from baselines.qdax.core.neuroevolution.losses.scopa_loss import make_scopa_loss_fn
from baselines.qdax.core.neuroevolution.mdp_utils import TrainingState, get_first_episode
from baselines.qdax.core.neuroevolution.networks.networks import MLPDC, QModuleDC, SFModuleDC
from baselines.qdax.core.neuroevolution.sac_td3_utils import generate_unroll
from baselines.qdax.types import Metrics, Params, Reward, RNGKey, Observation, Skill, StateDescriptor, Action
from dreamerv3.embodied.core.space import AngularSpace


class SCOPATrainingState(TrainingState):
    """Training state for the SCOPA algorithm"""

    actor_optimizer_state: optax.OptState
    actor_params: Params
    critic_optimizer_state: optax.OptState
    critic_params: Params
    target_critic_params: Params
    successor_features_optimizer_state: optax.OptState
    successor_features_params: Params
    target_successor_features_params: Params
    lagrange_optimizer_state: optax.OptState
    lagrange_params: Params
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    random_key: RNGKey
    steps: jnp.ndarray


@dataclass
class SCOPAConfig(SacConfig):
    """Configuration for the SCOPA algorithm"""

    delta: float = 0.01
    delta_pseudo_huber_loss: float = 1e-3
    lambda_: float = 1.0
    num_random_skill_batches: int = 0
    normalize_feat: bool = False


class SCOPA:
    """Implements SCOPA algorithm.
    """

    def __init__(self, config: SCOPAConfig, env: Env):
        self._config: SCOPAConfig = config
        if self._config.normalize_observations:
            raise NotImplementedError("Normalization is not implemented for SCOPA yet")

        self._env = env
        self._feat_space = env.feat_space

        self._obs_size = env.observation_size
        self._action_size = env.action_size
        self._feat_size = env.feat_size

        @jax.jit
        def normalize_feat(feat: jnp.ndarray) -> jnp.ndarray:
            if isinstance(self._feat_space["vector"], AngularSpace):
                return feat
            else:
                return 2*(feat - self._feat_space["vector"].low)/(self._feat_space["vector"].high - self._feat_space["vector"].low) - 1

        if self._config.normalize_feat:
            self._normalize_feat = normalize_feat
        else:
            self._normalize_feat = lambda x: x

        # define the networks
        self._actor = MLPDC(
            layer_sizes=self._config.hidden_layer_sizes + (2 * self._action_size,),
            kernel_init=jax.nn.initializers.lecun_uniform(),
        )
        self._critic = QModuleDC(
            hidden_layer_sizes=self._config.hidden_layer_sizes,
            n_critics=2,
        )
        self._successor_features = SFModuleDC(
            hidden_layer_sizes=self._config.hidden_layer_sizes,
            n_successor_features=2,
            feat_size=self._feat_size,
        )
        self._lagrange = MLPDC(
            layer_sizes=self._config.hidden_layer_sizes + (1,),
            kernel_init=jax.nn.initializers.lecun_uniform(),
        )

        # define the action distribution
        self._parametric_action_distribution = NormalTanhDistribution(
            event_size=self._action_size
        )
        self._sample_action_fn = self._parametric_action_distribution.sample

        # define the losses
        (
            self._actor_loss_fn,
            self._critic_loss_fn,
            self._successor_features_loss_fn,
            self._lagrange_loss_fn,
            self._alpha_loss_fn,
        ) = make_scopa_loss_fn(
            actor_fn=self._actor.apply,
            critic_fn=self._critic.apply,
            successor_features_fn=self._successor_features.apply,
            lagrange_fn=self._lagrange.apply,
            parametric_action_distribution=self._parametric_action_distribution,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            action_size=self._action_size,
            delta=self._config.delta,
            delta_pseudo_huber_loss=self._config.delta_pseudo_huber_loss,
            lambda_=self._config.lambda_,
        )

        # define the optimizers
        self._actor_optimizer = optax.adam(learning_rate=self._config.learning_rate)
        self._critic_optimizer = optax.adam(learning_rate=self._config.learning_rate)
        self._successor_features_optimizer = optax.adam(learning_rate=self._config.learning_rate)
        self._lagrange_optimizer = optax.adam(learning_rate=self._config.learning_rate)
        self._alpha_optimizer = optax.adam(learning_rate=self._config.learning_rate)

    def init(self, random_key: RNGKey) -> SCOPATrainingState:
        """Initialise the training state of the algorithm.

        Args:
            random_key: a jax random key
            action_size: the size of the environment's action space
            observation_size: the size of the environment's observation space
            feat_size: the size of the environment's descriptor space

        Returns:
            the initial training state of SCOPA
        """

        # define actor and critic params
        dummy_obs = jnp.zeros((1, self._obs_size,))
        dummy_action = jnp.zeros((1, self._action_size))
        dummy_skill = jnp.zeros((1, self._feat_size,))

        random_key, subkey = jax.random.split(random_key)
        actor_params = self._actor.init(subkey, dummy_obs, dummy_skill)

        random_key, subkey = jax.random.split(random_key)
        critic_params = self._critic.init(subkey, dummy_obs, dummy_action, dummy_skill)

        target_critic_params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x.copy()), critic_params
        )

        random_key, subkey = jax.random.split(random_key)
        successor_features_params = self._successor_features.init(subkey, dummy_obs, dummy_action, dummy_skill)

        target_successor_features_params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x.copy()), successor_features_params
        )

        random_key, subkey = jax.random.split(random_key)
        lagrange_params = self._lagrange.init(subkey, dummy_obs, dummy_skill)

        alpha_params = jnp.asarray(jnp.log(self._config.alpha_init), dtype=jnp.float32)

        actor_optimizer_state = self._actor_optimizer.init(actor_params)
        critic_optimizer_state = self._critic_optimizer.init(critic_params)
        successor_features_optimizer_state = self._successor_features_optimizer.init(successor_features_params)
        lagrange_optimizer_state = self._lagrange_optimizer.init(lagrange_params)
        alpha_optimizer_state = self._alpha_optimizer.init(alpha_params)

        return SCOPATrainingState(
            actor_optimizer_state=actor_optimizer_state,
            actor_params=actor_params,
            critic_optimizer_state=critic_optimizer_state,
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            successor_features_optimizer_state=successor_features_optimizer_state,
            successor_features_params=successor_features_params,
            target_successor_features_params=target_successor_features_params,
            lagrange_optimizer_state=lagrange_optimizer_state,
            lagrange_params=lagrange_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            random_key=random_key,
            steps=jnp.array(0),
        )

    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def select_action(
        self,
        obs: Observation,
        skill: Skill,
        actor_params: Params,
        random_key: RNGKey,
        deterministic: bool = False,
    ) -> Tuple[Action, RNGKey]:
        """Selects an action acording to SAC actor.

        Args:
            obs: agent observation(s)
            skill: agent skill(s)
            actor_params: parameters of the agent's actor
            random_key: jax random key
            deterministic: whether to select action in a deterministic way.
                Defaults to False.

        Returns:
            The selected action and a new random key.
        """
        skill_normalized = self._normalize_feat(skill)
        dist_params = self._actor.apply(actor_params, obs, skill_normalized)
        if not deterministic:
            random_key, key_sample = jax.random.split(random_key)
            actions = self._sample_action_fn(dist_params, key_sample)

        else:
            # The first half of parameters is for mean and the second half for variance
            actions = jax.nn.tanh(dist_params[..., : dist_params.shape[-1] // 2])

        return actions, random_key

    @partial(jax.jit, static_argnames=("self",))
    def _sample_skill(self, random_key):
        values = jax.random.uniform(
            random_key,
            shape=self._feat_space["vector"].shape_no_transform,
            minval=self._feat_space["vector"].low,
            maxval=self._feat_space["vector"].high,
        )
        return self._feat_space["vector"].transform(values)

    @partial(jax.jit, static_argnames=("self", "env", "deterministic"))
    def play_step_fn(
        self,
        env_state: EnvState,
        train_state: SCOPATrainingState,
        env: Env,
        deterministic: bool = False,
    ) -> Tuple[EnvState, SCOPATrainingState, QDTransition]:
        """Plays a step in the environment. Concatenates skills to the observation
        vector, selects an action according to SAC rule and performs the environment
        step.

        Args:
            env_state: the current environment state
            train_state: the SCOPA training state
            skills: the skills concatenated to the observation vector
            deterministic: whether or not to select action in a deterministic way.
                Defaults to False.

        Returns:
            the new environment state
            the new SCOPA training state
            the played transition
        """

        # Sample skill from prior when new episode starts
        random_keys = jax.random.split(train_state.random_key, env_state.obs.shape[0]+1)
        random_key, random_keys = random_keys[0], random_keys[1:]
        new_skills = jax.vmap(self._sample_skill)(random_keys)
        old_skills = env_state.info["skills"]
        condition = (env_state.done == 1.) | (env_state.info["steps"] == 0.)
        skills = jnp.where(condition[..., None], new_skills, old_skills)
        env_state.info["skills"] = skills

        # If the env does not support state descriptor, we set it to (0,0)
        state_desc = env_state.info["feat"]

        actions, random_key = self.select_action(
            obs=env_state.obs,
            skill=skills,
            actor_params=train_state.actor_params,
            random_key=random_key,
            deterministic=deterministic,
        )

        next_env_state = env.step(env_state, actions)
        next_state_desc = next_env_state.info["feat"]
        truncations = next_env_state.info["truncation"]
        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_env_state.obs,
            state_desc=state_desc,
            next_state_desc=next_state_desc,
            rewards=next_env_state.reward,
            dones=next_env_state.done,
            actions=actions,
            truncations=truncations,
            desc=jnp.zeros((env_state.obs.shape[0], env.behavior_descriptor_length,)) * jnp.nan,
            desc_prime=skills,
        )
        train_state = train_state.replace(random_key=random_key)

        return next_env_state, train_state, transition

    @partial(jax.jit, static_argnames=("self", "play_step_fn", "env",),)
    def eval_actor_fn(
        self,
        train_state: SCOPATrainingState,
        play_step_fn: Callable[
            [EnvState, Params, RNGKey],
            Tuple[EnvState, Params, RNGKey, QDTransition],
        ],
        env,
    ):
        """Evaluates the agent's actor over an entire episode, across all batched
        environments.


        Args:
            train_state: the SCOPA training state
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
        random_key = train_state.random_key
        random_key, random_subkey = jax.random.split(random_key)
        init_state = env.reset(random_subkey)
        init_state.info["skills"] = jnp.nan * jnp.ones((env.batch_size, self._feat_size,))

        # Rollout
        _, train_state, transitions = generate_unroll(
            init_state=init_state,
            training_state=train_state,
            episode_length=self._config.episode_length,
            play_step_fn=play_step_fn,
        )

        transitions = get_first_episode(transitions)

        returns = jnp.nansum(transitions.rewards, axis=0)

        lagrange_coeffs = jax.nn.sigmoid(jnp.squeeze(self._lagrange.apply(train_state.lagrange_params, transitions.obs, self._normalize_feat(transitions.desc_prime))))
        lagrange_coeffs_mean = jnp.nanmean(lagrange_coeffs, axis=0)

        observed_skills = jnp.nanmean(transitions.state_desc, axis=0)
        distance_to_skills = jnp.linalg.norm(observed_skills - transitions.desc_prime[0], axis=-1)

        return returns, observed_skills, distance_to_skills, lagrange_coeffs_mean

    @partial(jax.jit, static_argnames=("self",))
    def _update_networks(
        self,
        train_state: SCOPATrainingState,
        transitions: QDTransition,
    ) -> Tuple[SCOPATrainingState, Metrics]:
        """Updates all the networks of the training state.

        Args:
            train_state: the current training state.
            transitions: transitions sampled from the replay buffer.

        Returns:
            The update training state, metrics and a new random key.
        """
        # Normalize descriptors
        transitions = transitions.replace(
            state_desc=self._normalize_feat(transitions.state_desc),
            next_state_desc=self._normalize_feat(transitions.next_state_desc),
            desc_prime=self._normalize_feat(transitions.desc_prime))

        # Udpate lagrange
        train_state, lagrange_loss = self._update_lagrange(
            train_state=train_state,
            transitions=transitions,
        )

        # Udpate alpha
        train_state, alpha_loss = self._update_alpha(
            train_state=train_state,
            transitions=transitions,
        )

        # Update critic
        train_state, critic_loss = self._update_critic(
            train_state=train_state,
            transitions=transitions,
        )

        # Update successor_features
        train_state, successor_features_loss = self._update_successor_features(
            train_state=train_state,
            transitions=transitions,
        )

        # Update actor
        train_state, actor_loss = self._update_actor(
            train_state=train_state,
            transitions=transitions,
        )

        # Update train_state
        train_state = train_state.replace(steps=train_state.steps + 1)
        metrics = {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "successor_features_loss": successor_features_loss,
            "lagrange_loss": lagrange_loss,
            "alpha_loss": alpha_loss,
        }

        return train_state, metrics

    @partial(jax.jit, static_argnames=("self",))
    def _update_lagrange(
        self,
        train_state: SCOPATrainingState,
        transitions: QDTransition,
    ):
        """Updates the lagrange following the method described in the SCOPA
        paper.

        Args:
            train_state: the current training state.
            transitions: a batch of transitions sampled from the replay buffer.

        Returns:
            New train state and loss.
        """
        random_key, subkey = jax.random.split(train_state.random_key)
        lagrange_loss, lagrange_gradient = jax.value_and_grad(self._lagrange_loss_fn)(
            train_state.lagrange_params,
            actor_params=train_state.actor_params,
            successor_features_params=train_state.successor_features_params,
            transitions=transitions,
            random_key=subkey,
        )
        (lagrange_updates, lagrange_optimizer_state,) = self._lagrange_optimizer.update(
            lagrange_gradient, train_state.lagrange_optimizer_state
        )
        lagrange_params = optax.apply_updates(train_state.lagrange_params, lagrange_updates)

        train_state = train_state.replace(
            lagrange_params=lagrange_params,
            lagrange_optimizer_state=lagrange_optimizer_state,
            random_key=random_key,)
        return train_state, lagrange_loss

    @partial(jax.jit, static_argnames=("self",))
    def _update_alpha(
        self,
        train_state: SCOPATrainingState,
        transitions: QDTransition,
    ) -> Tuple[Params, optax.OptState, jnp.ndarray, RNGKey]:
        """Updates the alpha parameter if necessary. Else, it keeps the
        current value.

        Args:
            train_state: the current training state.
            transitions: a sample of transitions from the replay buffer.

        Returns:
            New train state and loss.
        """
        if not self._config.fix_alpha:
            random_key, subkey = jax.random.split(train_state.random_key)
            alpha_loss, alpha_gradient = jax.value_and_grad(self._alpha_loss_fn)(
                train_state.alpha_params,
                actor_params=train_state.actor_params,
                transitions=transitions,
                random_key=subkey,
            )
            (alpha_updates, alpha_optimizer_state,) = self._alpha_optimizer.update(
                alpha_gradient, train_state.alpha_optimizer_state
            )
            alpha_params = optax.apply_updates(train_state.alpha_params, alpha_updates)
        else:
            alpha_params = train_state.alpha_params
            alpha_optimizer_state = train_state.alpha_optimizer_state
            alpha_loss = jnp.array(0.0)

        train_state = train_state.replace(
            alpha_params=alpha_params,
            alpha_optimizer_state=alpha_optimizer_state,
            random_key=random_key,
        )
        return train_state, alpha_loss

    @partial(jax.jit, static_argnames=("self",))
    def _update_critic(
        self,
        train_state: SCOPATrainingState,
        transitions: QDTransition,
    ) -> Tuple[Params, Params, optax.OptState, jnp.ndarray, RNGKey]:
        """Updates the critic following the method described in the
        Soft Actor Critic paper.

        Args:
            train_state: the current training state.
            transitions: a batch of transitions sampled from the replay buffer.

        Returns:
            New train state and loss.
        """
        random_key, subkey = jax.random.split(train_state.random_key)
        critic_loss, critic_gradient = jax.value_and_grad(self._critic_loss_fn)(
            train_state.critic_params,
            target_critic_params=train_state.target_critic_params,
            actor_params=train_state.actor_params,
            alpha=jnp.exp(train_state.alpha_params),
            transitions=transitions,
            random_key=subkey,
        )
        (critic_updates, critic_optimizer_state,) = self._critic_optimizer.update(
            critic_gradient, train_state.critic_optimizer_state
        )
        critic_params = optax.apply_updates(train_state.critic_params, critic_updates)
        target_critic_params = jax.tree_util.tree_map(
            lambda x1, x2: (1.0 - self._config.tau) * x1 + self._config.tau * x2,
            train_state.target_critic_params,
            critic_params,
        )

        train_state = train_state.replace(
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            critic_optimizer_state=critic_optimizer_state,
            random_key=random_key,
        )
        return train_state, critic_loss

    @partial(jax.jit, static_argnames=("self",))
    def _update_successor_features(
        self,
        train_state: SCOPATrainingState,
        transitions: QDTransition,
    ) -> Tuple[Params, Params, optax.OptState, jnp.ndarray, RNGKey]:
        """Updates the successor_features following the method described in the
        SCOPA paper.

        Args:
            train_state: the current training state.
            transitions: a batch of transitions sampled from the replay buffer.

        Returns:
            New train state and loss.
        """
        random_key, subkey = jax.random.split(train_state.random_key)
        successor_features_loss, successor_features_gradient = jax.value_and_grad(self._successor_features_loss_fn)(
            train_state.successor_features_params,
            target_successor_features_params=train_state.target_successor_features_params,
            actor_params=train_state.actor_params,
            alpha=jnp.exp(train_state.alpha_params),
            transitions=transitions,
            random_key=subkey,
        )
        (successor_features_updates, successor_features_optimizer_state,) = self._successor_features_optimizer.update(
            successor_features_gradient, train_state.successor_features_optimizer_state
        )
        successor_features_params = optax.apply_updates(train_state.successor_features_params, successor_features_updates)
        target_successor_features_params = jax.tree_util.tree_map(
            lambda x1, x2: (1.0 - self._config.tau) * x1 + self._config.tau * x2,
            train_state.target_successor_features_params,
            successor_features_params,
        )

        train_state = train_state.replace(
            successor_features_params=successor_features_params,
            target_successor_features_params=target_successor_features_params,
            successor_features_optimizer_state=successor_features_optimizer_state,
            random_key=random_key,
        )
        return train_state, successor_features_loss

    @partial(jax.jit, static_argnames=("self",))
    def _update_actor(
        self,
        train_state: SCOPATrainingState,
        transitions: QDTransition,
    ) -> Tuple[Params, optax.OptState, jnp.ndarray, RNGKey]:
        """Updates the actor parameters following the stochastic
        actor gradient theorem with the method introduced in SAC.

        Args:
            train_state: the current training state.
            transitions: a batch of transitions sampled from the replay
                buffer.

        Returns:
            New train state and loss.
        """
        random_key, subkey = jax.random.split(train_state.random_key)
        actor_loss, actor_gradient = jax.value_and_grad(self._actor_loss_fn)(
            train_state.actor_params,
            critic_params=train_state.critic_params,
            successor_features_params=train_state.successor_features_params,
            lagrange_params=train_state.lagrange_params,
            alpha=jnp.exp(train_state.alpha_params),
            transitions=transitions,
            random_key=subkey,
        )
        (actor_updates, actor_optimizer_state,) = self._actor_optimizer.update(
            actor_gradient, train_state.actor_optimizer_state
        )
        actor_params = optax.apply_updates(train_state.actor_params, actor_updates)

        train_state = train_state.replace(
            actor_params=actor_params,
            actor_optimizer_state=actor_optimizer_state,
            random_key=random_key,
        )
        return train_state, actor_loss

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        train_state: SCOPATrainingState,
        replay_buffer: ReplayBuffer,
    ) -> Tuple[SCOPATrainingState, ReplayBuffer, Metrics]:
        """Performs a training step to update the parameters.

        Args:
            train_state: the current SCOPA training state
            replay_buffer: the replay buffer

        Returns:
            the updated SCOPA training state
            the replay buffer
            the training metrics
        """
        # Sample a batch of transitions in the buffer
        transitions, random_key = replay_buffer.sample(
            train_state.random_key,
            sample_size=self._config.batch_size,
        )

        # Sample random skills
        transitions_list = [transitions]
        for i in range(self._config.num_random_skill_batches):
            random_key, subkey = jax.random.split(random_key)
            keys = jax.random.split(subkey, self._config.batch_size)
            sampled_desc_prime = jax.vmap(self._sample_skill)(keys)
            transitions_list.append(transitions.replace(desc_prime=sampled_desc_prime))

        transitions = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), *transitions_list)

        # update params of networks in the training state
        train_state = train_state.replace(random_key=random_key)
        train_state, metrics = self._update_networks(
            train_state,
            transitions
        )

        return train_state, replay_buffer, metrics
