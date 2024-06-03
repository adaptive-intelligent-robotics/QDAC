from dataclasses import dataclass
from typing import Callable, Tuple

import flax.linen as nn

from baselines.qdax.core.emitters.multi_emitter import MultiEmitter
from baselines.qdax.core.emitters.qdcg_emitter import QualityDCGConfig, QualityDCGEmitter
from baselines.qdax.core.emitters.standard_emitters import MixingEmitter
from baselines.qdax.environments.base_wrappers import QDEnv
from baselines.qdax.types import Params, RNGKey


@dataclass
class DCGMEConfig:
    """Configuration for DCGME Algorithm"""

    env_batch_size: int = 256
    proportion_mutation_ga: float = 0.5

    # PG emitter
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    num_critic_training_steps: int = 3000
    num_pg_training_steps: int = 150
    batch_size: int = 100
    replay_buffer_size: int = 1_000_000
    discount: float = 0.99
    reward_scaling: float = 1.0
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-3
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    soft_tau_update: float = 0.005
    policy_delay: int = 2

    # DCG-MAP-Elites
    min_bd: float = 0.0
    max_bd: float = 1.0
    lengthscale: float = 0.008


class DCGMEEmitter(MultiEmitter):
    def __init__(
        self,
        config: DCGMEConfig,
        policy_network: nn.Module,
        actor_network: nn.Module,
        env: QDEnv,
        variation_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
    ) -> None:
        self._config = config
        self._policy_network = policy_network
        self._actor_network = actor_network
        self._env = env
        self._variation_fn = variation_fn

        ga_batch_size = int(self._config.proportion_mutation_ga * config.env_batch_size)
        qpg_batch_size = config.env_batch_size - ga_batch_size

        qdcg_config = QualityDCGConfig(
            env_batch_size=qpg_batch_size,
            critic_hidden_layer_size=config.critic_hidden_layer_size,
            num_critic_training_steps=config.num_critic_training_steps,
            num_pg_training_steps=config.num_pg_training_steps,
            batch_size=config.batch_size,
            replay_buffer_size=config.replay_buffer_size,
            discount=config.discount,
            reward_scaling=config.reward_scaling,
            critic_learning_rate=config.critic_learning_rate,
            actor_learning_rate=config.actor_learning_rate,
            policy_learning_rate=config.policy_learning_rate,
            noise_clip=config.noise_clip,
            policy_noise=config.policy_noise,
            soft_tau_update=config.soft_tau_update,
            policy_delay=config.policy_delay,
            min_bd=config.min_bd,
            max_bd=config.max_bd,
            lengthscale=config.lengthscale,
        )

        # define the quality emitter
        q_emitter = QualityDCGEmitter(
            config=qdcg_config,
            policy_network=policy_network,
            actor_network=actor_network,
            env=env
        )

        # define the GA emitter
        ga_emitter = MixingEmitter(
            mutation_fn=lambda x, r: (x, r),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=ga_batch_size,
        )

        super().__init__(emitters=(q_emitter, ga_emitter))
