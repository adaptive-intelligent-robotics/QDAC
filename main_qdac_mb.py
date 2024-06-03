import warnings

import dreamerv3
from baselines.qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from dreamerv3.agent import ImagActorCritic
from dreamerv3 import embodied
from dreamerv3.embodied.core.goal_sampler import GoalSampler, GoalSamplerCyclic

import hydra
from hydra.core.config_store import ConfigStore
import wandb
from utils.env_utils import Config, get_env, get_argv_from_config
warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


def certify_algo_params(config):
  algo_name = config.algo.name
  use_fixed_lagrangian = config.goal.use_fixed_lagrangian

  if algo_name == "qdac_mb":
    assert not use_fixed_lagrangian, "qdac_mb requires use_fixed_lagrangian=False"
    assert config.critic_type == "sf_v_function", "qdac_mb requires critic_type=sf_v_function"

  elif algo_name == "qdac_mb_fixed_lambda":
    assert use_fixed_lagrangian, "qdac_mb_fixed_lambda requires use_fixed_lagrangian=False"
    assert 0.0 <= config.goal.fixed_lagrangian_coeff <= 1.0, "fixed_lagrangian_coeff must be in [0, 1]"
    assert config.critic_type == "sf_v_function", "qdac_mb_fixed_lambda requires critic_type=sf_v_function"

  elif algo_name == "qdac_mb_no_sf":
    assert not use_fixed_lagrangian, "qdac_mb_no_sf requires use_fixed_lagrangian=False"
    assert config.critic_type == "constraint_v_function", "qdac_mb_no_sf requires critic_type=constraint_v_function"

  elif algo_name == "uvfa":
    assert use_fixed_lagrangian, "uvfa requires use_fixed_lagrangian=True"
    assert 0.0 <= config.goal.fixed_lagrangian_coeff <= 1.0, "fixed_lagrangian_coeff must be in [0, 1]"
    assert config.critic_type == "uvfa_critic_type", "uvfa requires critic_type=uvfa_critic_type"

  else:
    raise NotImplementedError(f"algo_name={algo_name} not implemented")

def get_override_config(name_algo):
  if name_algo == "qdac_mb":
    return {
      "goal.use_fixed_lagrangian": False,
      "critic_type": "sf_v_function",
    }
  elif name_algo == "qdac_mb_fixed_lambda":
    return {
      "goal.use_fixed_lagrangian": True,
      "critic_type": "sf_v_function",
    }
  elif name_algo == "qdac_mb_no_sf":
    return {
      "goal.use_fixed_lagrangian": False,
      "critic_type": "constraint_v_function",
    }
  elif name_algo == "uvfa":
    return {
      "goal.use_fixed_lagrangian": True,
      "critic_type": "uvfa_critic_type",
    }
  else:
    raise NotImplementedError(f"algo_name={name_algo} not implemented")


@hydra.main(version_base="1.2", config_path="configs/")
def main(config):
  name_algo = config.algo.name

  dict_override_config = get_override_config(name_algo)

  # Create config
  logdir = '.'  # hydra automatically changes the working directory
  config_defaults = embodied.Config(dreamerv3.configs["defaults"])
  config_defaults = config_defaults.update(dreamerv3.configs["brax"])
  config_defaults = config_defaults.update({
      "logdir": logdir,
      "run.train_ratio": 32,
      "run.log_every": 60,  # Seconds
      "batch_size": 16,
      **dict_override_config,
  })
  argv = get_argv_from_config(config)
  config = embodied.Flags(config_defaults).parse(argv=argv)

  # Verify config
  certify_algo_params(config)

  # Create logger
  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
      embodied.logger.TensorBoardOutput(logdir),
      embodied.logger.WandBOutput(logdir, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])

  # Create environment
  env = get_env(config, mode="train")

  # Create agent and replay buffer
  agent = dreamerv3.Agent(env.obs_space, env.act_space, env.feat_space, step, config)
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / "replay")
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)

  # Create goal sampler
  goal_sampler = GoalSampler(env.feat_space)
  resolution = ImagActorCritic.get_resolution(env.feat_space, config)
  delta_constraint = ImagActorCritic.calculate_delta_constraint(resolution=resolution, feat_space=env.feat_space)
  wandb.log({"goal/delta_constraint": delta_constraint}, commit=False)
  wandb.log({"goal/resolution_in_practice": resolution}, commit=False)

  # Train or evaluate
  embodied.run.train(agent, env, replay, goal_sampler, config.goal.period_sample_goals, logger, args)

if __name__ == "__main__":
  cs = ConfigStore.instance()
  cs.store(name="config", node=Config)
  main()
