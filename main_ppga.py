import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import torch

v = torch.ones(1, device='cuda')  # init torch cuda before jax

from baselines.PPGA.algorithm.config_ppga import PPGAConfig

import warnings

from baselines.PPGA.algorithm.train_ppga import train_ppga
from baselines.PPGA.envs.brax_custom.brax_env import make_vec_env_brax_ppga
from baselines.PPGA.utils.utilities import config_wandb, log

import hydra

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


@hydra.main(version_base="1.2", config_path="configs/")
def main(hydra_config):
    # Verify config
    cfg = PPGAConfig.create(hydra_config)
    cfg = cfg.as_dot_dict()

    cfg.num_emitters = 1

    vec_env = make_vec_env_brax_ppga(task_name=hydra_config.task, feat_name=hydra_config.feat, batch_size=cfg.env_batch_size,
                                     seed=cfg.seed, backend=cfg.backend, clip_obs_rew=cfg.clip_obs_rew, episode_length=hydra_config.algo.episode_length)

    cfg.batch_size = int(cfg.env_batch_size * cfg.rollout_length)
    cfg.num_envs = int(cfg.env_batch_size)

    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.obs_shape = vec_env.single_observation_space.shape
    cfg.action_shape = vec_env.single_action_space.shape

    cfg.bd_min = vec_env.behavior_descriptor_limits[0][0]
    cfg.bd_max = vec_env.behavior_descriptor_limits[1][0]

    if cfg.use_wandb:
        config_wandb(batch_size=cfg.batch_size, total_iters=cfg.total_iterations, run_name=cfg.wandb_run_name,
                     wandb_project="PPGA", cfg=cfg)
    outdir = os.path.join(cfg.expdir, str(cfg.seed))
    cfg.outdir = outdir
    assert not os.path.exists(outdir) or cfg.load_scheduler_from_cp is not None or cfg.load_archive_from_cp is not None, \
        f"Warning: experiment dir {outdir} exists. Danger of overwriting previous run"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not cfg.save_scheduler:
        log.warning('Warning. You have set save scheduler to false. Only the archive dataframe will be saved in each '
                    'checkpoint. If you plan to restart this experiment from a checkpoint or wish to have the added '
                    'safety of recovering from a potential crash, it is recommended that you enable save_scheduler.')

    train_ppga(cfg, vec_env)


if __name__ == "__main__":
    main()
