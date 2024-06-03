import argparse
import os
import sys
import time
from distutils.util import strtobool
from attrdict import AttrDict
from baselines.PPGA.utils.utilities import config_wandb, log
from baselines.PPGA.RL.ppo import PPO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help="Choose from [QDAntBulletEnv-v0,"
                                                     "QDHalfCheetahBulletEnv-v0]")
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--use_wandb", default=False, type=lambda x: bool(strtobool(x)),
                        help='Use weights and biases to track the exp')
    parser.add_argument('--wandb_run_name', type=str, default='ppo_ant')
    parser.add_argument('--wandb_group', type=str)
    parser.add_argument('--wandb_project', type=str, default='PPGA')
    parser.add_argument('--report_interval', type=int, default=5, help='Log objective results every N updates')

    # algorithm args
    parser.add_argument('--total_timesteps', type=int, default=1000000)
    parser.add_argument('--env_type', type=str, choices=['brax', 'isaac'], help='Whether to use cpu-envs or gpu-envs for rollouts')
    # args for brax
    parser.add_argument('--env_batch_size', default=1, type=int, help='Number of parallel environments to run')

    # args for cpu-envs
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes to spawn. '
                             'Should always be <= number of logical cores on your machine')
    parser.add_argument('--envs_per_worker', type=int, default=1,
                        help='Num envs each worker process will step through sequentially')
    parser.add_argument('--rollout_length', type=int, default=2048,
                        help='the number of steps to run in each environment per policy rollout')
    # ppo hyperparams
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--anneal_lr', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help='Toggle learning rate annealing for policy and value networks')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='Lambda discount used for general advantage est')
    parser.add_argument('--num_minibatches', type=int, default=32)
    parser.add_argument('--update_epochs', type=int, default=10, help='The K epochs to update the policy')
    parser.add_argument("--norm_adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip_coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip_value_coef", type=float, default=0.2,
                        help="value clipping coefficient")
    parser.add_argument("--clip_vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--entropy_coef", type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None,
                        help="the target KL divergence threshold")
    parser.add_argument('--normalize_obs', type=lambda x: bool(strtobool(x)), default=False, help='Normalize observations across a batch using running mean and stddev')
    parser.add_argument('--normalize_returns', type=lambda x: bool(strtobool(x)), default=False, help='Normalize rewards across a batch using running mean and stddev')
    parser.add_argument('--value_bootstrap', type=lambda x: bool(strtobool(x)), default=False, help='Use bootstrap value estimates')

    parser.add_argument('--weight_decay', type=float, default=None, help='Apply L2 weight regularization to the NNs')
    # vestigial QD params
    parser.add_argument('--num_dims', type=int)

    args = parser.parse_args()
    cfg = AttrDict(vars(args))
    return cfg


if __name__ == '__main__':
    cfg = parse_args()

    if cfg.seed is None:
        cfg.seed = int(time.time()) + int(os.getpid())

    if cfg.env_type == 'brax':
        from baselines.PPGA.envs.brax_custom.brax_env import make_vec_env_brax_ppga
        vec_env = make_vec_env_brax_ppga(cfg)
    else:
        raise NotImplementedError(f'{cfg.env_type} is undefined for "env_type"')

    cfg.batch_size = int(cfg.env_batch_size * cfg.rollout_length)
    cfg.num_envs = int(cfg.env_batch_size)
    cfg.num_emitters = 1
    cfg.envs_per_model = cfg.num_envs // cfg.num_emitters
    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.obs_shape = vec_env.single_observation_space.shape
    cfg.action_shape = vec_env.single_action_space.shape

    log.debug(f'Environment: {cfg.env_name}, obs_shape: {cfg.obs_shape}, action_shape: {cfg.action_shape}')

    if cfg.use_wandb:
        config_wandb(cfg=cfg, batch_size=cfg.batch_size, total_steps=cfg.total_timesteps, run_name=cfg.wandb_run_name, wandb_group=cfg.wandb_group, wandb_project=cfg.wandb_project)

    alg = PPO(cfg)
    num_updates = cfg.total_timesteps // cfg.batch_size
    alg.train(vec_env, num_updates, rollout_length=cfg.rollout_length)
    sys.exit(0)
