import argparse
import time
import os
import sys
import csv
import torch
import pickle
import numpy as np
import wandb
import shutil
from pathlib import Path

from distutils.util import strtobool
from ribs.archives import CVTArchive, GridArchive
from ribs.emitters import GradientAborescenceEmitter, PPGAEmitter
from ribs.schedulers import Scheduler

from baselines.PPGA.algorithm.config_ppga import DotDict
from baselines.PPGA.RL.ppo import PPO
from baselines.PPGA.utils.utilities import log, config_wandb, get_checkpoints, set_file_handler
from baselines.PPGA.models.actor_critic import Actor
from baselines.PPGA.envs.brax_custom.brax_env import make_vec_env_brax_ppga
from baselines.PPGA.utils.normalize import ReturnNormalizer, ObsNormalizer
from baselines.PPGA.utils.utilities import save_cfg
from baselines.PPGA.utils.archive_utils import save_heatmap, load_scheduler_from_checkpoint, archive_df_to_archive
from baselines.PPGA.envs.brax_custom import reward_offset


def parse_args():
    # PPO params
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)  # TODO
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=False, nargs="?",
                        const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--use_wandb", default=False, type=lambda x: bool(strtobool(x)),
                        help='Use weights and biases to track the exp')
    parser.add_argument('--wandb_run_name', type=str, default='ppo_ant')
    parser.add_argument('--wandb_group', type=str)
    parser.add_argument('--wandb_project', type=str, default='PPGA')

    # args for brax
    parser.add_argument('--env_batch_size', default=1, type=int, help='Number of parallel environments to run')

    # ppo hyperparams
    parser.add_argument('--report_interval', type=int, default=5, help='Log objective results every N updates')
    parser.add_argument('--rollout_length', type=int, default=2048,
                        help='the number of steps to run in each environment per policy rollout')
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
    parser.add_argument("--clip_vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--clip_value_coef", type=float, default=0.2,
                        help="value clipping coefficient")
    parser.add_argument("--entropy_coef", type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None,
                        help="the target KL divergence threshold")
    parser.add_argument('--normalize_obs', type=lambda x: bool(strtobool(x)), default=False,
                        help='Normalize observations across a batch using running mean and stddev')
    parser.add_argument('--normalize_returns', type=lambda x: bool(strtobool(x)), default=False,
                        help='Normalize returns across a batch using running mean and stddev')
    parser.add_argument('--value_bootstrap', type=lambda x: bool(strtobool(x)), default=False,
                        help='Use bootstrap value estimates')
    parser.add_argument('--weight_decay', type=float, default=None, help='Apply L2 weight regularization to the NNs')
    parser.add_argument('--clip_obs_rew', type=lambda x: bool(strtobool(x)), default=False, help='Clip obs and rewards b/w -10 and 10')

    # QD Params
    parser.add_argument("--num_emitters", type=int, default=1, help="Number of parallel"
                                                                    " CMA-ES instances exploring the archive")
    parser.add_argument('--grid_size', type=int, required=True, help='Number of cells per archive dimension')
    parser.add_argument("--num_dims", type=int, required=True, help="Dimensionality of measures")
    parser.add_argument("--popsize", type=int, required=True,
                        help="Branching factor for each step of MEGA i.e. the number of branching solutions from the current solution point")
    parser.add_argument('--log_arch_freq', type=int, default=10,
                        help='Frequency in num iterations at which we checkpoint the archive')
    parser.add_argument('--save_scheduler', type=lambda x: bool(strtobool(x)), default=True,
                        help='Choose whether or not to save the scheduler during checkpointing. If the archive is too big,'
                             'it may be impractical to save both the scheduler and the archive_df. However, you cannot later restart from '
                             'a scheduler checkpoint and instead will have to restart from an archive_df checkpoint, which may impact the performance of the run.')
    parser.add_argument('--load_scheduler_from_cp', type=str, default=None,
                        help='Load an existing QD scheduler from a checkpoint path')
    parser.add_argument('--load_archive_from_cp', type=str, default=None,
                        help='Load an existing archive from a checkpoint path. This can be used as an alternative to loading the scheduler if save_scheduler'
                             'was disabled and only the archive df checkpoint is available. However, this can affect the performance of the run. Cannot be used together with save_scheduler')
    parser.add_argument('--total_iterations', type=int, default=100,
                        help='Number of iterations to run the entire dqd-rl loop')
    parser.add_argument('--dqd_algorithm', type=str, choices=['cma_mega_adam', 'cma_maega'],
                        help='Which DQD algorithm should be running in the outer loop')
    parser.add_argument('--expdir', type=str, help='Experiment results directory')
    parser.add_argument('--save_heatmaps', type=lambda x: bool(strtobool(x)), default=True,
                        help='Save the archive heatmaps. Only applies to archives with <= 2 measures')
    parser.add_argument('--use_surrogate_archive', type=lambda x: bool(strtobool(x)), default=False,
                        help="Use a surrogate archive at a higher resolution to get a better gradient signal for DQD")
    parser.add_argument('--sigma0', type=float, default=1.0,
                        help='Initial standard deviation parameter for the covariance matrix used in NES methods')
    parser.add_argument('--restart_rule', type=str, choices=['basic', 'no_improvement'])
    parser.add_argument('--calc_gradient_iters', type=int,
                        help='Number of iters to run PPO when estimating the objective-measure gradients (N1)')
    parser.add_argument('--move_mean_iters', type=int,
                        help='Number of iterations to run PPO when moving the mean solution point (N2)')
    parser.add_argument('--archive_lr', type=float, help='Archive learning rate for MAEGA')
    parser.add_argument('--threshold_min', type=float, default=0.0,
                        help='Min objective threshold for adding new solutions to the archive')
    parser.add_argument('--take_archive_snapshots', type=lambda x: bool(strtobool(x)), default=False,
                        help='Log the objective scores in every cell in the archive every log_freq iterations. Useful for pretty visualizations')
    parser.add_argument('--adaptive_stddev', type=lambda x: bool(strtobool(x)), default=True,
                        help='If False, the log stddev parameter in the actor will be reset on each QD iteration. Can potentially help exploration but may lose performance')

    args = parser.parse_args()

    cfg = dict(vars(args))
    cfg = DotDict(cfg)
    return cfg


def save_scheduler(scheduler, save_path):
    # cannot pickle generator objects so need to remove it temporarily
    gen = scheduler.emitters[0].opt.problem._generator
    scheduler.emitters[0].opt.problem._generator = None
    # save the scheduler for checkpointing
    with open(save_path, 'wb') as f:
        pickle.dump(scheduler, f, protocol=pickle.HIGHEST_PROTOCOL)
    scheduler.emitters[0].opt.problem._generator = gen


def create_scheduler(cfg,
                     archive_learning_rate: float = None,
                     use_result_archive: bool = True,
                     initial_sol: np.ndarray = None):
    '''Creates a scheduler that uses the ppga emitter
        Args:
        cfg (AttrDict): config file
        archive_learning_rate (float): Learning rate of archive.
        use_result_archive (bool): Whether to use a separate archive to store
            the results.
        initial_sol: initial solution (agent)
    Returns:
        ribs.schedulers.Scheduler: A ribs scheduler for running the algorithm.
    """
    '''
    num_emitters = 1
    obs_shape, action_shape = cfg.obs_shape, cfg.action_shape
    action_dim, obs_dim = np.prod(action_shape), np.prod(obs_shape)
    log.debug(f'Environment {cfg.env_name}, {action_dim=}, {obs_dim=}')
    batch_size = cfg.popsize
    # empirically calculated for brax envs to make the qd-score strictly positive
    qd_offset = reward_offset[cfg.env_name.split("_")[0]]

    if initial_sol is None:
        initial_agent = Actor(obs_shape, action_shape, cfg.normalize_obs, cfg.normalize_returns)
        initial_sol = initial_agent.serialize()
    solution_dim = len(initial_sol)
    mode = 'batch'
    # threshold for adding solutions to the archive
    threshold_min = -np.inf

    bounds = [(cfg.bd_min, cfg.bd_max)] * cfg.num_dims
    archive_dims = [cfg.grid_size] * cfg.num_dims

    if cfg.dqd_algorithm == 'cma_maega':
        threshold_min = cfg.threshold_min

    if archive_learning_rate is None:
        if cfg.dqd_algorithm == 'cma_maega':
            archive_learning_rate = cfg.archive_lr
        else:
            archive_learning_rate = 1.0

    archive, result_archive = None, None
    if cfg.load_archive_from_cp is not None and cfg.load_scheduler_from_cp is None:
        log.info('Loading an existing archive dataframe...')
        with open(cfg.load_archive_from_cp, 'rb') as f:
            archive_df = pickle.load(f)
        archive = archive_df_to_archive(archive_df,
                                        solution_dim=solution_dim,
                                        dims=archive_dims,
                                        ranges=bounds,
                                        learning_rate=archive_learning_rate,
                                        threshold_min=threshold_min,
                                        seed=cfg.seed,
                                        qd_offset=qd_offset)

        if use_result_archive:
            result_archive = archive_df_to_archive(archive_df,
                                                   solution_dim=solution_dim,
                                                   dims=archive_dims,
                                                   ranges=bounds,
                                                   seed=cfg.seed,
                                                   qd_offset=qd_offset)
    else:
        archive = GridArchive(solution_dim=solution_dim,
                              dims=archive_dims,
                              ranges=bounds,
                              learning_rate=archive_learning_rate,
                              threshold_min=threshold_min,
                              seed=cfg.seed,
                              qd_offset=qd_offset)

        if use_result_archive:
            result_archive = GridArchive(solution_dim=solution_dim,
                                         dims=archive_dims,
                                         ranges=bounds,
                                         seed=cfg.seed,
                                         qd_offset=qd_offset)

    ppo = PPO(cfg)

    # Create emitters. Each emitter needs a different seed, so that they do not
    # all do the same thing.
    emitter_seeds = [None] * num_emitters if cfg.seed is None else np.arange(
        cfg.seed, cfg.seed + num_emitters)

    if cfg.dqd_algorithm == 'cma_mega_adam':
        # Note that only one emitter is used for cma_mega_adam. This is to be
        # consistent with Fontaine 2021 <https://arxiv.org/abs/2106.03894>.
        emitters = [
            PPGAEmitter(
                ppo,
                archive,
                initial_sol,
                sigma0=cfg.sigma0,
                batch_size=batch_size,
                seed=emitter_seeds[0],
                use_wandb=cfg.use_wandb,
                normalize_obs=cfg.normalize_obs,
                normalize_returns=cfg.normalize_returns,
            )
        ]
    else:
        # cma_maega
        emitters = [
            PPGAEmitter(
                ppo,
                archive,
                initial_sol,
                sigma0=cfg.sigma0,
                batch_size=batch_size,
                ranker='imp',
                restart_rule=cfg.restart_rule,
                bounds=None,
                seed=emitter_seeds[0],
                use_wandb=cfg.use_wandb,
                normalize_obs=cfg.normalize_obs,
                normalize_returns=cfg.normalize_returns,
            )
        ]

    log.debug(
        f"Created Scheduler for {cfg.dqd_algorithm} with an archive learning rate of {archive_learning_rate}, "
        f"and add mode {mode}, using solution dim {solution_dim} and archive "
        f"dims {archive_dims}. Min threshold is {threshold_min}. Restart rule is {cfg.restart_rule}")

    return Scheduler(archive, emitters, result_archive, add_mode=mode, reward_offset=qd_offset)


def train_ppga(cfg, vec_env):
    # setup logging
    exp_dir = Path(cfg.outdir)
    logdir = exp_dir.joinpath(Path('logs'))
    if not logdir.is_dir():
        logdir.mkdir()
    set_file_handler(logdir)
    save_cfg(str(exp_dir), cfg)

    # checkpointing
    cp_dir = exp_dir.joinpath(Path('checkpoints'))
    if not cp_dir.is_dir():
        cp_dir.mkdir()

    # (optional) save 2d archive heatmaps
    heatmap_dir = exp_dir.joinpath(Path('heatmaps'))
    if not heatmap_dir.is_dir():
        heatmap_dir.mkdir()

    # path to summary file
    summary_filename = os.path.join(str(exp_dir), 'summary.csv')
    if os.path.exists(summary_filename):
        os.remove(summary_filename)
    with open(summary_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'QD-Score', 'Coverage', 'Maximum', 'Average'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_result_archive = cfg.dqd_algorithm == 'cma_maega'

    if cfg.load_scheduler_from_cp:
        log.info("Loading an existing scheduler!")
        scheduler = load_scheduler_from_checkpoint(cfg.load_scheduler_from_cp, cfg.seed, device)
    else:
        scheduler = create_scheduler(cfg, use_result_archive=use_result_archive)

    # (optional) take 3d archive snapshots and use to construct a gif
    archive_snapshot_filename = os.path.join(str(logdir), 'archive_snapshots.csv')
    if cfg.take_archive_snapshots:
        if os.path.exists(archive_snapshot_filename):
            os.remove(archive_snapshot_filename)
        num_cells = np.prod(scheduler.archive.dims)
        with open(archive_snapshot_filename, 'w') as archive_snapshot_file:
            row = ['Iteration'] + [f'cell_{i}' for i in range(num_cells)]
            writer = csv.writer(archive_snapshot_file)
            writer.writerow(row)

    result_archive = scheduler.result_archive
    best = 0.0

    obs_shape = cfg.obs_shape
    action_shape = cfg.action_shape

    ppo = scheduler.emitters[0].ppo

    # save the initial heatmap
    if cfg.num_dims == 2:
        save_heatmap(result_archive, os.path.join(str(heatmap_dir), f'heatmap_{0:05d}.png'))

    log_freq = 1
    log_arch_freq = cfg.log_arch_freq

    starting_iter = scheduler.emitters[0].itrs  # if loading a checkpoint, this will be > 0
    itrs = cfg.total_iterations
    # main loop
    for itr in range(starting_iter, itrs + 1):
        # Current solution point. returns a single sol per emitter
        solution_batch = scheduler.ask_dqd()
        mean_agent = Actor(obs_shape, action_shape, cfg.normalize_obs, cfg.normalize_returns).deserialize(
            solution_batch.flatten()).to(device)
        # whether to reset the stddev param for the action distribution
        if not cfg.adaptive_stddev:
            mean_agent.actor_logstd = torch.nn.Parameter(torch.zeros(1, np.prod(cfg.action_shape)))

        if cfg.normalize_obs:
            if scheduler.emitters[0].mean_agent_obs_normalizer is not None:
                mean_agent.obs_normalizer = scheduler.emitters[0].mean_agent_obs_normalizer

        if cfg.normalize_returns:
            if scheduler.emitters[0].mean_agent_return_normalizer is not None:
                mean_agent.return_normalizer = scheduler.emitters[0].mean_agent_return_normalizer

        ppo.agents = [mean_agent]
        # calculate gradients of f and m
        objs, measures, jacobian, metadata = ppo.train(vec_env=vec_env,
                                                       num_updates=cfg.calc_gradient_iters,
                                                       rollout_length=cfg.rollout_length,
                                                       calculate_dqd_gradients=True,
                                                       negative_measure_gradients=False)

        # for plotting purposes
        best = max(best, max(objs))

        # return the gradients to the scheduler. Will be used for the next step
        scheduler.tell_dqd(objs, measures, jacobian, metadata)

        # using grads from previous step, sample a batch of branched solution points and evaluate their f and m
        branched_sols = scheduler.ask()
        branched_agents = [
            Actor(obs_shape, action_shape, cfg.normalize_obs, cfg.normalize_returns).deserialize(sol).to(device) for sol
            in branched_sols]
        for agent in branched_agents:
            agent.actor_logstd.data = mean_agent.actor_logstd.data
        ppo.agents = branched_agents

        # since we branched from mean_agent, we will use its obs/return normalizer for the branched agents
        # if obs/return normalization is enabled
        eval_obs_normalizer = mean_agent.obs_normalizer if cfg.normalize_obs else None
        eval_rew_normalizer = mean_agent.return_normalizer if cfg.normalize_returns else None

        # evaluate the f and m of each branched agent
        objs, measures, metadata = ppo.evaluate(ppo.vec_inference,
                                                vec_env,
                                                verbose=True,
                                                obs_normalizer=eval_obs_normalizer,
                                                return_normalizer=eval_rew_normalizer)

        if cfg.weight_decay:
            reg_loss = cfg.weight_decay * np.array([np.linalg.norm(sol) for sol in branched_sols]).reshape(objs.shape)
            objs -= reg_loss

        best = max(best, max(objs))

        # return the evals to the scheduler. Will be used to update the search distribution in xnes
        restarted = scheduler.tell(objs, measures, metadata)
        if restarted:
            log.debug("Emitter restarted. Changing the mean agent...")
            mean_soln_point = scheduler.emitters[0].theta
            mean_agent = Actor(obs_shape, action_shape, cfg.normalize_obs, cfg.normalize_returns).deserialize(
                mean_soln_point).to(device)

            # load the obs/return normalizer used for this agent
            if cfg.normalize_obs:
                mean_agent.obs_normalizer = scheduler.emitters[0].mean_agent_obs_normalizer
            if cfg.normalize_returns:
                mean_agent.return_normalizer = scheduler.emitters[0].mean_agent_return_normalizer
            if not cfg.adaptive_stddev:
                mean_agent.actor_logstd = torch.nn.Parameter(torch.zeros(1, np.prod(cfg.action_shape)))

        mean_grad_coeffs = scheduler.emitters[0].opt.mu  # keep track of where the emitter is taking us
        mean_grad_coeffs = np.expand_dims(mean_grad_coeffs, axis=0).astype(np.float32)
        log.info(f'New mean coefficients: {mean_grad_coeffs}')

        # now we walk the solution point in the direction given by the new gradient coefficients
        ppo.grad_coeffs = mean_grad_coeffs
        ppo.agents = [mean_agent]
        log.info('Moving the mean solution point...')
        ppo.train(vec_env=vec_env,
                  num_updates=cfg.move_mean_iters,
                  rollout_length=cfg.rollout_length,
                  calculate_dqd_gradients=False,
                  move_mean_agent=True)

        # get the resulting new mean solution point and update the scheduler
        trained_mean_agent = ppo.agents[0]
        scheduler.emitters[0].update_theta(trained_mean_agent.serialize())

        # update the obs and return normalizers in the scheduler
        if cfg.normalize_obs:
            scheduler.emitters[0].mean_agent_obs_normalizer = trained_mean_agent.obs_normalizer
        if cfg.normalize_returns:
            scheduler.emitters[0].mean_agent_return_normalizer = trained_mean_agent.return_normalizer

        # logging
        log.debug(f'{itr=}, {itrs=}, Progress: {(100.0 * (itr / itrs)):.2f}%')

        if cfg.num_dims == 2:
            emitter_loc = (measures[0][0], measures[0][1])
            save_heatmap(result_archive, os.path.join(str(heatmap_dir), f'heatmap_{itr:05d}.png'),
                         emitter_loc=emitter_loc, forces=None)

        # Save the archive at the given frequency.
        # Always save on the final iteration.
        final_itr = itr == itrs
        if (itr > 0 and itr % log_arch_freq == 0) or final_itr:
            final_cp_dir = os.path.join(cp_dir, f'cp_{itr:08d}')
            if not os.path.exists(final_cp_dir):
                os.mkdir(final_cp_dir)
            # Save a full archive for analysis.
            df = result_archive.as_pandas(include_solutions=True, include_metadata=True)
            df.to_pickle(os.path.join(final_cp_dir, f"archive_df_{itr:08d}.pkl"))

            if cfg.save_scheduler:
                scheduler_savepath = os.path.join(final_cp_dir, f'scheduler_{itr:08d}.pkl')
                save_scheduler(scheduler, scheduler_savepath)

        # save the top 2 checkpoints, delete older ones
        while len(get_checkpoints(str(cp_dir))) > 2:
            oldest_checkpoint = get_checkpoints(str(cp_dir))[0]
            if os.path.exists(oldest_checkpoint):
                log.info(f'Removing checkpoint {oldest_checkpoint}')
                shutil.rmtree(oldest_checkpoint)

        # Update the summary statistics for the archive
        if (itr > 0 and itr % log_freq == 0) or final_itr:
            with open(summary_filename, 'a') as summary_file:
                writer = csv.writer(summary_file)
                data = [itr, result_archive.stats.qd_score, result_archive.stats.coverage,
                        result_archive.stats.obj_max, result_archive.stats.obj_mean]
                writer.writerow(data)

        if (itr > 0 and itr % log_freq == 0 and cfg.take_archive_snapshots) or (
                final_itr and cfg.take_archive_snapshots):
            with open(archive_snapshot_filename, 'a') as archive_snapshot_file:
                writer = csv.writer(archive_snapshot_file)
                num_cells = np.prod(scheduler.result_archive.dims)
                elite_scores = [0 for _ in range(num_cells)]
                for elite in scheduler.result_archive:
                    score, index = elite.objective, elite.index
                    elite_scores[index] = score
                data = [itr] + elite_scores
                writer.writerow(data)

        if cfg.use_wandb:
            with torch.no_grad():
                normA = torch.linalg.norm(scheduler.emitters[0].opt.A).cpu().numpy().item()
            wandb.log({
                "QD/QD Score": scheduler.result_archive.offset_qd_score,
                # use regular archive for qd score because it factors in the return offset
                "QD/average performance": result_archive.stats.obj_mean,
                "QD/coverage (%)": result_archive.stats.coverage * 100.0,
                "QD/best score": result_archive.stats.obj_max,
                "QD/iteration": itr,
                "QD/restarts": scheduler.emitters[0].restarts,
                'QD/mean_coeff_obj': mean_grad_coeffs[0][0],
                'XNES/norm_A': normA
            })
            for i in range(1, cfg.num_dims + 1):
                wandb.log({
                    'QD/iteration': itr,
                    f'QD/mean_coeff_measure{i}': mean_grad_coeffs[0][i]
                })


if __name__ == '__main__':
    cfg = parse_args()
    cfg.num_emitters = 1
    print(cfg)
    print(cfg.load_scheduler_from_cp)

    vec_env = make_vec_env_brax_ppga(task_name="humanoid", feat_name="jump", batch_size=cfg.env_batch_size, seed=cfg.seed, backend="spring", clip_obs_rew=True, episode_length=1000)
    cfg.batch_size = int(cfg.env_batch_size * cfg.rollout_length)
    cfg.num_envs = int(cfg.env_batch_size)

    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.obs_shape = vec_env.single_observation_space.shape
    cfg.action_shape = vec_env.single_action_space.shape

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
