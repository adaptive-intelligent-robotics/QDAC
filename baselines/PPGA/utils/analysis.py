import argparse
import json
import logging
import pickle
import wandb

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.projections as proj
import scienceplots
import seaborn as sns
import glob
import pandas as pd
import os
import numpy as np
import copy

from itertools import cycle
from pathlib import Path
from collections import OrderedDict
from baselines.PPGA.utils.archive_utils import pgame_checkpoint_to_objective_df, pgame_repertoire_to_pyribs_archive, \
    reevaluate_pgame_archive, reevaluate_ppga_archive, save_heatmap
from attrdict import AttrDict
from ribs.visualize import grid_archive_heatmap
from utilities import DataPostProcessor
from collections import OrderedDict
from typing import NamedTuple

plt.style.use('science')

api = wandb.Api()

shared_params = OrderedDict({
    'humanoid':
        {
            'objective_range': (0, 10000),
            'objective_resolution': 100,
            'archive_resolution': 2500,
            'skip_len': 200,
            'algorithm_name': 'cma_mae_100_0.01',
            'env_cfg': {
                'env_name': 'humanoid',
                'num_dims': 2,
                'episode_length': 1000,
                'grid_size': 50,
                'clip_obs_rew': True
            }
        },
    'walker2d':
        {
            'objective_range': (0, 5000),
            'objective_resolution': 100,
            'archive_resolution': 2500,
            'skip_len': 200,
            'algorithm_name': 'cma_mae_100_0.01',
            'env_cfg': {
                'env_name': 'walker2d',
                'num_dims': 2,
                'episode_length': 1000,
                'grid_size': 50,
                'clip_obs_rew': True
            }
        },
    'halfcheetah':
        {
            'objective_range': (0, 9000),
            'objective_resolution': 100,
            'archive_resolution': 2500,
            'skip_len': 200,
            'algorithm_name': 'cma_mae_100_0.01',
            'env_cfg': {
                'env_name': 'halfcheetah',
                'num_dims': 2,
                'episode_length': 1000,
                'grid_size': 50,
                'clip_obs_rew': False
            }
        },
    'ant':
        {
            'objective_range': (0, 6000),
            'objective_resolution': 100,
            'archive_resolution': 10000,
            'skip_len': 200,
            'algorithm_name': 'cma_mae_100_0.01',
            'env_cfg': {
                'env_name': 'ant',
                'num_dims': 4,
                'episode_length': 1000,
                'grid_size': 10,
                'clip_obs_rew': False
            }
        }
})

PGAME_DIRS = AttrDict({
    'humanoid': f'{Path.home()}/QDax/experiments/pga_me_humanoid_uni_baseline/',
    'walker2d': f'{Path.home()}/QDax/experiments/pga_me_walker2d_uni_baseline/',
    'halfcheetah': f'{Path.home()}/QDax/experiments/pga_me_halfcheetah_uni_baseline/',
    'ant': f'{Path.home()}/QDax/experiments/pga_me_ant_uni_baseline/'
})

PPGA_DIRS = AttrDict({
    'humanoid': 'experiments/paper_ppga_humanoid_v2_clipped_nonadaptive',
    'walker2d': 'experiments/paper_ppga_walker2d_v2_clipped',
    'halfcheetah': 'experiments/paper_ppga_halfcheetah_v2',
    'ant': 'experiments/paper_ppga_ant_v2'
})

QDPG_DIRS = AttrDict({
    'humanoid': 'experiments/qdpg_humanoid',
    'walker2d': 'experiments/qdpg_walker2d',
    'halfcheetah': 'experiments/qdpg_halfcheetah',
    'ant': 'experiments/qdpg_ant'
})

SEP_CMA_MAE_DIRS = AttrDict({
    'humanoid': 'experiments/sep_cma_mae_humanoid_baseline',
    'walker2d': 'experiments/sep_cma_mae_walker2d_baseline',
    'halfcheetah': 'experiments/sep_cma_mae_halfcheetah_baseline',
    'ant': 'experiments/sep_cma_mae_ant_baseline'
})

CMA_MAEGA_TD3_ES_DIRS = AttrDict({
    'humanoid': 'experiments/cma_maega_td3_es_humanoid_baseline',
    'walker2d': 'experiments/cma_maega_td3_es_walker2d_baseline',
    'halfcheetah': 'experiments/cma_maega_td3_es_halfcheetah_baseline',
    'ant': 'experiments/cma_maega_td3_es_ant_baseline'
})

# these are set to match the hue order of seaborn lineplot
HUES = OrderedDict({'PPGA': (0.1215, 0.4667, 0.7058),
                    'PGA-ME': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
                    'QDPG': (1.0, 0.4980392156862745, 0.054901960784313725),
                    'SEP-CMA-MAE': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
                    'CMA-MAEGA(TD3, ES)': (0.5803921568627451, 0.403921568627451, 0.7411764705882353)})

list1 = ['PPGA', 'SEP-CMA-MAE', 'CMA-MAEGA(TD3, ES)', 'TD3GA']
list2 = ['QDPG', 'PGA-ME']

algorithms = OrderedDict({
    'PPGA': {'keywords': ['paper', 'v2'], 'evals_per_iter': 300},
    # 'TD3GA': {'keywords': ['td3ga'], 'evals_per_iter': 300},
    'PGA-ME': {'keywords': ['pga_me'], 'evals_per_iter': 300},
    'QDPG': {'keywords': ['qdpg'], 'evals_per_iter': 300},
    'SEP-CMA-MAE': {'keywords': ['sep'], 'evals_per_iter': 200},
    'CMA-MAEGA(TD3, ES)': {'keywords': ['td3_es'], 'evals_per_iter': 100},
})

matplotlib.rcParams.update(
    {
        "font.size": 16,
    }
)


def index_of(env_name):
    return list(shared_params.keys()).index(env_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str)

    args = parser.parse_args()
    return vars(args)


def compile_cdf(cfg, dataframes=None):
    num_cells = cfg.archive_resolution

    if not dataframes:
        df_dir = cfg.archive_dir
        filenames = next(os.walk(df_dir), (None, None, []))[2]  # [] if no file
        dataframes = []
        for f in filenames:
            full_path = os.path.join(df_dir, f)
            df = pd.read_pickle(full_path)
            dataframes.append(df)

    x = np.linspace(cfg.objective_range[0], cfg.objective_range[1], cfg.objective_resolution)
    all_y_vals = []
    for df in dataframes:
        y_vals = []
        df_cells = np.array(sorted(df['objective']))
        for x_val in x:
            count = len(df_cells[df_cells > x_val])
            percentage = (count / num_cells) * 100.0
            y_vals.append(percentage)
        all_y_vals.append(np.array(y_vals))

    all_y_vals = np.vstack(all_y_vals)
    mean, stddev = np.mean(all_y_vals, axis=0), np.std(all_y_vals, axis=0)

    all_data = np.vstack((x, mean, mean - stddev, mean + stddev))
    cdf = pd.DataFrame(all_data.T, columns=['Objective',
                                            'Threshold Percentage (Mean)',
                                            'Threshold Percentage (Min)',
                                            'Threshold Percentage (Max)'])

    return cdf


def get_results_dataframe(env_name: str, algorithm: str, keywords: list[str], name = None, scaling_exp=False):
    runs = api.runs('qdrl/PPGA', filters={
        "$and": [{'tags': algorithm}, {'tags': env_name}]
    })

    keys = []
    if algorithm in list1:
        if scaling_exp:
            keys = ['QD/iteration', 'QD/coverage (%)', 'QD/QD Score', 'QD/Best Score', 'QD/average performance']
        else:
            keys = ['QD/iteration', 'QD/coverage (%)', 'QD/QD Score', 'QD/best score']
    elif algorithm in list2:
        if scaling_exp:
            keys = ['coverage', 'qd_score', 'max_fitness', 'avg_fitness']
        else:
            keys = ['coverage', 'qd_score', 'max_fitness']

    hist_list = []
    cache_dir = Path('./.cache')
    cache_dir.mkdir(exist_ok=True)
    for run in runs:
        res = all([key in run.name for key in keywords]) and '24hr' not in run.name
        if res:
            if algorithm in list1:
                cached_data_path = cache_dir.joinpath(Path(f'{run.storage_id}.csv'))
                if cached_data_path.exists():
                    print(f'Loading cached data for run {run.name}')
                    hist = pd.read_csv(str(cached_data_path))
                else:
                    # this takes a long time
                    hist = pd.DataFrame(
                        run.scan_history(keys=keys))
                    # use this for debugging/tweaking the figure
                    # hist = run.history(keys=keys)
                    hist.to_csv(str(cached_data_path))
            else:
                # for some reason, baselines run from the qdax library can't be one-shot concatenated
                # so we load dataframes key-by-key and concat together at the end
                hists = []
                for key in keys:
                    cached_data_path = cache_dir.joinpath(Path(f'{run.storage_id}_{key}.csv'))
                    if cached_data_path.exists():
                        print(f'Loading cached data for run {run.name} and {key=} from cache')
                        df = pd.read_csv(str(cached_data_path))
                    else:
                        df = pd.DataFrame(run.scan_history(keys=['iteration'] + [key]))
                        df.to_csv(str(cached_data_path))
                    hists.append(df)
                    # hists.append(run.history(keys=[key]))
                hist = pd.concat(hists, axis=1, ignore_index=False)
                # remove duplicate cols
                hist = hist.loc[:, ~hist.columns.duplicated()].copy()
                if scaling_exp:
                    hist = pd.DataFrame(data=hist, columns=keys).rename(columns={
                        'iteration': 'QD/iteration',
                        'coverage': 'QD/coverage (%)',
                        'qd_score': 'QD/QD Score',
                        'max_fitness': 'QD/best score',
                        'avg_fitness': 'QD/average performance'
                    })
                else:
                    hist = pd.DataFrame(data=hist, columns=keys + ['iteration'])\
                        .rename(columns={'iteration': 'QD/iteration',
                                         'coverage': 'QD/coverage (%)',
                                         'qd_score': 'QD/QD Score',
                                         'max_fitness': 'QD/best score'})
            # hist = pd.DataFrame(data=hist, columns=['QD/iteration', 'QD/coverage (%)', 'QD/QD Score', 'QD/best sore'])
            hist['name'] = name if name is not None else algorithm
            hist_list.append(hist)

    df = pd.concat(hist_list, ignore_index=True)
    return df


def make_cdf_plot(cfg, data: pd.DataFrame, ax: plt.axis, standalone: bool = False, **kwargs):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    y_label = "Archive CDF"

    # Color mapping for algorithms
    palette = {
        "CMA-MAE": "C0",
        "CMA-ME": "C1",
        "MAP-Elites (line)": "C2",
        "MAP-Elites": "C3",
    }

    x = data['Objective'].to_numpy().flatten()
    y_avg = data.filter(regex='Mean').to_numpy().flatten()
    y_min = data.filter(regex='Min').to_numpy().flatten()
    y_max = data.filter(regex='Max').to_numpy().flatten()
    ax.plot(x, y_avg, linewidth=1.0, label=cfg.algorithm, **kwargs)
    ax.fill_between(x, y_min, y_max, alpha=0.2, **kwargs)
    ax.set_xlim(cfg.objective_range)
    ax.set_yticks(np.arange(0, 101, 25.0))
    ax.set_xlabel("Objective", fontsize=16)
    if standalone:
        ax.set_ylabel(y_label)
        ax.set_title(cfg.title)
        ax.legend()


def get_pgame_df(exp_dir, reevaluated_archive=False, save=False):
    out_dir = os.path.join(exp_dir, 'cdf_analysis')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    seeds = [1111, 2222, 3333, 4444]
    dataframes = []
    for seed in seeds:
        subdir = sorted(glob.glob(exp_dir + '/' + f'*{seed}*/checkpoints/checkpoint_*'))[0]
        if reevaluated_archive:
            filepath = glob.glob(subdir + '/' + '*reeval_archive*')[0]
            with open(filepath, 'rb') as f:
                df = pickle.load(f).as_pandas()
        else:
            df = pgame_checkpoint_to_objective_df(subdir)
        dataframes.append(df)
        if save:
            df.to_pickle(os.path.join(out_dir, f'scores_{seed}.pkl'))
    return dataframes


def get_ppga_df(exp_dir, reevaluated_archive=False):
    seeds = [1111, 2222, 3333, 4444]
    dataframes = []
    for seed in seeds:
        subdir = sorted(glob.glob(exp_dir + '/' + f'*{seed}*/checkpoints/cp_*'))[-1]  # gets the most recent checkpoint
        if reevaluated_archive:
            filename = glob.glob(subdir + '/' + '*reeval_archive*')[0]
            with open(filename, 'rb') as f:
                df = pickle.load(f).as_pandas()
        else:
            filename = glob.glob(subdir + '/' + 'archive_*')[0]
            df = pd.read_pickle(filename)
        dataframes.append(df)

    return dataframes


def plot_cdf_data(algorithm: str, alg_data_dirs: dict, archive_type: str, reevaluated_archives=False, axs=None, standalone_plot=False, **kwargs):
    '''
    :param algorithm: name of the algorithm
    :param alg_data_dirs: contains env: path string-string pairs for all envs for this algorithm
    :param archive_type: either pyribs or qdax depending on which repo produced the archive
    :param reevaluated_archives: whether to plot corrected QD metrics or not
    :param axs: matplotlib axes objects
    '''
    if axs is None:
        standalone_plot = True
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    subtitle = 'Archive CDFs'
    prefix = 'Corrected ' if reevaluated_archives else ''
    title = prefix + subtitle

    for i, (exp_name, env_dir) in enumerate(alg_data_dirs.items()):
        base_cfg = AttrDict(shared_params[exp_name])
        base_cfg['title'] = exp_name

        # TODO: temporary hack
        if algorithm == 'QDPG' and exp_name in ['walker2d', 'halfcheetah']:
            base_cfg['env_cfg']['grid_size'] = 10
            base_cfg['archive_resolution'] = 100

        cfg = copy.copy(base_cfg)
        cfg.update({'archive_dir': env_dir, 'algorithm': algorithm})
        dataframe_fn = get_ppga_df if archive_type == 'pyribs' else get_pgame_df
        algo_dataframes = dataframe_fn(env_dir, reevaluated_archives)
        algo_cdf = compile_cdf(cfg, dataframes=algo_dataframes)

        if standalone_plot:
            (j, k) = np.unravel_index(i, (2, 2))  # b/c there are 4 envs
            make_cdf_plot(cfg, algo_cdf, axs[j][k], standalone=True)
        else:
            env_idx = index_of(exp_name)
            make_cdf_plot(cfg, algo_cdf, axs[3][env_idx], color=HUES[algorithm])


def load_and_eval_pgame_archive(exp_name, exp_dirs, seed, data_is_saved=False):
    exp_dir = exp_dirs[exp_name]
    cp_path = sorted(glob.glob(exp_dir + '/' + f'*{seed}*/checkpoints/checkpoint_*'))[0]
    save_path = cp_path
    print(f'{save_path=}')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    base_cfg = AttrDict(shared_params[exp_name])
    env_cfg = base_cfg.env_cfg
    env_cfg.seed = seed
    if data_is_saved:
        orig_archive_fp = glob.glob(save_path + '/' + '*original_archive*')[0]
        with open(orig_archive_fp, 'rb') as f:
            original_archive = pickle.load(f)

        new_archive_fp = os.path.join(save_path, f'{exp_name}_reeval_archive.pkl')
        with open(new_archive_fp, 'rb') as f:
            new_archive = pickle.load(f)
        print(f'{exp_name} Re-evaluated PGAME Archive \n'
              f'Coverage: {new_archive.stats.coverage} \n'
              f'Max fitness: {new_archive.stats.obj_max} \n'
              f'Avg Fitness: {new_archive.stats.obj_mean} \n'
              f'QD Score: {new_archive.offset_qd_score}')
    else:
        original_archive, pgame_sols = pgame_repertoire_to_pyribs_archive(cp_path + '/', env_cfg, save_path=save_path)
        new_archive = reevaluate_pgame_archive(env_cfg, archive_df=original_archive.as_pandas(), save_path=save_path)
    return original_archive, new_archive


def load_and_eval_ppga_archive(exp_name, exp_dirs, seed, data_is_saved=False):
    exp_dir = exp_dirs[exp_name]
    cp_path = sorted(glob.glob(exp_dir + '/' + f'*{seed}*/checkpoints/cp_*'))[-1]  # gets the most recent checkpoint
    save_path = cp_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    base_cfg = AttrDict(shared_params[exp_name])
    env_cfg = base_cfg.env_cfg
    env_cfg.seed = seed

    agent_cfg_fp = exp_dir + f'/{seed}/' + 'cfg.json'
    with open(agent_cfg_fp, 'r') as f:
        agent_cfg = json.load(f)
        agent_cfg = AttrDict(agent_cfg)

    scheduler_fp = glob.glob(cp_path + '/' + 'scheduler_*')[0]
    with open(scheduler_fp, 'rb') as f:
        scheduler = pickle.load(f)
        original_archive = scheduler.archive

    if data_is_saved:
        new_archive_fp = os.path.join(save_path, f'{exp_name}_reeval_archive.pkl')
        with open(new_archive_fp, 'rb') as f:
            new_archive = pickle.load(f)
        print(f'{exp_name} Re-evaluated PPGA Archive \n'
              f'Coverage: {new_archive.stats.coverage} \n'
              f'Max fitness: {new_archive.stats.obj_max} \n'
              f'Avg Fitness: {new_archive.stats.obj_mean} \n'
              f'QD Score: {new_archive.offset_qd_score}')
    else:
        new_archive = reevaluate_ppga_archive(env_cfg, agent_cfg=agent_cfg, original_archive=original_archive,
                                              save_path=save_path)

    return original_archive, new_archive


def visualize_reevaluated_archives():
    seed = 1111
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))

    for i, exp_name in enumerate(PGAME_DIRS.keys()):
        _, new_pgame_archive = load_and_eval_pgame_archive(exp_name, seed, data_is_saved=True)
        _, new_ppga_archive = load_and_eval_ppga_archive(exp_name, seed, data_is_saved=True)

        grid_archive_heatmap(new_pgame_archive, ax=axs[0][i])
        grid_archive_heatmap(new_ppga_archive, ax=axs[1][i])
        axs[0][i].set_title(exp_name)

    axs[0][0].set_ylabel('PGA-ME')
    axs[1][0].set_ylabel('PPGA')
    fig.tight_layout()
    plt.show()


def print_corrected_qd_metrics(algorithm: str, exp_dirs, algorithm_type: str):
    assert algorithm_type in ['pyribs', 'qdax']
    seeds = [1111, 2222, 3333, 4444]
    alg_data = {'coverage': [],
                 'obj_max': [],
                 'obj_mean': [],
                 'qd_score': [],
                 'num_elites': [],
                 'offset_qd_score': []}
    final_alg_data = {}

    for exp_name in exp_dirs.keys():
        # clear any old data and start fresh
        for key in alg_data.keys():
            alg_data[key] = []
        for seed in seeds:
            eval_fn = load_and_eval_ppga_archive if algorithm_type == 'pyribs' else load_and_eval_pgame_archive
            _, new_archive = eval_fn(exp_name, exp_dirs, seed, data_is_saved=False)

            stats = new_archive.stats
            stats_dict = {'num_elites': stats.num_elites,
                          'coverage': stats.coverage,
                          'qd_score': stats.qd_score,
                          'obj_max': stats.obj_max,
                          'obj_mean': stats.obj_mean}
            for name, val in stats_dict.items():
                alg_data[name].append(val)
            # this is not in stats, so we need to add it manually
            alg_data['offset_qd_score'].append(new_archive.offset_qd_score)

        # now that we've collected data from all seeds, we can average and put it into the final dict
        final_alg_data[exp_name] = {}

        for name, data in alg_data.items():
            final_alg_data[exp_name][name] = np.mean(np.array(data))

    # once we've done this for all experiments, we can print the final result
    for exp_name, data in final_alg_data.items():
        print(f'{algorithm} {exp_name}: Averaged Results: {data}')


def get_hyperparam_gridsearch_results():
    '''Plot the results from the N1, N2 gridsearch experiments'''
    sns.set(rc={'figure.figsize': (8, 6)})
    sns.set_style(style='white')
    runs = api.runs('qdrl/PPGA', filters={
        "$and": [{"tags": "PPGA"}, {"tags": "humanoid"}]
    })
    hist_list = []
    for run in runs:
        if 'gradsteps' in run.name or 'walksteps' in run.name or 'v2_clipped_nonadaptive' in run.name:
            # this takes a long time
            hist = pd.DataFrame(
                run.scan_history(keys=['QD/iteration', 'QD/coverage (%)', 'QD/QD Score', 'QD/best score']))
            # use this for debugging/tweaking the figure
            # hist = run.history(keys=['QD/iteration', 'QD/coverage (%)', 'QD/QD Score', 'QD/best score'])
            hist['name'] = f'(N1, N2) = ({run.config["calc_gradient_iters"]}, {run.config["move_mean_iters"]})'
            hist_list.append(hist)

    df = pd.concat(hist_list, ignore_index=True)

    evals_per_iter = 300  # b/c PPGA branches 300 solutions per QD iteration
    evals = df['QD/iteration'] * evals_per_iter
    df['Num Evals'] = evals

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    sns.lineplot(x='Num Evals', y='QD/QD Score', errorbar='sd', data=df, ax=axs[0], hue='name')
    sns.lineplot(x='Num Evals', y='QD/best score', errorbar='sd', data=df, ax=axs[1], hue='name')
    sns.lineplot(x='Num Evals', y="QD/coverage (%)", errorbar='sd', data=df, ax=axs[2], hue='name')
    axs[0].set_ylabel('QD Score')
    axs[1].set_ylabel('Best Reward')
    axs[2].set_ylabel('Coverage (\%)')
    for ax in axs:
        ax.get_legend().remove()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0)
    plt.tight_layout()
    plt.show()


def plot_qd_results_main():
    alg_names = list(algorithms.keys())
    envs = ['humanoid', 'walker2d', 'halfcheetah', 'ant']
    proj.register_projection(DataPostProcessor)
    fig, axs = plt.subplots(4, 4, figsize=(16, 12), subplot_kw=dict(projection='data_post_processor'))

    for j, env in enumerate(envs):
        all_data = []
        for algorithm in algorithms.keys():
            df = get_results_dataframe(env, algorithm, keywords=algorithms[algorithm]['keywords'])

            evals = df['QD/iteration'] * algorithms[algorithm]['evals_per_iter']
            df['Num Evals'] = evals
            df['env'] = env
            df = df.sort_values(by=['Num Evals'])
            if algorithm == 'QDPG' and env in ['walker2d', 'halfcheetah']:
                #  TODO: temporary hack
                df['QD/QD Score'] *= 25.0

            # trim PPGA to 500k and PGA-ME to 1mil
            # if algorithm == 'PPGA':
            #     df = df.loc[: df[(df['QD/iteration'] == 1667)].index[0] - 1, :]
            #     df = df[:-1]
            # if algorithm == 'PGA-ME':
            #     df = df.loc[: df[(df['QD/iteration'] == 3330)].index[0], :]
            #     df = df[:-1]

            all_data.append(df)

        all_data = pd.concat(all_data, ignore_index=True).sort_values(by=['Num Evals'])

        ax_best = sns.lineplot(x='Num Evals', y='QD/best score', errorbar='sd', data=all_data,
                               ax=axs[0][envs.index(env)], hue='name', hue_order=alg_names, legend=False)
        ax_qd = sns.lineplot(x='Num Evals', y='QD/QD Score', errorbar='sd', data=all_data, ax=axs[1][envs.index(env)],
                             hue='name', hue_order=alg_names, legend=False)
        ax_cov = sns.lineplot(x='Num Evals', y="QD/coverage (%)", errorbar='sd', data=all_data,
                              hue='name', ax=axs[2][envs.index(env)], hue_order=alg_names, legend=False)
        ax_cov.set_xlabel('Num Evals', fontsize=16)

        axs[0][j].set_ylabel('Best Reward', fontsize=16)
        axs[1][j].set_ylabel('QD Score', fontsize=16)
        axs[2][j].set_ylabel('Coverage (\%)', fontsize=16)

    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            if i < 3:
                ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
            if i <= 1:
                ax.set(xlabel=None)
            if j >= 1:
                ax.set(ylabel=None)

    plot_cdf_data('PPGA', PPGA_DIRS, archive_type='pyribs', reevaluated_archives=False, axs=axs)
    plot_cdf_data('PGA-ME', PGAME_DIRS, archive_type='qdax', reevaluated_archives=False, axs=axs)
    plot_cdf_data('QDPG', QDPG_DIRS, archive_type='qdax', reevaluated_archives=False, axs=axs)
    plot_cdf_data('SEP-CMA-MAE', SEP_CMA_MAE_DIRS, archive_type='pyribs', reevaluated_archives=False, axs=axs)
    plot_cdf_data('CMA-MAEGA(TD3, ES)', CMA_MAEGA_TD3_ES_DIRS, archive_type='pyribs', reevaluated_archives=False, axs=axs)

    # add titles
    for i, ax in enumerate(axs[0][:]):
        ax.set_title(envs[i], fontsize=16)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    h, l = axs.flatten()[-1].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, borderaxespad=0, fancybox=True, fontsize=16)
    plt.show()


def n1_n2_plots():
    proj.register_projection(DataPostProcessor)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), subplot_kw=dict(projection='data_post_processor'))

    all_data = []
    all_data.append(get_results_dataframe('humanoid', 'PPGA', keywords=['paper', 'v2_clipped_nonadaptive'], name='(N1, N2) = (10, 10)'))  # baseline
    all_data.append(get_results_dataframe('humanoid', 'PPGA', keywords=['gradsteps_10_walksteps_5'], name='(N1, N2) = (10, 5)'))
    all_data.append(get_results_dataframe('humanoid', 'PPGA', keywords=['gradsteps_5_walksteps_10'], name='(N1, N2) = (5, 10)'))
    all_data.append(get_results_dataframe('humanoid', 'PPGA', keywords=['gradsteps_1_walksteps_1'], name='(N1, N2) = (1, 1)'))
    all_data.append(get_results_dataframe('humanoid', 'PPGA', keywords=['gradsteps_5_walksteps_5'], name='(N1, N2) = (5, 5)'))
    for df in all_data:
        evals = df['QD/iteration'] * algorithms['PPGA']['evals_per_iter']
        df['Num Evals'] = evals
        df.sort_values(by=['QD/iteration'], inplace=True)

    all_data = pd.concat(all_data, ignore_index=True).sort_values(by=['QD/iteration'])

    sns.lineplot(x='Num Evals', y='QD/QD Score', errorbar='sd', data=all_data, ax=axs[0], hue='name')
    sns.lineplot(x='Num Evals', y='QD/best score', errorbar='sd', data=all_data, ax=axs[1], hue='name')
    sns.lineplot(x='Num Evals', y="QD/coverage (%)", errorbar='sd', data=all_data, ax=axs[2], hue='name')

    axs[0].set_ylabel('QD Score')
    axs[1].set_ylabel('Best Reward')
    axs[2].set_ylabel('Coverage (\%)')
    for ax in axs:
        ax.get_legend().remove()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0)
    plt.tight_layout()
    plt.show()


def td3_ablation_plots():
    proj.register_projection(DataPostProcessor)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), subplot_kw=dict(projection='data_post_processor'))

    all_data = []
    ppga_df = get_results_dataframe('humanoid', 'PPGA', keywords=['paper', 'v2_clipped_nonadaptive'])  # baseline
    evals = ppga_df['QD/iteration'] * algorithms['PPGA']['evals_per_iter']
    ppga_df['Num Evals'] = evals
    ppga_df = ppga_df.sort_values(by=['QD/iteration'])
    ppga_df = ppga_df[:-10]
    all_data.append(ppga_df)

    td3ga_df = get_results_dataframe('humanoid', 'TD3GA', keywords=['td3ga'])
    evals = ppga_df['QD/iteration'] * algorithms['TD3GA']['evals_per_iter']
    td3ga_df = td3ga_df.sort_values(by=['QD/iteration'])
    td3ga_df['Num Evals'] = evals
    td3ga_df = td3ga_df[:-10]
    all_data.append(td3ga_df)

    all_data = pd.concat(all_data, ignore_index=True).sort_values(by=['QD/iteration'])

    sns.lineplot(x='Num Evals', y='QD/QD Score', errorbar='se', data=all_data, ax=axs[0], hue='name')
    sns.lineplot(x='Num Evals', y='QD/best score', errorbar='se', data=all_data, ax=axs[1], hue='name')
    sns.lineplot(x='Num Evals', y="QD/coverage (%)", errorbar='se', data=all_data, ax=axs[2], hue='name')

    axs[0].set_ylabel('QD Score')
    axs[1].set_ylabel('Best Reward')
    axs[2].set_ylabel('Coverage (\%)')
    for ax in axs:
        ax.get_legend().remove()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0)
    plt.tight_layout()
    plt.show()


def plot_scaling_experiment():
    proj.register_projection(DataPostProcessor)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), subplot_kw=dict(projection='data_post_processor'))

    ppga_df = get_results_dataframe('humanoid', 'PPGA', keywords=['4k'], scaling_exp=True)
    evals = ppga_df['QD/iteration'] * algorithms['PPGA']['evals_per_iter']
    ppga_df['Num Evals'] = evals

    pgame_df = get_results_dataframe('humanoid', 'PGA-ME', keywords=['24hr'], scaling_exp=True)
    pgame_df = pgame_df.sort_values(by=['Num Evals'])
    evals = pgame_df['QD/iteration'] * algorithms['PGA-ME']['evals_per_iter']
    pgame_df['Num Evals'] = evals

    all_data = pd.concat([ppga_df, pgame_df], ignore_index=True).sort_values(by='Relative Time (Wall)')

    sns.lineplot(x='Num Evals', y='QD/QD Score', errorbar='sd', data=all_data, ax=axs[0], hue='name')
    sns.lineplot(x='Num Evals', y='QD/best score', errorbar='sd', data=all_data, ax=axs[1], hue='name')
    sns.lineplot(x='Num Evals', y="QD/coverage (%)", errorbar='sd', data=all_data, ax=axs[2], hue='name')
    sns.lineplot(x='Num Evals', y="QD/average performance (%)", errorbar='sd', data=all_data, ax=axs[3], hue='name')

    axs[0].set_ylabel('QD Score')
    axs[1].set_ylabel('Best Reward')
    axs[2].set_ylabel('Coverage (\%)')
    axs[3].set_ylabel('Avg. Performance')
    for ax in axs:
        ax.get_legend().remove()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0)
    plt.tight_layout()
    plt.show()


def plot_corrected_cdfs():
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    plot_cdf_data('PPGA', PPGA_DIRS, archive_type='pyribs', reevaluated_archives=True, standalone_plot=True, axs=axs)
    plot_cdf_data('PGA-ME', PGAME_DIRS, archive_type='qdax', reevaluated_archives=True, standalone_plot=True, axs=axs)
    plot_cdf_data('QDPG', QDPG_DIRS, archive_type='qdax', reevaluated_archives=True, standalone_plot=True, axs=axs)
    plot_cdf_data('SEP-CMA-MAE', SEP_CMA_MAE_DIRS, archive_type='pyribs', reevaluated_archives=True, standalone_plot=True, axs=axs)
    plot_cdf_data('CMA-MAEGA(TD3, ES)', CMA_MAEGA_TD3_ES_DIRS, archive_type='pyribs', reevaluated_archives=True, axs=axs)

    fig.tight_layout()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    # print_corrected_qd_metrics('PPGA', PPGA_DIRS, 'pyribs')
    # plot_scaling_experiment()
    plot_corrected_cdfs()
    # plot_qd_results_main()

