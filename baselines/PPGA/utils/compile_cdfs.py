
# Computes CDFs for each archive

import re
import csv
import glob
import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
from attrdict import AttrDict

name_mapping_1 = {
    'cma_me_imp_100': 'CMA-ME',
    'cma_mae_100_0.01': 'CMA-MAE',
    'map_elites_100': 'MAP-Elites',
    'map_elites_line_100': 'MAP-Elites (line)',
}

name_mapping_2 = {
    'cma_me_imp': 'CMA-ME',
    'cma_mae': 'CMA-MAE',
    'map_elites': 'MAP-Elites',
    'map_elites_line': 'MAP-Elites (line)',
}

name_mapping = name_mapping_1

algo_order = [
    'CMA-MAE',
    'CMA-ME',
    'MAP-Elites (line)',
    'MAP-Elites',
]


def order_func(datum):
    return algo_order.index(datum[0])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objective_range', action='append', type=int)
    parser.add_argument('--objective_resolution', type=int)
    parser.add_argument('--archive_resolution', action='append', type=int)
    parser.add_argument('--experiment_path', type=str)
    parser.add_argument('--archive_path', type=str)
    parser.add_argument('--skip_len', type=int, default=200)  # not sure what this does
    parser.add_argument('--archive_summary_filename', type=str, default='cdf.csv')
    parser.add_argument('--algorithm_name', type=str, default='cma_mae_100_0.01')

    args = parser.parse_args()
    cfg = AttrDict(vars(args))
    return cfg


def make_cdf_plot(cfg, plot_dir):
    csv_filepath = os.path.join(plot_dir, 'cdf.csv')
    image_filepath = os.path.join(plot_dir, 'cdf.pdf')

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    data = pd.read_csv(csv_filepath)

    y_label = "Threshold Percentage"

    plt.figure(figsize=(16, 12))

    # Color mapping for algorithms
    palette = {
        "CMA-MAE": "C0",
        "CMA-ME": "C1",
        "MAP-Elites (line)": "C2",
        "MAP-Elites": "C3",
    }

    sns.set(font_scale=4)
    with sns.axes_style("white"):
        sns.set_style("white", {'font.family': 'serif', 'font.serif': 'Palatino'})
        sns.set_palette("colorblind")

        # Plot the responses for different events and regions
        sns_plot = sns.lineplot(x="Objective",
                                y=y_label,
                                linewidth=3.0,
                                hue="Algorithm",
                                data=data,
                                legend=False,
                                palette=palette,
                                )
        sns_plot.set(xlim=tuple(cfg.objective_range))
        # plt.xticks([96, 98, 100])
        # plt.xticks([0, 50, 100])

        plt.yticks(np.arange(0, 101, 10.0))
        plt.xlabel("Objective")
        plt.ylabel(y_label)

        legend = plt.legend(loc='upper left', frameon=False, prop={'size': 40})
        # legend.set_bbox_to_anchor((0.35, 0.45))
        for line in legend.get_lines():
            line.set_linewidth(4.0)

        frame = legend.get_frame()
        frame.set_facecolor('white')
        plt.tight_layout()
        # plt.show()
        sns_plot.figure.savefig(image_filepath)


def compile_cdf(cfg):
    num_cells = cfg.archive_resolution

    # Compile all the data
    all_data = []
    for archive_filename in glob.glob(cfg.archive_path):
        head, filename = path.split(archive_filename)
        head, trial_name = path.split(head)
        head, algo_name = path.split(head)
        algo_name = cfg.algorithm_name
        print(algo_name)
        if algo_name not in name_mapping:
            continue
        algo_name = name_mapping[algo_name]
        _, trial_id = re.split('cp_|checkpoint_', trial_name)
        print(algo_name, trial_id)

        df = pd.read_pickle(archive_filename)
        df_cells = sorted(df['objective'])

        n = len(df_cells)
        ptr = 0

        lo, hi = tuple(cfg.objective_range)
        values = []
        for i in range(cfg.objective_resolution):

            thresh = (hi - lo) * (i / (cfg.objective_resolution - 1)) + lo
            # thresh = round(thresh+1e-9, 2)
            thresh = int(thresh + 1e-9)

            while ptr < n and df_cells[ptr] < thresh:
                ptr += 1

            values.append((thresh, n - ptr))

        for thresh, cnt in values:
            cnt = (cnt / num_cells) * 100.0
            datum = [algo_name, trial_id, thresh, cnt]
            all_data.append(datum)

    # Sort the data by the names in the given order.
    all_data.sort(key=order_func)
    all_data.insert(0,
                    ['Algorithm', 'Trial', 'Objective', 'Threshold Percentage']
                    )
    return all_data


if __name__ == '__main__':
    cfg = parse_args()
    cdf_data = compile_cdf(cfg)

    # Output the summary of summary files.
    plot_dir = os.path.join(cfg.experiment_path, 'plots')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    archive_summary_fp = os.path.join(plot_dir, cfg.archive_summary_filename)
    with open(archive_summary_fp, 'w') as summary_file:
        writer = csv.writer(summary_file)
        for datum in cdf_data:
            writer.writerow(datum)

    make_cdf_plot(cfg, plot_dir)
