import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas
import wandb
import os
import torch
import glob
import json
import matplotlib.axis as maxis
from colorlog import ColoredFormatter
from matplotlib.pyplot import Axes
from collections import OrderedDict

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s][%(process)05d] %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'white,bold',
        'INFOV': 'cyan,bold',
        'WARNING': 'yellow',
        'ERROR': 'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

fh = logging.FileHandler('log.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

log = logging.getLogger('rl')
log.setLevel(logging.DEBUG)
log.handlers = []  # No duplicated handlers
log.propagate = False  # workaround for duplicated logs in ipython
log.addHandler(ch)
log.addHandler(fh)


def set_file_handler(logdir):
    global fh
    global log
    log.removeHandler(fh)
    filepath = os.path.join(logdir, 'log.txt')
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)


def config_wandb(**kwargs):
    # wandb initialization
    wandb.init(project=kwargs['wandb_project'], name=kwargs['run_name'])
    cfg = kwargs.get('cfg', None)
    if cfg is None:
        cfg = {}
        for key, val in kwargs.items():
            cfg[key] = val
    wandb.config.update(cfg)


def save_checkpoint(cp_dir, cp_name, model, optimizer, **kwargs):
    os.makedirs(cp_dir, exist_ok=True)
    params = {}
    params['model_state_dict'] = model.state_dict()
    params['optim_state_dict'] = optimizer.state_dict()
    for key, val in kwargs:
        params[key] = val
    torch.save(params, os.path.join(cp_dir, cp_name))


def save_cfg(dir, cfg):
    def to_dict(cfg):
        cfg = dict(cfg)
        return cfg
    filename = 'cfg.json'
    fp = os.path.join(dir, filename)
    with open(fp, 'w') as f:
        json.dump(cfg, f, default=to_dict, indent=4)


def get_checkpoints(checkpoints_dir):
    checkpoints = glob.glob(os.path.join(checkpoints_dir, 'cp_*'))
    return sorted(checkpoints)


class DataPostProcessor(Axes):
    name = 'data_post_processor'

    def __init__(self, fig, *args, **kwargs):
        super().__init__(fig, *args, **kwargs)

    def _init_axis(self):
        self.y1 = None
        self.y2 = None
        self.xaxis = maxis.XAxis(self)
        self.yaxis = maxis.YAxis(self)

    def fill_between(self, x, y1, y2=0, where=None, interpolate=False,
                     step=None, monotonic=True, **kwargs):
        if isinstance(y1, pandas.Series):
            y1 = y1.to_numpy()
        if isinstance(y2, pandas.Series):
            y2 = y2.to_numpy()
        if isinstance(x, pandas.Series):
            x = x.to_numpy()

        if monotonic:
            # gets rid of wandb artifacts where what should be monotonic metrics appear non-monotonic
            y1 = np.maximum.accumulate(y1)
            y2 = np.maximum.accumulate(y2)
            for i in range(len(self.lines)):
                # this performs unnecessary computations but idk a better way
                mean = self.lines[i].get_ydata()
                mean = np.maximum.accumulate(mean)
                self.lines[i].set_ydata(mean)
        Axes.fill_between(self, x, y1, y2, where=where, interpolate=interpolate,
                          step=step, **kwargs)



