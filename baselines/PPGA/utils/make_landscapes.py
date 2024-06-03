import os
import argparse
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from attrdict import AttrDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, help='Path to the archive snapshots')
    parser.add_argument('--fps', type=int)
    parser.add_argument('--outdir', type=str, default='./videos')
    args = parser.parse_args()
    cfg = AttrDict(vars(args))
    return cfg


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"}
    }


def animated_surface(cfg):
    filepath = cfg.datapath
    outdir = cfg.outdir
    df = pd.read_csv(filepath)
    z_data = df.drop('Iteration', axis=1).to_numpy().reshape(-1, 30, 30)  # TODO: make this general
    sh0, sh1 = z_data.shape[1], z_data.shape[2]
    x = np.linspace(0, 1, sh0)
    y = np.linspace(0, 1, sh1)

    fps = cfg.fps
    duration_ms = 1000 * (1 / fps)

    fig = go.Figure(
        data=[go.Surface(z=z_data[0])],
        layout=go.Layout(updatemenus=[dict(type='buttons', buttons=[dict(label="Play", method='animate', args=
        [None, {"frame": {"duration": duration_ms, 'redraw': True}}])])]),
        frames=[go.Frame(data=[go.Surface(z=k)], name=str(i)) for i, k in enumerate(z_data)]
    )

    # fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="tomato", project_z=True), colorscale='portland')
    fig.update_layout(title='Archive 3D Surface', autosize=False, width=1920, height=1080,
                      margin=dict(l=65, r=50, b=65, t=90))
    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    fig.update_layout(sliders=sliders)
    fig.show()


if __name__ == '__main__':
    cfg = parse_args()
    animated_surface(cfg)
