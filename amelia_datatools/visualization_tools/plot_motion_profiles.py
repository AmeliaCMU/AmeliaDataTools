

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from amelia_datatools.utils import common as C
from amelia_datatools.utils import utils
from amelia_datatools.utils.processor_utils import polar_histogram
from amelia_datatools.trajectory_data_tools.compute_motion_profiles import run_processor


def format_func(value):
    return "%.1f" % value


def generate_bin_edges(num_bins, min_value, max_value):
    bin_width = (max_value - min_value) / num_bins
    bin_edges = [min_value + i * bin_width for i in range(num_bins + 1)]
    return bin_edges


def plot_vertical_hist(base_dir: str, traj_version: str, to_process: bool, input_path: str, motion_profile: str, drop_interp: bool, agent_type: bool, dpi: int):
    out_dir = os.path.join(VIS_DIR, utils.get_file_name(__file__))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Created output directory in: {out_dir}")

    suffix = '_dropped_int' if drop_interp else ''
    suffix += f'_{agent_type}'
    base_file = f"{motion_profile}_profiles{suffix}"
    input_file = os.path.join(input_path, f"{base_file}.pkl")

    if not os.path.exists(input_file):
        run_processor(base_dir, traj_version, to_process, drop_interp, agent_type, dpi)
    with open(input_file, 'rb') as f:
        x = pickle.load(f)

    bins = 80
    qlow, qupp = True, True
    fontcolor = 'dimgray'
    num_airports = len(C.AIRPORT_COLORMAP.keys())
    fig_width = 8  # Adjust as needed
    fig_height = fig_width * num_airports

    if motion_profile == 'heading absolute':
        fig, ax = plt.subplots(num_airports, 1, figsize=(
            fig_width, fig_height), subplot_kw=dict(projection='polar'))
    else:
        fig, ax = plt.subplots(num_airports, 1, figsize=(fig_width, fig_height))

    for i, (airport, data) in enumerate(x.items()):
        # print(f"Selected: {selected_airport}, Current: {airport}, Data: {data.shape}")
        # ax[na].set_xlabel('𝚫Vel (㎨)', color=fontcolor, fontsize=20)
        data = data['mean']

        if qlow:
            q_lower = np.quantile(data, 0.005)
            data = data[data >= q_lower]

        if qupp:
            q_upper = np.quantile(data, 0.995)
            data = data[data <= q_upper]

        _, _, color, label = 0.1, 1, C.AIRPORT_COLORMAP[airport], airport.upper()
        alpha, zorder = 1, 1000
        if (motion_profile == 'heading absolute'):
            polar_histogram(
                ax=ax[i],
                data=data,
                color=color,
                bins=bins,
                offset=np.pi/2)
        else:
            freq, bins, patches = ax[i].hist(
                data,
                bins=bins,
                color=color,
                # edgecolor=fontcolor,
                linewidth=0.1,
                alpha=alpha,
                density=True,
                label=label,
                zorder=zorder
            )
            ax[i].legend(loc='upper right', labelcolor=fontcolor, fontsize=20)

    label, unit = C.MOTION_PROFILE[motion_profile]['label'], C.MOTION_PROFILE[motion_profile]['unit']
    ax[-1].set_xlabel(f'{label} ({unit})', color=fontcolor, fontsize=20)
    for a in ax:
        a.tick_params(color=fontcolor, labelcolor=fontcolor)
        for spine in a.spines.values():
            spine.set_edgecolor(fontcolor)
        if (motion_profile != "heading absolute"):
            a.set_ylabel('Frequency', color=fontcolor, fontsize=20)
            a.ticklabel_format(style='sci', scilimits=(0, 3), axis='both')

    plt.savefig(f"{out_dir}/{base_file}.png", dpi=dpi, bbox_inches='tight')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--base_dir', default=C.DATA_DIR, type=str, help='Input path')
    parser.add_argument('--traj_version', default=C.VERSION, type=str, help='Trajectory version')
    parser.add_argument('--to_process', default=1.0, type=float)
    parser.add_argument('--input_path', default=f"{C.CACHE_DIR}/compute_motion_profiles",
                        type=str, help='Input path')
    parser.add_argument('--motion_profile', default='acceleration',
                        choices=['acceleration', 'speed', 'heading'])
    parser.add_argument('--drop_interp', action='store_true')
    parser.add_argument('--agent_type', default='aircraft',
                        choices=['aircraft', 'vehicle', 'unknown', 'all'])
    parser.add_argument('--dpi', type=int, default=C.DPI)
    args = parser.parse_args()

    plot_vertical_hist(**vars(args))
