

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from amelia_datatools.utils import common as C
from amelia_datatools.utils import utils as U
from amelia_datatools.utils.processor_utils import polar_histogram
from amelia_datatools.trajectory_data_tools.compute_motion_profiles import run_processor


def format_func(value):
    return "%.1f" % value


def generate_bin_edges(num_bins, min_value, max_value):
    bin_width = (max_value - min_value) / num_bins
    bin_edges = [min_value + i * bin_width for i in range(num_bins + 1)]
    return bin_edges


def plot_vertical_hist(base_dir: str, traj_version: str, airport: str, to_process: bool, input_path: str, motion_profile: str, drop_interp: bool, agent_type: bool, dpi: int, output_dir: str):
    out_dir = os.path.join(output_dir, U.get_file_name(__file__))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Created output directory in: {out_dir}")

    if airport != 'all':
        airports = [airport]
    else:
        airports = U.get_airport_list()

    bins = 80
    qlow, qupp = True, True
    fontcolor = 'dimgray'
    num_airports = len(airports)
    fig_width = 8  # Adjust as needed
    fig_height = fig_width * num_airports

    if motion_profile == 'heading absolute':
        fig, ax = plt.subplots(num_airports, 1, figsize=(
            fig_width, fig_height), subplot_kw=dict(projection='polar'))
    else:
        fig, ax = plt.subplots(num_airports, 1, figsize=(fig_width, fig_height))

    for i, airport in enumerate(airports):

        suffix = '_dropped_int' if drop_interp else ''
        suffix += f'_{agent_type}'
        base_file = f"{motion_profile}_profiles{suffix}"
        input_file = os.path.join(input_path, f"{base_file}_{airport}.pkl")

        if not os.path.exists(input_file):
            run_processor(base_dir, airport, traj_version, to_process, drop_interp, agent_type, dpi)
        with open(input_file, 'rb') as f:
            x = pickle.load(f)

        data = x['mean']

        if qlow:
            q_lower = np.quantile(data, 0.005)
            data = data[data >= q_lower]

        if qupp:
            q_upper = np.quantile(data, 0.995)
            data = data[data <= q_upper]

        if airport not in C.AIRPORT_COLORMAP:
            color = 'black'
        else:
            color = C.AIRPORT_COLORMAP[airport]

        _, _, label = 0.1, 1, airport.upper()
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


def plot_hist(base_dir: str, traj_version: str, airport: str, to_process: bool, input_path: str, motion_profile: str, drop_interp: bool, agent_type: bool, dpi: int, output_dir: str):
    out_dir = os.path.join(output_dir, U.get_file_name(__file__))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Created output directory in: {out_dir}")

    if airport != 'all':
        airports = [airport]
    else:
        airports = U.get_airport_list()

    for airport in airports:

        suffix = '_dropped_int' if drop_interp else ''
        suffix += f'_{agent_type}'
        base_file = f"{motion_profile}_profiles{suffix}_{airport}"
        input_file = os.path.join(input_path, f"{base_file}.pkl")

        if not os.path.exists(input_file):
            run_processor(base_dir, airport, traj_version, to_process, drop_interp, agent_type, dpi)
        with open(input_file, 'rb') as f:
            x = pickle.load(f)

        bins = 80
        qlow, qupp = True, True
        fontcolor = 'dimgray'

        if motion_profile == 'heading absolute':
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        else:
            fig, ax = plt.subplots()

        data = x['mean']

        if qlow:
            q_lower = np.quantile(data, 0.005)
            data = data[data >= q_lower]

        if qupp:
            q_upper = np.quantile(data, 0.995)
            data = data[data <= q_upper]

        if airport not in C.AIRPORT_COLORMAP:
            color = 'black'
        else:
            color = C.AIRPORT_COLORMAP[airport]

        _, _, label = 0.1, 1, airport.upper()
        alpha, zorder = 1, 1000
        if (motion_profile == 'heading absolute'):
            polar_histogram(
                ax=ax,
                data=data,
                color=color,
                bins=bins,
                offset=np.pi/2)
        else:
            freq, bins, patches = ax.hist(
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
            ax.legend(loc='upper right', labelcolor=fontcolor, fontsize=20)

        label, unit = C.MOTION_PROFILE[motion_profile]['label'], C.MOTION_PROFILE[motion_profile]['unit']
        ax.set_xlabel(f'{label} ({unit})', color=fontcolor, fontsize=20)

        ax.tick_params(color=fontcolor, labelcolor=fontcolor)
        for spine in ax.spines.values():
            spine.set_edgecolor(fontcolor)
        if (motion_profile != "heading absolute"):
            ax.set_ylabel('Frequency', color=fontcolor, fontsize=20)
            ax.ticklabel_format(style='sci', scilimits=(0, 3), axis='both')

        plt.savefig(f"{out_dir}/{base_file}.png", dpi=dpi, bbox_inches='tight')


# def plot(ipath: str, motion_profile: str, drop_interp: bool, agent_type: bool, dpi: int):
def plot(base_dir: str, traj_version: str, airport: str, to_process: bool, input_path: str, motion_profile: str, drop_interp: bool, agent_type: bool, dpi: int, output_dir: str):

    out_dir = os.path.join(output_dir, U.get_file_name(__file__))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Created output directory in: {out_dir}")

    if airport != 'all':
        airports = [airport]
    else:
        airports = U.get_airport_list()

    bins = 80
    fontcolor = 'dimgray'
    qlow, qupp = True, True

    nrows = 2
    num_airports = len(airports)
    ncols = num_airports // 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows), sharey=True, squeeze=True)

    for airport in airports:
        for na, selected_airport in enumerate(airports):
            suffix = '_dropped_int' if drop_interp else ''
            suffix += f'_{agent_type}'
            base_file = f"{motion_profile}_profiles{suffix}"
            input_file = os.path.join(input_path, f"{base_file}_{selected_airport}.pkl")

            with open(input_file, 'rb') as f:
                x = pickle.load(f)

            data = x['mean']

            if qlow:
                q_lower = np.quantile(data, 0.005)
                data = data[data >= q_lower]

            if qupp:
                q_upper = np.quantile(data, 0.995)
                data = data[data <= q_upper]

            if airport not in C.AIRPORT_COLORMAP:
                color = 'black'
            else:
                color = C.AIRPORT_COLORMAP[airport]

            alpha, zorder, label = 0.1, 1, airport.upper()
            if airport == selected_airport:
                alpha, zorder = 0.7, 1000

            i, j = 0, na
            if na > ncols-1:
                i, j = 1, na - 1 - ncols

            freq, bins, patches = ax[i, j].hist(
                data, bins=bins, color=color, linewidth=0.1, alpha=alpha, label=label, zorder=zorder
            )

            ax[i, j].legend(loc='upper right', labelcolor=fontcolor, fontsize=20)

    ax[0, 0].set_ylabel('Frequency', color=fontcolor, fontsize=20)
    ax[0, 0].ticklabel_format(style='sci', scilimits=(0, 3), axis='both')
    ax[1, 0].set_ylabel('Frequency', color=fontcolor, fontsize=20)
    ax[1, 0].ticklabel_format(style='sci', scilimits=(0, 3), axis='both')
    for a in ax.reshape(-1):
        a.tick_params(color=fontcolor, labelcolor=fontcolor)
        for spine in a.spines.values():
            spine.set_edgecolor(fontcolor)
            a.yaxis.set_tick_params(labelsize=20)
            a.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.savefig(f"{out_dir}/{base_file}_overlay.png", dpi=dpi, bbox_inches='tight')


if __name__ == '__main__':
    from argparse import ArgumentParser
    airports = U.get_airport_list()
    parser = ArgumentParser()
    parser.add_argument('--base_dir', default=C.DATA_DIR, type=str, help='Input path')
    parser.add_argument('--traj_version', default=C.VERSION, type=str, help='Trajectory version')
    parser.add_argument('--airport', default="all", type=str, choices=["all"] + airports)
    parser.add_argument('--to_process', default=1.0, type=float)
    parser.add_argument('--input_path', default=f"{C.CACHE_DIR}/compute_motion_profiles",
                        type=str, help='Input path')
    parser.add_argument('--motion_profile', default='acceleration',
                        choices=['acceleration', 'speed', 'heading'])
    parser.add_argument('--drop_interp', action='store_true')
    parser.add_argument('--agent_type', default='aircraft',
                        choices=['aircraft', 'vehicle', 'unknown', 'all'])
    parser.add_argument('--dpi', type=int, default=C.DPI)
    parser.add_argument('--output_dir', type=str, default=C.VIS_DIR)
    parser.add_argument('--overlay', action='store_true', default=False)
    parser.add_argument('--plot_vertical', action='store_true', default=False)
    args = parser.parse_args()

    overlay = args.overlay
    plot_vertical = args.plot_vertical
    del args.overlay
    del args.plot_vertical

    if overlay:
        plot(**vars(args))
    elif plot_vertical and args.airport == 'all':
        plot_vertical_hist(**vars(args))
    else:
        plot_hist(**vars(args))
