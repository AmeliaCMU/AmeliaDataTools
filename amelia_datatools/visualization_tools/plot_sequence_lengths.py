import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from amelia_datatools.utils import common as C
from amelia_datatools.utils import utils


def get_agent_type(agent_type_vals):
    if (agent_type_vals == 0.0).sum() > (agent_type_vals.shape[0]//2):
        return 'Aircraft'
    elif (agent_type_vals == 1.0).sum() > (agent_type_vals.shape[0]//2):
        return 'Vehicle'
    return 'Unknown'


def plot(base_dir: str, traj_version: str,  dpi: int, num_files: int, output_dir: str):
    input_dir = os.path.join(base_dir, f"traj_data_{traj_version}", 'raw_trajectories')
    out_dir = os.path.join(output_dir, utils.get_file_name(__file__))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Created output directory in: {out_dir}")
    plt.rcParams['font.size'] = 6

    airports = C.AIRPORT_COLORMAP.keys()
    num_airports = len(airports)

    seqlens = {

    }
    for i, airport in enumerate(airports):
        airport_up = airport.upper()
        print(f"Running airport: {airport_up}")
        airport_dir = os.path.join(input_dir, airport)
        traj_files = [os.path.join(airport_dir, f) for f in os.listdir(airport_dir)]
        seqlens[airport_up] = []
        N = num_files
        if num_files == -1:
            N = len(traj_files)

        for j, traj_file in enumerate(tqdm(traj_files)):
            if j > N:
                break

            data = pd.read_csv(traj_file)
            unique_IDs = data.ID.unique()
            for agent_ID in unique_IDs:
                traj_len = data[:][data.ID == agent_ID].shape[0]
                seqlens[airport_up].append(traj_len)

    bins = 100
    fontcolor = 'dimgray'
    qlow, qupp = True, True

    nrows = 2
    ncols = num_airports // 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows), sharey=True, squeeze=True)

    for na, (selected_airport, color) in enumerate(C.AIRPORT_COLORMAP.items()):
        print(f"Selected airport: {selected_airport}")
        for airport, data in seqlens.items():

            data = np.asarray(data)
            alpha, zorder, color, label = 0.1, 1, C.AIRPORT_COLORMAP[airport.lower(
            )], airport.upper()
            if airport.upper() == selected_airport.upper():
                alpha, zorder = 0.7, 1000

            if qlow:
                q_lower = np.quantile(data, 0.005)
                data = data[data >= q_lower]

            if qupp:
                q_upper = np.quantile(data, 0.90)
                data = data[data <= q_upper]

            i, j = 0, na
            if na > ncols-1:
                i, j = 1, na - 1 - ncols
            freq, bins, patches = ax[i, j].hist(
                data,
                bins=bins,
                color=color,
                # edgecolor=fontcolor,
                linewidth=0.1,
                alpha=alpha,
                # density=True,
                label=label,
                zorder=zorder
            )

        ax[i, j].legend(
            loc='upper right', labelcolor=fontcolor, fontsize=16, ncols=2)

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
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.savefig(f"{out_dir}/seqlens.png", dpi=dpi, bbox_inches='tight')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--base_dir', default=C.DATA_DIR, type=str, help='Input path')
    parser.add_argument('--traj_version', type=str, default=C.VERSION)
    parser.add_argument('--dpi', type=int, default=C.DPI)
    parser.add_argument('--num_files', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=C.VIS_DIR)
    args = parser.parse_args()

    plot(**vars(args))
