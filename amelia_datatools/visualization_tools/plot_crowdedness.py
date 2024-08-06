import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

from tqdm import tqdm

from amelia_datatools.utils import common as C
from amelia_datatools.utils import utils


def get_agent_type(agent_type_vals):
    if (agent_type_vals == 0.0).sum() > (agent_type_vals.shape[0]//2):
        return 'Aircraft'
    elif (agent_type_vals == 1.0).sum() > (agent_type_vals.shape[0]//2):
        return 'Vehicle'
    return 'Unknown'


def get_crowdedness(base_dir: str, traj_version: str, num_files: int, output_dir: str):
    input_dir = os.path.join(base_dir, f"traj_data_{traj_version}", 'raw_trajectories')
    out_dir = os.path.join(output_dir, utils.get_file_name(__file__))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory in: {out_dir}")
    plt.rcParams['font.size'] = 6

    airports = C.AIRPORT_COLORMAP.keys()

    crowdedness = {

    }
    for i, airport in enumerate(airports):
        airport_up = airport.upper()
        print(f"Running airport: {airport_up}")
        airport_dir = os.path.join(input_dir, airport)
        traj_files = [os.path.join(airport_dir, f) for f in os.listdir(airport_dir)]
        crowdedness[airport_up] = []
        N = num_files
        if num_files == -1:
            N = len(traj_files)

        for j, traj_file in enumerate(tqdm(traj_files)):
            if j > N:
                break

            data = pd.read_csv(traj_file)
            frames = data.Frame.unique()
            for frame in frames:
                unique_IDs = data[:][data.Frame == frame].shape[0]
                crowdedness[airport_up].append(unique_IDs)
    return crowdedness


def plot(input_path: str, dpi: int, num_files: int, output_dir: str):
    out_dir = os.path.join(out_dir, utils.get_file_name(__file__))
    os.makedirs(out_dir, exist_ok=True)
    airports = C.AIRPORT_COLORMAP.keys()
    num_airports = len(airports)

    crowdedness = {

    }
    for i, airport in enumerate(airports):
        airport_up = airport.upper()
        print(f"Running airport: {airport_up}")
        airport_dir = os.path.join(input_path, airport)
        traj_files = [os.path.join(airport_dir, f) for f in os.listdir(airport_dir)]
        crowdedness[airport_up] = []
        N = num_files
        if num_files == -1:
            N = len(traj_files)

        for j, traj_file in enumerate(tqdm(traj_files)):
            if j > N:
                break

            data = pd.read_csv(traj_file)
            frames = data.Frame.unique()
            for frame in frames:
                unique_IDs = data[:][data.Frame == frame].shape[0]
                crowdedness[airport_up].append(unique_IDs)

    bins = 50
    fontcolor = 'dimgray'
    qlow, qupp = True, True

    nrows = 2
    ncols = num_airports // 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(9 * ncols, 8 * nrows), sharey=True, squeeze=True)

    for na, (selected_airport, color) in enumerate(C.AIRPORT_COLORMAP.items()):
        print(f"Selected airport: {selected_airport}")
        for airport, data in crowdedness.items():

            alpha, zorder, color, label = 0.1, 1, C.AIRPORT_COLORMAP[airport.lower(
            )], airport.upper()
            if airport.upper() == selected_airport.upper():
                alpha, zorder = 0.7, 1000

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

    ax[0, 0].set_ylabel('Frequency', color=fontcolor, fontsize=15)
    ax[0, 0].ticklabel_format(style='sci', scilimits=(0, 3), axis='both')
    ax[1, 0].set_ylabel('Frequency', color=fontcolor, fontsize=15)
    ax[1, 0].ticklabel_format(style='sci', scilimits=(0, 3), axis='both')
    for a in ax.reshape(-1):
        a.tick_params(color=fontcolor, labelcolor=fontcolor)
        for spine in a.spines.values():
            spine.set_edgecolor(fontcolor)
            a.yaxis.set_tick_params(labelsize=20)
            a.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.savefig(f"{out_dir}/crowded.png", dpi=dpi, bbox_inches='tight')


def plot_vertical(input_path: str, dpi: int, output_dir: str):
    out_dir = os.path.join(output_dir, utils.get_file_name(__file__))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Created output directory in: {out_dir}")
    airports = C.AIRPORT_COLORMAP.keys()
    num_airports = len(airports)

    with open(input_path, 'rb') as f:
        crowdedness = pickle.load(f)

    bins = 50
    fontcolor = 'dimgray'
    # qlow, qupp = True, True

    fig_width = 8  # Adjust as needed
    fig_height = fig_width * num_airports

    fig, ax = plt.subplots(num_airports, 1, figsize=(fig_width, fig_height))

    alpha, zorder = 1, 1000

    for i, (airport, data) in enumerate(crowdedness.items()):

        _, _, color, label = 0.1, 1, C.AIRPORT_COLORMAP[airport.lower()], airport.upper()

        freq, bins, patches = ax[i].hist(
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

        ax[i].legend(
            loc='upper right', labelcolor=fontcolor, fontsize=16, ncols=2)

    ax[-1].set_xlabel(f'Simultaneos agents per frame', color=fontcolor, fontsize=20)
    for a in ax:
        a.tick_params(color=fontcolor, labelcolor=fontcolor)
        for spine in a.spines.values():
            spine.set_edgecolor(fontcolor)
            a.yaxis.set_tick_params(labelsize=20)
            a.xaxis.set_tick_params(labelsize=20)
            a.set_ylabel('Frequency', color=fontcolor, fontsize=20)
            a.set_ylabel('Frequency', color=fontcolor, fontsize=20)

    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.savefig(f"{out_dir}/crowded.png", dpi=dpi, bbox_inches='tight')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--base_dir', default=C.DATA_DIR, type=str, help='Input path')
    parser.add_argument('--traj_version', default=C.VERSION, type=str)
    parser.add_argument('--dpi', type=int, default=C.DPI)
    parser.add_argument('--num_files', type=int, default=-1)
    parser.add_argument('--process', action='store_true')
    parser.add_argument('--output_dir', type=str, default=C.VIS_DIR)
    args = parser.parse_args()

    cache_dir = os.path.join(C.CACHE_DIR, utils.get_file_name(__file__))
    os.makedirs(cache_dir, exist_ok=True)
    cahe_file = os.path.join(cache_dir, f'crowdedness.pkl')
    if (args.process):
        crowdedness = get_crowdedness(base_dir=args.base_dir,
                                      traj_version=args.traj_version,
                                      num_files=args.num_files,
                                      output_dir=args.output_dir)
        with open(cahe_file, 'wb') as f:
            pickle.dump(crowdedness, f)

    plot_vertical(input_path=cahe_file, dpi=args.dpi, output_dir=args.output_dir)
