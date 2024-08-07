import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from amelia_datatools.utils import common as C
from amelia_datatools.utils import utils


def plot_moving(base_dir: str, traj_version: str, dpi: int, num_files: int, output_dir: str):
    input_dir = os.path.join(base_dir, f"traj_data_{traj_version}", 'raw_trajectories')
    out_dir = os.path.join(output_dir, utils.get_file_name(__file__))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Created output directory in: {out_dir}")
    plt.rcParams['font.size'] = 6

    airports = C.AIRPORT_COLORMAP.keys()
    num_airports = len(airports)
    timestep_count = [0 for _ in range(num_airports)]
    interp_counts = {
        'Valid': [0 for _ in range(num_airports)],
        'Interp': [0 for _ in range(num_airports)],
    }
    # 'Moving-Agents': [],
    # 'Stationary-Agent': [],
    agent_motion = {}
    for i, airport in enumerate(airports):
        airport_up = airport.upper()
        print(f"Running airport: {airport_up}")
        airport_dir = os.path.join(input_dir, airport)
        traj_files = [os.path.join(airport_dir, f) for f in os.listdir(airport_dir)]

        if num_files == -1:
            num_files = len(traj_files)

        for j, traj_file in enumerate(tqdm(traj_files)):
            if j > num_files:
                break

            data = pd.read_csv(traj_file)
            unique_IDs = data.ID.unique()
            for agent_ID in unique_IDs:
                traj_data = data[:][data.ID == agent_ID]
                total_speed = traj_data.Speed.sum()
                if total_speed == 0.0:
                    continue

                traj_len = traj_data.shape[0]
                timestep_count[i] += traj_len

                interp_data = traj_data[:][traj_data.Interp == '[INT]']
                interp_len = interp_data.shape[0]
                interp_counts['Valid'][i] += (traj_len - interp_len)
                interp_counts['Interp'][i] += interp_len

    fig, ax = plt.subplots()
    airports = [airport.upper() for airport in airports]
    bottom = np.zeros(shape=num_airports)
    fontcolor = 'dimgray'
    width = 0.6
    colors = C.AIRPORT_COLORMAP.values()
    for n, (data_type, data_counts) in enumerate(interp_counts.items()):
        alpha = 1.0 if data_type == 'Valid' else 0.4
        p = ax.bar(
            airports, data_counts, width, label=data_type, bottom=bottom, color=colors, alpha=alpha)
        bottom += data_counts

    for spine in ax.spines.values():
        spine.set_edgecolor(fontcolor)

    ax.set_title('Interpolated Data Stats by Airport', color=fontcolor, fontsize=15)
    ax.tick_params(color=fontcolor, labelcolor=fontcolor)
    ax.legend()
    plt.savefig(f"{out_dir}/interp_data_moving.png", dpi=dpi, bbox_inches='tight')
    # plt.show()

    interp_counts['Valid'] = [v / timestep_count[i] for i, v in enumerate(interp_counts['Valid'])]
    interp_counts['Interp'] = [v / timestep_count[i] for i, v in enumerate(interp_counts['Interp'])]
    fig, ax = plt.subplots()
    airports = [airport.upper() for airport in airports]
    bottom = np.zeros(shape=num_airports)
    fontcolor = 'dimgray'
    width = 0.6
    colors = C.AIRPORT_COLORMAP.values()
    for n, (data_type, data_counts) in enumerate(interp_counts.items()):
        alpha = 1.0 if data_type == 'Valid' else 0.4
        p = ax.bar(
            airports, data_counts, width, label=data_type, bottom=bottom, color=colors, alpha=alpha)
        bottom += data_counts

    for spine in ax.spines.values():
        spine.set_edgecolor(fontcolor)

    ax.set_title('Interpolated Data Percentage by Airport', color=fontcolor, fontsize=15)
    ax.tick_params(color=fontcolor, labelcolor=fontcolor)
    ax.legend()
    plt.savefig(f"{out_dir}/interp_data_perc_moving.png", dpi=dpi, bbox_inches='tight')


def plot(base_dir: str, traj_version: str, dpi: int, num_files: int, output_dir: str):
    input_dir = os.path.join(base_dir, f"traj_data_{traj_version}", 'raw_trajectories')
    out_dir = os.path.join(output_dir, utils.get_file_name(__file__))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Created output directory in: {out_dir}")
    plt.rcParams['font.size'] = 6

    airports = C.AIRPORT_COLORMAP.keys()
    num_airports = len(airports)
    timestep_count = [0 for _ in range(num_airports)]
    interp_counts = {
        'Valid': [0 for _ in range(num_airports)],
        'Interp': [0 for _ in range(num_airports)],
    }
    # 'Moving-Agents': [],
    # 'Stationary-Agent': [],
    agent_motion = {}
    for i, airport in enumerate(airports):
        airport_up = airport.upper()
        print(f"Running airport: {airport_up}")
        airport_dir = os.path.join(input_dir, airport)
        traj_files = [os.path.join(airport_dir, f) for f in os.listdir(airport_dir)]

        if num_files == -1:
            num_files = len(traj_files)

        for j, traj_file in enumerate(tqdm(traj_files)):
            if j > num_files:
                break

            data = pd.read_csv(traj_file)
            unique_IDs = data.ID.unique()
            for agent_ID in unique_IDs:
                traj_data = data[:][data.ID == agent_ID]

                traj_len = traj_data.shape[0]
                timestep_count[i] += traj_len

                interp_data = traj_data[:][traj_data.Interp == '[INT]']
                interp_len = interp_data.shape[0]
                interp_counts['Valid'][i] += (traj_len - interp_len)
                interp_counts['Interp'][i] += interp_len

    fig, ax = plt.subplots()
    airports = [airport.upper() for airport in airports]
    bottom = np.zeros(shape=num_airports)
    fontcolor = 'dimgray'
    width = 0.6
    colors = C.AIRPORT_COLORMAP.values()
    for n, (data_type, data_counts) in enumerate(interp_counts.items()):
        alpha = 1.0 if data_type == 'Valid' else 0.4
        p = ax.bar(
            airports, data_counts, width, label=data_type, bottom=bottom, color=colors, alpha=alpha)
        bottom += data_counts

    for spine in ax.spines.values():
        spine.set_edgecolor(fontcolor)

    ax.set_title('Interpolated Data Stats by Airport', color=fontcolor, fontsize=15)
    ax.tick_params(color=fontcolor, labelcolor=fontcolor)
    ax.legend()
    plt.savefig(f"{out_dir}/interp_data_all.png", dpi=dpi, bbox_inches='tight')
    # plt.show()

    interp_counts['Valid'] = [v / timestep_count[i] for i, v in enumerate(interp_counts['Valid'])]
    interp_counts['Interp'] = [v / timestep_count[i] for i, v in enumerate(interp_counts['Interp'])]
    fig, ax = plt.subplots()
    airports = [airport.upper() for airport in airports]
    bottom = np.zeros(shape=num_airports)
    fontcolor = 'dimgray'
    width = 0.6
    colors = C.AIRPORT_COLORMAP.values()
    for n, (data_type, data_counts) in enumerate(interp_counts.items()):
        alpha = 1.0 if data_type == 'Valid' else 0.4
        p = ax.bar(
            airports, data_counts, width, label=data_type, bottom=bottom, color=colors, alpha=alpha)
        bottom += data_counts

    for spine in ax.spines.values():
        spine.set_edgecolor(fontcolor)

    ax.set_title('Interpolated Data Percentage by Airport', color=fontcolor, fontsize=15)
    ax.tick_params(color=fontcolor, labelcolor=fontcolor)
    ax.legend()
    plt.savefig(f"{out_dir}/interp_data_perc_all.png", dpi=dpi, bbox_inches='tight')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--base_dir', default=C.DATA_DIR, type=str, help='Input path')
    parser.add_argument('--traj_version', default=C.VERSION, type=str)
    parser.add_argument('--moving', action='store_true', help='Filter by moving agents')
    parser.add_argument('--dpi', type=int, default=C.DPI)
    parser.add_argument('--num_files', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=C.VIS_DIR)
    args = parser.parse_args()

    if args.moving:
        plot_moving(args.base_dir, args.traj_version, args.dpi, args.num_files)
    else:
        plot(args.base_dir, args.traj_version, args.dpi, args.num_files, args.output_dir)
