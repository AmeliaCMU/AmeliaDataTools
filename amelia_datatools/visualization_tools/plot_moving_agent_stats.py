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


def plot(base_dir: str, traj_version: str, dpi: int, num_files: int, output_dir: str):
    input_dir = os.path.join(base_dir, f"traj_data_{traj_version}", 'raw_trajectories')
    out_dir = os.path.join(out_dir, utils.get_file_name(__file__))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Created output directory in: {out_dir}")

    plt.rcParams['font.size'] = 6

    airports = C.AIRPORT_COLORMAP.keys()
    num_airports = len(airports)

    moving_agents = {
        'All': {
            'Moving': [0 for _ in range(num_airports)],
            'Stationary': [0 for _ in range(num_airports)],
        },
        'Aircraft': {
            'Moving': [0 for _ in range(num_airports)],
            'Stationary': [0 for _ in range(num_airports)],
        },
        'Vehicle': {
            'Moving': [0 for _ in range(num_airports)],
            'Stationary': [0 for _ in range(num_airports)],
        },
        'Unknown': {
            'Moving': [0 for _ in range(num_airports)],
            'Stationary': [0 for _ in range(num_airports)],
        }
    }
    for i, airport in enumerate(airports):
        airport_up = airport.upper()
        print(f"Running airport: {airport_up}")
        airport_dir = os.path.join(input_dir, airport)
        traj_files = [os.path.join(airport_dir, f) for f in os.listdir(airport_dir)]

        N = num_files
        if num_files == -1:
            N = len(traj_files)

        for j, traj_file in enumerate(tqdm(traj_files)):
            if j > N:
                break

            data = pd.read_csv(traj_file)
            unique_IDs = data.ID.unique()
            for agent_ID in unique_IDs:
                traj_data = data[:][data.ID == agent_ID]

                moving_key = 'Stationary' if traj_data.Speed.sum() == 0.0 else 'Moving'
                agent_type = get_agent_type(traj_data.Type.values)

                moving_agents['All'][moving_key][i] += 1
                moving_agents[agent_type][moving_key][i] += 1

    colors = C.AIRPORT_COLORMAP.values()
    fontcolor = 'dimgray'
    width = 0.25  # the width of the bars
    multiplier = 0
    airports = [airport.upper() for airport in airports]

    for agent_type, moving_data in moving_agents.items():

        fig, ax = plt.subplots()
        bottom = np.zeros(shape=num_airports)
        for n, (data_type, data_counts) in enumerate(moving_data.items()):
            alpha = 1.0 if data_type == 'Moving' else 0.4
            p = ax.bar(
                airports, data_counts, width, label=data_type, bottom=bottom, color=colors, alpha=alpha)
            bottom += data_counts

        for spine in ax.spines.values():
            spine.set_edgecolor(fontcolor)

        ax.set_title('Moving vs. Stationary Agents', color=fontcolor, fontsize=15)
        ax.tick_params(color=fontcolor, labelcolor=fontcolor)
        ax.legend()
        plt.savefig(f"{out_dir}/moving_stationary_{agent_type.lower()}.png",
                    dpi=dpi, bbox_inches='tight')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--base_dir', default=C.DATA_DIR, type=str, help='Input path')
    parser.add_argument('--traj_version', default=C.VERSION, type=str)
    parser.add_argument('--dpi', type=int, default=C.DPI)
    parser.add_argument('--num_files', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=C.VIS_DIR)
    args = parser.parse_args()

    plot(**vars(args))
