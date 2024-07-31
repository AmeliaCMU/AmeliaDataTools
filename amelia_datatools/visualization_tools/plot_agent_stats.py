from turtle import color
import matplotlib.pyplot as plt 
import numpy as np
import os
import pandas as pd
import pickle
import sys

from tqdm import tqdm

sys.path.insert(1, '../utils/')
from common import *

plt.rcParams['font.size'] = 6
num_airports = len(AIRPORT_COLORMAP.keys())

OUT_DIR = os.path.join(VIS_DIR, __file__.split('/')[-1].split(".")[0])
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Created output directory in: {OUT_DIR}")

def plot(ipath: str, dpi: int, num_files: int):
    agent_counts = {}
    agent_types = {
        'Aircraft': [0 for _ in range(num_airports)],
        'Vehicle': [0 for _ in range(num_airports)],
        'Unknown': [0 for _ in range(num_airports)],
    }
    for i, airport in enumerate(AIRPORT_COLORMAP.keys()):
        airport_up = airport.upper()
        print(f"Running airport: {airport_up}")
        agent_counts[airport_up] = []
        airport_dir = os.path.join(ipath, airport)
        traj_files = [os.path.join(airport_dir, f) for f in os.listdir(airport_dir)]

        if num_files == -1:
            num_files = len(traj_files)

        for j, traj_file in enumerate(tqdm(traj_files)):
            if j > num_files:
                break

            data = pd.read_csv(traj_file)
            unique_IDs = data.ID.unique()
            agent_counts[airport_up].append(len(unique_IDs))
            for agent_ID in unique_IDs:
                traj_data = data[:][data.ID == agent_ID]
                agent_type = traj_data.Type.unique()
                if len(agent_type) > 1:
                    continue

                if agent_type == 0.0:
                    agent_types['Aircraft'][i] += 1
                elif agent_type == 1.0:
                    agent_types['Vehicle'][i] += 1
                else:
                    agent_types['Unknown'][i] += 1
                

    # Plot unique agents
    fig, ax = plt.subplots()
    fontcolor = 'dimgray'
    airports = agent_counts.keys()
    counts = [np.asarray(c).sum() for c in agent_counts.values()]
    bar_labels = agent_counts.values()
    bar_colors = AIRPORT_COLORMAP.values()

    ax.bar(airports, counts, color=bar_colors)
    ax.set_ylabel('Num. of Agents', color=fontcolor, fontsize=10)
    ax.set_title('Total Num. of Unique Agents', color=fontcolor, fontsize=15)
    ax.tick_params(color=fontcolor, labelcolor=fontcolor)
    for spine in ax.spines.values():
        spine.set_edgecolor(fontcolor)

    # ax.legend()
    plt.savefig(f"{OUT_DIR}/unique_agents.png", dpi=dpi, bbox_inches='tight')
    plt.close()


    # Plot agent type
    fig, ax = plt.subplots()
    fontcolor = 'dimgray'
    airports = agent_counts.keys()
    bar_colors = AIRPORT_COLORMAP.values()
    
    x = np.arange(len(airports))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    color = {'Aircraft': 'darkblue', 'Vehicle': 'darkred', 'Unknown': 'darkgreen'}
    for attribute, measurement in agent_types.items():
        offset = width * multiplier
        rects = ax.bar(
            x + offset, measurement, width, label=attribute, alpha=0.6, color=color[attribute])
        ax.bar_label(rects, padding=3, fontsize=5, color=fontcolor)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(x + width, airports)
    ax.legend(loc='upper right', ncols=3)
    # ax.set_ylim(0, 250)

    ax.set_ylabel('Num. of Agents', color=fontcolor, fontsize=10)
    ax.set_title('Total Num. of Agents per Type', color=fontcolor, fontsize=15)
    ax.tick_params(color=fontcolor, labelcolor=fontcolor)
    for spine in ax.spines.values():
        spine.set_edgecolor(fontcolor)

    # ax.legend()
    plt.savefig(f"{OUT_DIR}/agent_types.png", dpi=dpi, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--ipath', default='../datasets/amelia/traj_data_a10v7/raw_trajectories', type=str, help='Input path.')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--num_files', type=int, default=-1)
    args = parser.parse_args()

    plot(**vars(args))