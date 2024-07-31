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
OUT_DIR = os.path.join(VIS_DIR, __file__.split('/')[-1].split(".")[0])
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Created output directory in: {OUT_DIR}")

def plot(ipath: str, dpi: int, num_files: int):
    
    airports = AIRPORT_COLORMAP.keys()
    num_airports = len(airports)
    timestep_count = [0 for _ in range(num_airports)]
    interp_counts = {
        'Valid' : [0 for _ in range(num_airports)],
        'Interp': [0 for _ in range(num_airports)],
    }
    # 'Moving-Agents': [],
    # 'Stationary-Agent': [],
    agent_motion = {}
    for i, airport in enumerate(airports):
        airport_up = airport.upper()
        print(f"Running airport: {airport_up}")
        airport_dir = os.path.join(ipath, airport)
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
                interp_counts['Valid'][i]  += (traj_len - interp_len)
                interp_counts['Interp'][i] += interp_len

                # total_speed = traj_data.Speed.sum()
                # if total_speed == 0.0:
                #     interp_counts[airport_up]['Stationary-Agents'].append(interp_len/traj_len)
                # else:
                #     interp_counts[airport_up]['Moving-Agents'].append(interp_len/traj_len)

    fig, ax = plt.subplots()
    airports = [airport.upper() for airport in airports]
    bottom = np.zeros(shape=num_airports)
    fontcolor = 'dimgray'
    width = 0.6 
    colors = AIRPORT_COLORMAP.values()
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
    plt.savefig(f"{OUT_DIR}/interp_data.png", dpi=dpi, bbox_inches='tight')
    # plt.show()

    interp_counts['Valid'] = [v / timestep_count[i] for i, v in enumerate(interp_counts['Valid'])]
    interp_counts['Interp'] = [v / timestep_count[i] for i, v in enumerate(interp_counts['Interp'])]
    fig, ax = plt.subplots()
    airports = [airport.upper() for airport in airports]
    bottom = np.zeros(shape=num_airports)
    fontcolor = 'dimgray'
    width = 0.6 
    colors = AIRPORT_COLORMAP.values()
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
    plt.savefig(f"{OUT_DIR}/interp_data_perc.png", dpi=dpi, bbox_inches='tight')
    # plt.show()
    

    # Plot unique agents
    # fig, ax = plt.subplots()
    # airports = agent_counts.keys()
    # counts = [np.asarray(c).sum() for c in agent_counts.values()]
    # bar_labels = agent_counts.values()
    # bar_colors = AIRPORT_COLORMAP.values()

    # ax.bar(airports, counts, color=bar_colors)
    # ax.set_ylabel('Num. of Agents', color=fontcolor, fontsize=10)
    # ax.set_title('Total Num. of Unique Agents', color=fontcolor, fontsize=15)

    # # ax.legend()
    # plt.savefig(f"{OUT_DIR}/unique_agents.png", dpi=dpi, bbox_inches='tight')
    # plt.close()


    # # Plot agent type
    # fig, ax = plt.subplots()
    # fontcolor = 'dimgray'
    # airports = agent_counts.keys()
    # bar_colors = AIRPORT_COLORMAP.values()
    
    # x = np.arange(len(airports))  # the label locations
    # width = 0.25  # the width of the bars
    # multiplier = 0

    # fig, ax = plt.subplots(layout='constrained')

    # color = {'Aircraft': 'darkblue', 'Vehicle': 'darkred', 'Unknown': 'darkgreen'}
    # for attribute, measurement in agent_types.items():
    #     offset = width * multiplier
    #     rects = ax.bar(
    #         x + offset, measurement, width, label=attribute, alpha=0.6, color=color[attribute])
    #     ax.bar_label(rects, padding=3, fontsize=5, color=fontcolor)
    #     multiplier += 1

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_xticks(x + width, airports)
    # ax.legend(loc='upper right', ncols=3)
    # # ax.set_ylim(0, 250)

    # ax.set_ylabel('Num. of Agents', color=fontcolor, fontsize=10)
    # ax.set_title('Total Num. of Agents per Type', color=fontcolor, fontsize=15)
    # ax.tick_params(color=fontcolor, labelcolor=fontcolor)
    # for spine in ax.spines.values():
    #     spine.set_edgecolor(fontcolor)

    # # ax.legend()
    # plt.savefig(f"{OUT_DIR}/agent_types.png", dpi=dpi, bbox_inches='tight')
    # plt.close()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--ipath', default='../datasets/amelia/traj_data_a10v7/raw_trajectories', type=str, help='Input path.')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--num_files', type=int, default=-1)
    args = parser.parse_args()

    plot(**vars(args))