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
    interp_moving = {}

    total = [0 for _ in range(num_airports)]
    interp = [0 for _ in range(num_airports)]
    for i, airport in enumerate(airports):
        airport_up = airport.upper()
        print(f"Running airport: {airport_up}")
        airport_dir = os.path.join(ipath, airport)
        traj_files = [os.path.join(airport_dir, f) for f in os.listdir(airport_dir)]

        interp_moving[airport_up] = []
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

                total[i] += 1
                if interp_len > 0:
                    interp[i] += 1

                interp_moving[airport_up].append(interp_len/traj_len)
    
    bins = 20
    fontcolor = 'dimgray'
    qlow, qupp = True, True

    nrows = 2
    ncols = num_airports // 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows), sharey=True, squeeze=True)

    for na, (selected_airport, color) in enumerate(AIRPORT_COLORMAP.items()):
        print(f"Selected airport: {selected_airport}")
        for airport, data in interp_moving.items():
            # print(f"Selected: {selected_airport}, Current: {airport}, Data: {data.shape}")
            # ax[na].set_xlabel('𝚫Vel (㎨)', color=fontcolor, fontsize=20)
            
            # if qlow:
            #     q_lower = np.quantile(data, 0.005)
            #     data = data[data >= q_lower]

            # if qupp:
            #     q_upper = np.quantile(data, 0.995)
            #     data = data[data <= q_upper]
            
            alpha, zorder, color, label = 0.1, 1, AIRPORT_COLORMAP[airport.lower()], airport.upper()
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
                linewidth=0.1 , 
                alpha=alpha, 
                # density=True, 
                label=label, 
                zorder=zorder
            )

        ax[i, j].legend(loc='upper right', labelcolor=fontcolor, fontsize=20)

    ax[0, 0].set_ylabel('Frequency', color=fontcolor, fontsize=20)
    ax[0, 0].ticklabel_format(style='sci', scilimits=(0,3),axis='both')
    ax[1, 0].set_ylabel('Frequency', color=fontcolor, fontsize=20)
    ax[1, 0].ticklabel_format(style='sci', scilimits=(0,3),axis='both')
    for a in ax.reshape(-1):
        a.tick_params(color=fontcolor, labelcolor=fontcolor)
        for spine in a.spines.values():
            spine.set_edgecolor(fontcolor)
            a.yaxis.set_tick_params(labelsize=20)
            a.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.savefig(f"{OUT_DIR}/interp_hists.png", dpi=dpi, bbox_inches='tight')

    print(airports)
    interp = np.asarray([int_ / total[i] for i, int_ in enumerate(interp)])
    print(100 * interp)

    

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--ipath', default='../datasets/amelia/traj_data_a10v7/raw_trajectories', type=str, help='Input path.')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--num_files', type=int, default=-1)
    args = parser.parse_args()

    plot(**vars(args))