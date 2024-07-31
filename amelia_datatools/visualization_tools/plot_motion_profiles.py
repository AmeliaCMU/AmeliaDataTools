import json
from locale import normalize
from turtle import color
from attr import s
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import pickle
import sys
import seaborn as sns
from sklearn import base
from tqdm import tqdm

sys.path.insert(1, '../utils/')
from common import *
from processor_utils import polar_histogram

num_airports = len(AIRPORT_COLORMAP.keys())

OUT_DIR = os.path.join(VIS_DIR, __file__.split('/')[-1].split(".")[0])
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Created output directory in: {OUT_DIR}")


def format_func(value, tick_number):
    return "%.1f" % value

def generate_bin_edges(num_bins, min_value, max_value):
    bin_width = (max_value - min_value) / num_bins
    bin_edges = [min_value + i * bin_width for i in range(num_bins + 1)]
    return bin_edges

def plot(ipath: str, motion_profile: str, drop_interp: bool, agent_type: bool, dpi: int):
    suffix = '_dropped_int' if drop_interp else ''
    suffix += f'_{agent_type}'
    base_file = f"{motion_profile}_profiles{suffix}"
    input_file = os.path.join(ipath, f"{base_file}.pkl")
    with open(input_file, 'rb') as f:
        x = pickle.load(f)

    bins = 80
    fontcolor = 'dimgray'
    qlow, qupp = True, True

    nrows = 2
    ncols = num_airports // 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows), sharey=True, squeeze=True)

    for na, (selected_airport, color) in enumerate(AIRPORT_COLORMAP.items()):
        print(f"Selected airport: {selected_airport}")
        for airport, data in x.items():
            # print(f"Selected: {selected_airport}, Current: {airport}, Data: {data.shape}")
            # ax[na].set_xlabel('𝚫Vel (㎨)', color=fontcolor, fontsize=20)
            data = data['mean']

            if qlow:
                q_lower = np.quantile(data, 0.005)
                data = data[data >= q_lower]

            if qupp:
                q_upper = np.quantile(data, 0.995)
                data = data[data <= q_upper]
            
            alpha, zorder, color, label = 0.1, 1, AIRPORT_COLORMAP[airport], airport.upper()
            if airport == selected_airport:
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
    plt.savefig(f"{OUT_DIR}/{base_file}.png", dpi=dpi, bbox_inches='tight')


def plot_heatmap(ipath: str, motion_profile: str, drop_interp: bool, agent_type: bool, dpi: int):
    suffix = '_dropped_int' if drop_interp else ''
    suffix += f'_{agent_type}'
    base_file = f"{motion_profile}_profiles{suffix}"
    input_file = os.path.join(ipath, f"{base_file}.pkl")
    with open(input_file, 'rb') as f:
        x = pickle.load(f)

    fontcolor = 'dimgray'
    qlow, qupp = True, True
    bins = 80
    ticks = 9
    bin_edges = generate_bin_edges(num_bins= bins, min_value= -1, max_value = 1)

    motion_data = {}
    normalize = False

    for i , (airport, data) in enumerate(x.items()):
        data = data['mean']
        if qlow:
                q_lower = np.quantile(data, 0.005)
                data = data[data >= q_lower]

        if qupp:
            q_upper = np.quantile(data, 0.995)
            data = data[data <= q_upper]

        freq, _ = np.histogram(data,bins = bin_edges)
        motion_data[airport] = {motion_profile: bin_edges[:-1], 'Frequency': freq}
      
    dfs = []
    for key, value in motion_data.items():
        df = pd.DataFrame(value)
        df['Airport'] = key
        dfs.append(df)
    airport_df = pd.concat(dfs, ignore_index=True)
    airport_df = airport_df.pivot("Airport", motion_profile, "Frequency")


    if normalize:
        airport_df= airport_df.apply(lambda row: (row-row.mean())/row.std(), axis = 1)

    ax = sns.heatmap(airport_df)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    ax.xaxis.set_major_locator(ticker.LinearLocator(ticks))
    breakpoint()
    ax.set_xticklabels(bin_edges[:-1][::ticks])
    # for a in ax.reshape(-1):
    plt.title('Acceleration Profile')
    plt.xlabel('Acceleration (m/s²)')

    ax.tick_params(color=fontcolor, labelcolor=fontcolor)
    for spine in ax.spines.values():
        spine.set_edgecolor(fontcolor)
        
    plt.savefig(f"{OUT_DIR}/{base_file}_heatmap_sns.png", dpi=dpi, bbox_inches='tight')


def plot_vertical_hist(ipath: str, motion_profile: str, drop_interp: bool, agent_type: bool, dpi: int):
    suffix = '_dropped_int' if drop_interp else ''
    suffix += f'_{agent_type}'
    base_file = f"{motion_profile}_profiles{suffix}"
    input_file = os.path.join(ipath, f"{base_file}.pkl")
    with open(input_file, 'rb') as f:
        x = pickle.load(f)

    bins = 80
    qlow, qupp = True, True
    fontcolor = 'dimgray'

    fig_width = 8  # Adjust as needed
    fig_height = fig_width * num_airports 

    if motion_profile == 'heading absolute':
        fig, ax = plt.subplots(num_airports, 1, figsize = (fig_width, fig_height), subplot_kw=dict(projection='polar'))
    else:
        fig, ax = plt.subplots(num_airports, 1, figsize = (fig_width, fig_height))

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
        
        _, _, color, label = 0.1, 1, AIRPORT_COLORMAP[airport], airport.upper()
        alpha, zorder = 1, 1000
        if(motion_profile == 'heading absolute'):
            polar_histogram(
                ax = ax[i], 
                data = data, 
                color = color,
                bins = bins,
                offset=np.pi/2)
        else:
            freq, bins, patches = ax[i].hist(
                data, 
                bins=bins, 
                color=color, 
                # edgecolor=fontcolor, 
                linewidth=0.1 , 
                alpha=alpha, 
                density=True, 
                label=label, 
                zorder=zorder
            )
            ax[i].legend(loc='upper right', labelcolor=fontcolor, fontsize=20)
    
    label, unit = MOTION_PROFILE[motion_profile]['label'], MOTION_PROFILE[motion_profile]['unit']
    ax[-1].set_xlabel(f'{label} ({unit})', color=fontcolor, fontsize=20)
    for a in ax:
        a.tick_params(color=fontcolor, labelcolor=fontcolor)
        for spine in a.spines.values():
            spine.set_edgecolor(fontcolor)
        if(motion_profile != "heading absolute"):
            a.set_ylabel('Frequency', color=fontcolor, fontsize=20)
            a.ticklabel_format(style='sci', scilimits=(0,3),axis='both')

    plt.savefig(f"{OUT_DIR}/{base_file}.png", dpi=dpi, bbox_inches='tight')

def plot_polar_histogram(ipath: str, motion_profile: str, drop_interp: bool, agent_type: bool, dpi: int):
    suffix = '_dropped_int' if drop_interp else ''
    suffix += f'_{agent_type}'
    base_file = f"{motion_profile}_profiles{suffix}"
    input_file = os.path.join(ipath, f"{base_file}.pkl")
    with open(input_file, 'rb') as f:
        x = pickle.load(f)

    


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--ipath', default='../out/cache/compute_motion_profiles', type=str, help='Input path.')
    parser.add_argument('--motion_profile', default='acceleration', choices=['acceleration', 'speed', 'heading'])
    parser.add_argument('--drop_interp', action='store_true')
    parser.add_argument('--agent_type', default='aircraft', choices=['aircraft', 'vehicle', 'unknown', 'all'])
    parser.add_argument('--dpi', type=int, default=400)
    args = parser.parse_args()

    plot_vertical_hist(**vars(args))

