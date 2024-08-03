import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm

from amelia_datatools.utils.common import CACHE_DIR, STATS_DIR, TRAJ_DATA_DIR, AIRPORT_COLORMAP, AgentType
from amelia_datatools.utils.utils import get_file_name


def get_agent_counts(file_list: list) -> dict:
    print(f"\tGetting agent counts...", end="\r")
    agent_counts = {
        'total': [], '0': [], '1': [], '2': []
    }
    for f in tqdm(file_list):
        data = pd.read_csv(f)

        unique_frames = np.unique(data.Frame)
        for frame in unique_frames:
            agents = data[data.Frame == frame]

            agent_counts['total'].append(agents.shape[0])
            agent_counts['0'].append(np.where(agents.Type == AgentType.AIRCRAFT.value)[0].shape[0])
            agent_counts['1'].append(np.where(agents.Type == AgentType.VEHICLE.value)[0].shape[0])
            agent_counts['2'].append(np.where(agents.Type == AgentType.UNKNOWN.value)[0].shape[0])
    print(f"\tGetting agent counts...done.",)
    return agent_counts


def compute_agent_count_stats(agent_counts: dict) -> dict:
    print(f"\tComputing agent counts statistics...", end='\r')
    agent_count_stats = {}
    for k, v in agent_counts.items():
        v = np.asarray(v)
        agent_count_stats[k] = {
            "min": round(v.min().astype(float), 5),
            "max": round(v.max().astype(float), 5),
            "mean": round(v.mean().astype(float), 5),
            "std": round(v.std().astype(float), 5)
        }

        agent_counts[k] = v
    print(f"\tComputing agent counts statistics...done.")
    return agent_count_stats, agent_counts


def agent_count():

    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    cache_dir = os.path.join(CACHE_DIR, get_file_name(__file__))
    os.makedirs(cache_dir, exist_ok=True)

    stats_dir = os.path.join(STATS_DIR, get_file_name(__file__))
    os.makedirs(stats_dir, exist_ok=True)

    airport_list = AIRPORT_COLORMAP.keys()
    name = {'total': 'Total', '0': 'Aircraft', '1': 'Vehicle', '2': 'Unknown'}

    for airport in airport_list:
        print(f"Running airport {airport.upper()}")

        stats_file = os.path.join(stats_dir, f"{airport}_stats.json")
        cache_file = os.path.join(cache_dir, f"{airport}.pkl")

        traj_dir = os.path.join(TRAJ_DATA_DIR, "raw_trajectories", airport)
        traj_files = [os.path.join(traj_dir, f) for f in os.listdir(traj_dir)]

        agent_counts = get_agent_counts(traj_files)
        agent_counts_stats, agent_counts = compute_agent_count_stats(agent_counts)

        with open(stats_file, 'w') as f:
            json.dump(agent_counts_stats, f, indent=2)

        with open(cache_file, 'wb') as f:
            pickle.dump(agent_counts, f, protocol=pickle.HIGHEST_PROTOCOL)

        N = len(name.keys())
        fig, axs = plt.subplots(nrows=1, ncols=len(name.keys()), sharey=True, figsize=(N * 10, 10))

        for i, (key, value) in enumerate(name.items()):
            arr = agent_counts[key]

            axs[i].hist(arr, bins=max(2, int(arr.max())), color=AIRPORT_COLORMAP[airport])
            axs[i].set_title(f"{value} - {airport.upper()}", fontsize=20)
            axs[i].set_xlabel("Number of Agents per Timestep", fontsize=20)
            if i == 0:
                axs[i].set_ylabel("Count")

        hist_file = os.path.join(stats_dir, f"{airport}.png")
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.02, hspace=0)
        plt.savefig(hist_file, dpi=600, bbox_inches='tight')
        plt.close()

        N = len(name.keys())
        fig, axs = plt.subplots(nrows=1, ncols=len(name.keys()), sharey=True, figsize=(N * 10, 10))
        for i, (key, value) in enumerate(name.items()):
            arr = agent_counts[key]
            q_lower = np.quantile(arr, 0.05)
            q_upper = np.quantile(arr, 0.95)

            arr = arr[(arr >= q_lower) & (arr <= q_upper)]
            axs[i].hist(arr, bins=max(2, int(arr.max())), color=AIRPORT_COLORMAP[airport])
            axs[i].set_title(f"{value} - {airport.upper()}", fontsize=20)
            axs[i].set_xlabel("Number of Agents per Timestep", fontsize=20)
            if i == 0:
                axs[i].set_ylabel("Count")

        hist_file = os.path.join(stats_dir, f"{airport}_q05-95.png")
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.02, hspace=0)
        plt.savefig(hist_file, dpi=600, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    agent_count()
