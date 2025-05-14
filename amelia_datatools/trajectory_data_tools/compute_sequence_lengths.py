import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm

from amelia_datatools.utils.common import TRAJ_DATA_DIR, STATS_DIR, CACHE_DIR, AIRPORT_COLORMAP
from amelia_datatools.utils.utils import get_file_name


def get_sequence_counts(file_list: list) -> dict:
    print(f"\tGetting sequence counts...", end="\r")
    agent_seqlens = {
        'total': [],
    }
    for f in tqdm(file_list):
        data = pd.read_csv(f)

        unique_IDs = np.unique(data.ID)
        for ID in unique_IDs:
            seq = data[data.ID == ID]

            atype = seq.Type.astype(int)
            atype = atype[np.diff(atype, prepend=np.nan).astype(bool)].astype(str).tolist()

            key = ''.join(atype)
            if agent_seqlens.get(key) is None:
                agent_seqlens[key] = []

            T = seq.shape[0]
            agent_seqlens[key].append(T)
            agent_seqlens['total'].append(T)
    print(f"\tGetting sequence counts...done.",)
    return agent_seqlens


def compute_sequence_count_stats(sequence_counts: dict) -> dict:
    print(f"\tComputing sequence counts statistics...", end='\r')
    agent_count_stats = {}
    for k, v in sequence_counts.items():
        v = np.asarray(v)
        agent_count_stats[k] = {
            "min": round(v.min().astype(float), 5),
            "max": round(v.max().astype(float), 5),
            "mean": round(v.mean().astype(float), 5),
            "std": round(v.std().astype(float), 5)
        }

        sequence_counts[k] = v
    print(f"\tComputing sequence counts statistics...done.")
    return agent_count_stats, sequence_counts


def compute_lenghts():
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
        traj_files = [os.path.join(traj_dir, f) for f in os.listdir(traj_dir) if f.endswith('.csv')]

        sequence_counts = get_sequence_counts(traj_files)
        sequence_counts_stats, sequence_counts = compute_sequence_count_stats(sequence_counts)

        with open(stats_file, 'w') as f:
            json.dump(sequence_counts_stats, f, indent=2)

        with open(cache_file, 'wb') as f:
            pickle.dump(sequence_counts, f, protocol=pickle.HIGHEST_PROTOCOL)

        N = len(name.keys())
        fig, axs = plt.subplots(nrows=1, ncols=len(name.keys()), sharey=True, figsize=(N * 10, 10))

        for i, (key, value) in enumerate(name.items()):
            arr = sequence_counts[key]

            axs[i].hist(arr, bins=int(arr.max()), color=AIRPORT_COLORMAP[airport])
            axs[i].set_title(f"{value} - {airport.upper()}", fontsize=20)
            axs[i].set_xlabel("Sequence Lengths", fontsize=20)
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
            arr = sequence_counts[key]
            q_lower = np.quantile(arr, 0.05)
            q_upper = np.quantile(arr, 0.95)

            arr = arr[(arr >= q_lower) & (arr <= q_upper)]
            axs[i].hist(arr, bins=int(arr.max()), color=AIRPORT_COLORMAP[airport])
            axs[i].set_title(f"{value} - {airport.upper()}", fontsize=20)
            axs[i].set_xlabel("Sequence Lengths", fontsize=20)
            if i == 0:
                axs[i].set_ylabel("Count")

        hist_file = os.path.join(stats_dir, f"{airport}_q05-95.png")
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.02, hspace=0)
        plt.savefig(hist_file, dpi=600, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    compute_lenghts()
