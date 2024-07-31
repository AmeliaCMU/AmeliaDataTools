from common import AIRPORT_COLORMAP, CACHE_DIR, TRAJ_DATA_DIR, STATS_DIR
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

from tqdm import tqdm

plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

sys.path.insert(1, '../utils/')

cache_dir = os.path.join(CACHE_DIR, __file__.split('/')[-1].split(".")[0])
os.makedirs(cache_dir, exist_ok=True)

stats_dir = os.path.join(STATS_DIR, __file__.split('/')[-1].split(".")[0])
os.makedirs(stats_dir, exist_ok=True)

airport_list = AIRPORT_COLORMAP.keys()
name = {'total': 'Total', '0': 'Aircraft', '1': 'Vehicle', '2': 'Unknown'}


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


def get_agent_type_counts(file_list: list) -> dict:
    print(f"\tGetting agent type counts...", end="\r")
    agent_types = {}
    total_count = 0
    for f in tqdm(file_list):
        data = pd.read_csv(f)

        unique_IDs = np.unique(data.ID)
        for ID in unique_IDs:
            atype = data[data.ID == ID].Type.astype(int)
            atype = atype[np.diff(atype, prepend=np.nan).astype(bool)].astype(str).tolist()
            key = ''.join(atype)

            if agent_types.get(key) is None:
                agent_types[key] = 0
                key_p = f"{key}_perc"
                agent_types[key_p] = 0.0

            agent_types[key] += 1
            total_count += 1

    for k, v in agent_types.items():
        if "perc" in k:
            continue
        kp = f"{k}_perc"
        agent_types[kp] = round(v / total_count, 3)

    agent_types['total'] = total_count

    print(f"\tGetting agent type counts...done.",)
    return agent_types


if __name__ == "__main__":
    for airport in airport_list:
        print(f"Running airport {airport.upper()}")

        stats_file = os.path.join(stats_dir, f"{airport}_stats.json")

        traj_dir = os.path.join(TRAJ_DATA_DIR, "raw_trajectories", airport)
        traj_files = [os.path.join(traj_dir, f) for f in os.listdir(traj_dir)]

        agent_type_counts = get_agent_type_counts(traj_files)
        with open(stats_file, 'w') as f:
            json.dump(agent_type_counts, f, indent=2)
