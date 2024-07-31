import json
import matplotlib.pyplot as plt 
import numpy as np
import os
import pandas as pd
import pickle
import sys

from tqdm import tqdm

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)

sys.path.insert(1, '../utils/')
from common import *

cache_dir = os.path.join(CACHE_DIR, __file__.split('/')[-1].split(".")[0])
os.makedirs(cache_dir, exist_ok=True) 

stats_dir = os.path.join(STATS_DIR, __file__.split('/')[-1].split(".")[0])
os.makedirs(stats_dir, exist_ok=True) 

airport_list = AIRPORT_COLORMAP.keys()
name = {'total': 'Total', '0': 'Aircraft', '1': 'Vehicle', '2': 'Unknown'}

for airport in airport_list:
    if airport not in ['ksea', 'kbos', 'kewr', 'kmdw']:
        continue
    print(f"Running airport {airport.upper()}")

    traj_data_v5 = os.path.join(DATA_DIR, f'traj_data_a4v5/raw_trajectories/{airport}')
    traj_files_v5 = os.listdir(traj_data_v5)
    print(f"Found: {len(traj_files_v5)} files in v5")

    traj_data_v7 = os.path.join(DATA_DIR, f'traj_data_a10v7/raw_trajectories/{airport}')
    traj_files_v7 = os.listdir(traj_data_v7)
    print(f"Found: {len(traj_files_v7)} files in v7")

    common_files = list(set(traj_files_v5).intersection(traj_files_v7))
    if len(common_files) == 0:
        continue
    print(f"Found: {len(common_files)} common files between v5 and v7")
    traj_files_v5 = [os.path.join(traj_data_v5, f) for f in common_files]
    traj_files_v7 = [os.path.join(traj_data_v7, f) for f in common_files]

    diff_ID_files_v5, diff_shape_files_v5 = [], []
    diff_ID_files_v7, diff_shape_files_v7 = [], []
    id_n, n = 0, 0
    for fv5, fv7 in tqdm(zip(traj_files_v5, traj_files_v7)):
        df_v5 = pd.read_csv(fv5)
        df_v7 = pd.read_csv(fv7)
        
        if df_v5.shape[0] != df_v7.shape[0]:
            diff_shape_files_v5.append(fv5)
            diff_shape_files_v7.append(fv7)
            n += 1
            continue

        if not all(df_v5.ID == df_v7.ID):
            diff_ID_files_v5.append(fv5)
            diff_ID_files_v7.append(fv7)
            id_n += 1
            continue

        x_error = np.linalg.norm(df_v5.x - df_v7.x)
        y_error = np.linalg.norm(df_v5.y - df_v7.y)
        z_error = np.linalg.norm(df_v5.Altitude - df_v7.Altitude)
        h_error = np.linalg.norm(df_v5.Heading - df_v7.Heading)
        # print(f"Errors:\n\tx:{x_error}\n\ty:{y_error}\n\tz:{z_error}\n\theading:{h_error}")
    print(f"Diff files: {round(100 * n/len(traj_files_v5), 3)}")
    print(f"Diff ID files: {round(100 * id_n/len(traj_files_v5), 3)}")

    with open(os.path.join(cache_dir, f"{airport}_v5_ID_diff.txt"), 'w') as f:
        f.writelines(diff_ID_files_v5)

    with open(os.path.join(cache_dir, f"{airport}_v7_ID_diff.txt"), 'w') as f:
        f.writelines(diff_ID_files_v7)

    with open(os.path.join(cache_dir, f"{airport}_v5_diff.txt"), 'w') as f:
        f.writelines(diff_shape_files_v5)

    with open(os.path.join(cache_dir, f"{airport}_v7_diff.txt"), 'w') as f:
        f.writelines(diff_shape_files_v7)