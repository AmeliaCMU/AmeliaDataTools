from ast import Return
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sys
from multiprocessing import Pool, cpu_count


N_JOBS = 10 

sys.path.insert(1, '../utils/')
from tqdm import tqdm

from common import *

def load_trajectory_files(subdir):
    """
    Load trajectory files in a given subdirectory.
    """
    return [os.path.join(trajectories_dir, subdir, f) for f in os.listdir(os.path.join(trajectories_dir, subdir))]

def count_nans_helper(trajectory_file):
    """
    Helper function to count NaNs in a trajectory file.
    """
    nan_count = 0
    affected_files = []
    with open(trajectory_file, "rb") as file:
        scene_dict = pickle.load(file)
    trajectories = scene_dict['sequences']
    if np.isnan(trajectories).any():
        print("Found nans")
        nan_count += 1
        affected_files.append(trajectory_file)
    return nan_count, affected_files


class TrajectoryProcessor():
    def __init__(
        self, airport: str, ipath: str, version: str, to_process: float, drop_interp: bool,
        agent_types: str, dpi: int
    ):
        self.base_dir = ipath

        self.airport = airport
        self.dpi = dpi 
        self.drop_interp = drop_interp
        self.agent_types = agent_types

        global trajectories_dir 
        trajectories_dir =os.path.join(
            self.base_dir, f'traj_data_{version}', 'proc_trajectories', self.airport)
        
        self.valid_dir = os.path.exists(trajectories_dir)
        if(self.valid_dir):
            print(f"Loading file list in: {trajectories_dir}")
            self.trajectory_files = []
            subdirs = os.listdir(trajectories_dir)
            with Pool(N_JOBS) as pool:
                results = list(tqdm(pool.imap(load_trajectory_files, subdirs), total=len(subdirs)))

            for result in results:
                self.trajectory_files.extend(result)

            self.trajectory_files = self.trajectory_files[:int(len(self.trajectory_files) * to_process)]

    def count_nans(self):
        with Pool(N_JOBS) as pool:
            results = list(tqdm(pool.imap(count_nans_helper, self.trajectory_files), total=len(self.trajectory_files)))

        for result in results:
            nan_count += result[0]
            affected_files.extend(result[1])

        print(f"Total NaN count: {nan_count}")
        return affected_files, nan_count

    




if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--ipath', default='../datasets/amelia', type=str, help='Input path.')
    parser.add_argument('--version', type=str, default='a10v08')
    parser.add_argument('--to_process', default=1.0,type=float)
    parser.add_argument('--drop_interp', action='store_true')
    parser.add_argument('--agent_types', default='aircraft', choices=['aircraft', 'vehicle', 'unknown', 'all'])
    parser.add_argument('--dpi', type=int, default=800)
    args = parser.parse_args()

    out_dir = os.path.join(CACHE_DIR, __file__.split('/')[-1].split(".")[0])
    os.makedirs(out_dir, exist_ok=True)

    affected_files, nan_count = [], 0 
    for airport in AIRPORT_COLORMAP.keys():
        args.airport = airport
        processor = TrajectoryProcessor(**vars(args))
        if(not processor.valid_dir):
            continue
        files, count = processor.count_nans()
        affected_files+= files
        nan_count += count
        print(F"Found {count} nans for {airport}")

    print(f"Dataset contains {nan_count} nans")

    out_file = os.path.join(out_dir, f'files_with_nans.pkl')
    with open(f'affected_files.pickle', 'wb') as f:
            pickle.dump(affected_files, f)