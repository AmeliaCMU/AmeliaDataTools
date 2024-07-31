import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sys

sys.path.insert(1, '../utils/')
from tqdm import tqdm

from common import *

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

        trajectories_dir = os.path.join(
            self.base_dir, f'traj_data_{version}', 'raw_trajectories', self.airport)
        print(f"Analyzing data in: {trajectories_dir}")
        self.trajectory_files = [os.path.join(trajectories_dir, f) for f in os.listdir(trajectories_dir)]
        random.shuffle(self.trajectory_files)
        self.trajectory_files = self.trajectory_files[:int(len(self.trajectory_files) * to_process)]

    def count_nans(self):
        affected_files = []
        nan_count = 0
        for trajectory_file in tqdm(self.trajectory_files):
            data = pd.read_csv(trajectory_file)
            nan_count += data.isna().sum().sum()
            if data.isna().any().any():
                affected_files.append(trajectory_file)

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
        files, count = processor.count_nans()
        affected_files+= files
        nan_count += count
        print(F"Found {count} nans for {airport}")

    print(f"Dataset contains {nan_count} nans")

    out_file = os.path.join(out_dir, f'files_with_nans.pkl')
    with open(f'affected_files.pickle', 'wb') as f:
            pickle.dump(affected_files, f)