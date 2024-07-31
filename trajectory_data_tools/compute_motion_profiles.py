import pickle
import cv2
import json
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
    
    def process_file(self, data):
        # data = data[:][data['Type'] == 0]
        agents_ids = data.ID.unique().tolist()
        
        delta_heading_mean, delta_heading_max = [], []
        delta_acceleration_mean, delta_acceleration_max  = [], []
        speed_mean, speed_max = [], []

        for agent_id in agents_ids:
            agent_data = data[:][data.ID == agent_id]
            agent_type = agent_data.Type.values
            if self.agent_types == 'aircraft':
                if (agent_type == 0.0).sum() < (agent_data.shape[0] // 2):
                    continue
            elif self.agent_types == 'vehicle':
                if (agent_type == 1.0).sum() < (agent_data.shape[0] // 2):
                    continue

            time_range = agent_data.Frame.values
            if len(time_range) < 2:
                continue # Skip sequences shorter than 2 steps
        
            # Get all data corresponding to the agent in question
            movement = agent_data[agent_data.Frame.isin(time_range)] # wth is this?
            if self.drop_interp:
                movement = movement[:][movement.Interp == '[ORG]']
                if movement.shape[0] < 2:
                    continue

            # Read speed and heading and get time step wise difference
            agent_heading = movement.Heading.values
            agent_speed = movement.Speed.values * KNOTS_2_MS #m/s

            heading_diff = 180 - abs(abs(agent_heading[1:] - agent_heading[:-1]) - 180) # Degrees
            speed_diff =  agent_speed[1:] - agent_speed[:-1]

            mean_acceleration = speed_diff.mean() #m/s2
            max_acceleration = max(abs(speed_diff))

            mean_heading = heading_diff.mean()
            max_heading = max(abs(heading_diff))

            # Append to information vectors
            delta_heading_mean.append(mean_heading)
            delta_heading_max.append(max_heading)
            
            speed_mean.append(agent_speed.mean())
            speed_max.append(agent_speed.max())

            delta_acceleration_mean.append(mean_acceleration)
            delta_acceleration_max.append(max_acceleration)

        acc = {'mean': delta_acceleration_mean, 'max': delta_acceleration_max}
        speed = {'mean': speed_mean, 'max': speed_max}
        heading = {'mean': delta_heading_mean, 'max': delta_heading_max}
        
        return acc, speed, heading

    def compute_motion_profiles(self):

        acc_profile = {
            "mean": [],
            "max": [],
            # "all": []
        }
        heading_profile = {
            "mean": [],
            "max": [], 
            # "all": []
        }
        speed_profile = {
            "mean": [],
            "max": [], 
            # "all": []
        }
        for trajectory_file in tqdm(self.trajectory_files):
            data = pd.read_csv(trajectory_file)
            acc, speed, heading = self.process_file(data)
            acc_profile['max'] += acc['max']
            acc_profile['mean'] += acc['mean']
            # acc_profile['all'] += acc['all']

            speed_profile['max'] += speed['max']
            speed_profile['mean'] += speed['mean']
            # speed_profile['all'] += speed['all']

            heading_profile['max'] += heading['max']
            heading_profile['mean'] += heading['mean']
            # heading_profile['all'] += heading['all']

        acc_profile['mean'] = np.asarray(acc_profile['mean'])
        acc_profile['max'] = np.asarray(acc_profile['max'])

        speed_profile['mean'] = np.asarray(speed_profile['mean'])
        speed_profile['max'] = np.asarray(speed_profile['max'])

        heading_profile['mean'] = np.asarray(heading_profile['mean'])
        heading_profile['max'] = np.asarray(heading_profile['max'])

        return acc_profile, speed_profile, heading_profile


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

    acc_profile, speed_profile, heading_profile = {}, {}, {}
    for airport in AIRPORT_COLORMAP.keys():
        args.airport = airport
        processor = TrajectoryProcessor(**vars(args))
        acc, speed, heading = processor.compute_motion_profiles()
        acc_profile[airport] = acc
        speed_profile[airport] = speed
        heading_profile[airport] = heading
    
    suffix = '_dropped_int' if args.drop_interp else ''
    suffix += f'_{args.agent_types}'
    out_file = os.path.join(out_dir, f'acceleration_profiles{suffix}.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(acc_profile, f)
    
    out_file = os.path.join(out_dir, f'speed_profiles{suffix}.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(speed_profile, f)

    out_file = os.path.join(out_dir, f'heading_profiles{suffix}.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(heading_profile, f)
    