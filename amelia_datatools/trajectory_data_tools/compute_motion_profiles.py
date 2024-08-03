from tqdm import tqdm
import pickle
import numpy as np
import os
import pandas as pd
import random

from amelia_datatools.utils.common import DPI, VERSION, CACHE_DIR, KNOTS_2_MS, AIRPORT_COLORMAP, DATA_DIR


class TrajectoryProcessor():
    def __init__(
        self, airport: str, base_dir: str, traj_version: str, to_process: float, drop_interp: bool,
        agent_type: str, dpi: int
    ):
        self.base_dir = base_dir

        self.airport = airport
        self.dpi = dpi
        self.drop_interp = drop_interp
        self.agent_type = agent_type

        trajectories_dir = os.path.join(
            self.base_dir, f'traj_data_{traj_version}', 'raw_trajectories', self.airport)
        print(f"Analyzing data in: {trajectories_dir}")
        self.trajectory_files = [os.path.join(trajectories_dir, f)
                                 for f in os.listdir(trajectories_dir)]
        random.shuffle(self.trajectory_files)
        self.trajectory_files = self.trajectory_files[:int(len(self.trajectory_files) * to_process)]

    def process_file(self, data):
        # data = data[:][data['Type'] == 0]
        agents_ids = data.ID.unique().tolist()

        delta_heading_mean, delta_heading_max = [], []
        delta_acceleration_mean, delta_acceleration_max = [], []
        speed_mean, speed_max = [], []

        for agent_id in agents_ids:
            agent_data = data[:][data.ID == agent_id]
            agent_type = agent_data.Type.values
            if self.agent_type == 'aircraft':
                if (agent_type == 0.0).sum() < (agent_data.shape[0] // 2):
                    continue
            elif self.agent_type == 'vehicle':
                if (agent_type == 1.0).sum() < (agent_data.shape[0] // 2):
                    continue

            time_range = agent_data.Frame.values
            if len(time_range) < 2:
                continue  # Skip sequences shorter than 2 steps

            # Get all data corresponding to the agent in question
            movement = agent_data[agent_data.Frame.isin(time_range)]  # wth is this?
            if self.drop_interp:
                movement = movement[:][movement.Interp == '[ORG]']
                if movement.shape[0] < 2:
                    continue

            # Read speed and heading and get time step wise difference
            agent_heading = movement.Heading.values
            agent_speed = movement.Speed.values * KNOTS_2_MS  # m/s

            heading_diff = 180 - abs(abs(agent_heading[1:] - agent_heading[:-1]) - 180)  # Degrees
            speed_diff = agent_speed[1:] - agent_speed[:-1]

            mean_acceleration = speed_diff.mean()  # m/s2
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


def run_processor(base_path, traj_version, to_process, drop_interp, agent_type, dpi):
    out_dir = os.path.join(CACHE_DIR, __file__.split('/')[-1].split(".")[0])
    os.makedirs(out_dir, exist_ok=True)
    acc_profile, speed_profile, heading_profile = {}, {}, {}
    for airport in AIRPORT_COLORMAP.keys():
        processor = TrajectoryProcessor(airport, base_path, traj_version,
                                        to_process, drop_interp, agent_type, dpi)
        acc, speed, heading = processor.compute_motion_profiles()
        acc_profile[airport] = acc
        speed_profile[airport] = speed
        heading_profile[airport] = heading

    suffix = '_dropped_int' if drop_interp else ''
    suffix += f'_{agent_type}'
    out_file = os.path.join(CACHE_DIR, 'compute_motion_profiles',
                            f'acceleration_profiles{suffix}.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(acc_profile, f)

    out_file = os.path.join(CACHE_DIR, 'compute_motion_profiles', f'speed_profiles{suffix}.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(speed_profile, f)

    out_file = os.path.join(CACHE_DIR, 'compute_motion_profiles', f'heading_profiles{suffix}.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(heading_profile, f)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
<<<<<<< HEAD

=======
>>>>>>> 8a0c86b (Refactored tarajectory tools (#3))
    parser.add_argument('--base_dir', default=DATA_DIR, type=str, help='Input path')
    parser.add_argument('--traj_version', type=str, default=VERSION)
    parser.add_argument('--to_process', default=1.0, type=float)
    parser.add_argument('--drop_interp', action='store_true')
    parser.add_argument('--agent_type', default='aircraft',
                        choices=['aircraft', 'vehicle', 'unknown', 'all'])
    parser.add_argument('--dpi', type=int, default=DPI)
    args = parser.parse_args()

    run_processor(**vars(args))
