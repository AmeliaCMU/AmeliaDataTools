import os
import numpy as np

import math
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import multiprocessing

from amelia_datatools.utils.common import AIRPORT_COLORMAP


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TrajectoryProcessor():
    def __init__(self, base_dir: str, traj_version: str, output_dir: str, airport: str, parrallel: bool):
        self.base_dir = base_dir
        self.out_dir = output_dir
        self.airport = airport
        self.parrallel = parrallel
        self.trajectories_dir = os.path.join(
            self.base_dir, f'traj_data_{traj_version}/raw_trajectories', airport)
        print('---- Analyzing data in ', self.trajectories_dir, '----')
        self.trajectories_files = [f for f in os.listdir(self.trajectories_dir)]
        os.makedirs(self.out_dir, exist_ok=True)

    def process_file(self, file):
        in_file = os.path.join(self.trajectories_dir, file)
        data = pd.read_csv(in_file)
        # data = data[:][data['Type'] == 0]
        agents_ids = data.ID.unique().tolist()

        delta_heading_mean = []
        delta_heading_max = []

        delta_acceleration_mean = []
        delta_acceleration_max = []

        for test_id in agents_ids:
            ego_data = data[:][data.ID == test_id]
            time_range = ego_data['Frame'].values
            if len(time_range) < 2:
                continue  # Skip for sequences less than 2 frames
            # Get all data corresponding to the agent in question
            movement = ego_data[ego_data['Frame'].isin(time_range)]
            # Read speed and heading and get time step wise difference
            agent_heading = movement['Heading'].values
            agent_speed = movement['Speed'].values
            heading_diff = 180 - abs(abs(agent_heading[1:] - agent_heading[:-1]) - 180)
            speed_diff = agent_speed[1:] - agent_speed[:-1]
            mean_acceleration = speed_diff.mean()
            max_acceleration = max(abs(speed_diff))
            mean_heading = heading_diff.mean()
            max_heading = max(abs(heading_diff))
            # Append to information vectors
            delta_heading_mean.append(mean_heading)
            delta_heading_max.append(max_heading)

            delta_acceleration_mean.append(mean_acceleration)
            delta_acceleration_max.append(max_acceleration)

        return delta_heading_mean, delta_heading_max, delta_acceleration_mean, delta_acceleration_max

    def process_file_sequential(self):
        data_length = len(self.trajectories_files)
        for i in tqdm(range(0, data_length)):
            self.process_file(self.trajectories_files[i])

    def get_histogram(self, data, metric,
                      title=None,  color='#773344', x_label='Value',
                      y_label='Frequency', limits=(None, None, None)):
        if title == None:
            title = metric
        num_differences = len(data)
        q_lower = np.quantile(data, 0.05)
        q_upper = np.quantile(data, 0.95)
        data = data[(data >= q_lower) & (data <= q_upper)]

        fig, histogram_plot = plt.subplots()
        freq, bins, patches = histogram_plot.hist(
            data, bins=80, color=color, edgecolor="k", linewidth=0.2, alpha=1)
        histogram_plot.set_title(f'{title} {self.airport}')
        histogram_plot.set_xlabel(x_label)
        histogram_plot.set_ylabel(y_label)

        min_lim, max_lim, step = limits
        if (min_lim == None):
            min_lim, max_lim = math.floor(min(bins)), math.floor(max(bins))
            plt.xlim([min_lim, max_lim])
        else:
            xticks = range(math.floor(min_lim), math.ceil(max_lim), step)
            plt.xticks(xticks)
        plt.grid()
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(f'{self.out_dir}/{metric}_{self.airport}.png', dpi=800)
        plt.close()

    def process_file_parrallel(self):
        pool = multiprocessing.Pool(processes=20)
        result = pool.map(self.process_file, self.trajectories_files)
        pool.close()
        pool.join()
        mean_heading, max_heading, mean_acceleration, max_acceleration = zip(*result)
        mean_heading = np.concatenate(mean_heading, axis=0)
        max_heading = np.concatenate(max_heading, axis=0)
        mean_acceleration = np.concatenate(mean_acceleration, axis=0)
        max_acceleration = np.concatenate(max_acceleration, axis=0)
        # Generate Heading Stats
        self.get_histogram(mean_heading, metric='mean_heading', title='Mean Heading Change$',
                           x_label='Heading Change(Degrees)', limits=(0, 10, 2))
        self.get_histogram(max_heading, metric='max_heading',
                           title='Max Heading Change$', x_label='Heading Change(Degrees)')
        # Generate Speed Stats
        self.get_histogram(mean_acceleration, metric='mean_acceleration', title='Mean Acceleration')
        self.get_histogram(max_acceleration, metric='max_acceleration', title='Max Acceleration')

    def get_statistics(self):
        if (self.parrallel):
            self.process_file_parrallel()
        else:
            self.process_file_sequential()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument(
        '--base_dir', default='./datasets/amelia', type=str, help='Input path.')
    parser.add_argument(
        '--traj_version', default='a10v08', type=str, help='Trajectory version.')
    parser.add_argument('--output_dir', default='./output/movement', type=str, help='Output path.')
    parser.add_argument('--airport', default='all', type=str, help='Airport to process.')
    parser.add_argument('--parrallel', default=True, type=str, help='Enable parrallel computing')
    args = parser.parse_args()

    if args.airport == 'all':
        for airport in AIRPORT_COLORMAP.keys():
            args.airport = airport
            processor = TrajectoryProcessor(**vars(args))
            processor.get_statistics()
    else:

        processor = TrajectoryProcessor(**vars(args))
        processor.get_statistics()
