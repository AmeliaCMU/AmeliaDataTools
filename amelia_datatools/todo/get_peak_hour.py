import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from amelia_datatools.utils.common import AIRPORT_COLORMAP
from amelia_datatools.utils.processor_utils import get_time_from_file


class DatasetProcessor():
    def __init__(self, base_dir: str, traj_version: str, output_dir: str, airport: str, parallel: bool, seq_len: int = 60, skip: int = 1, dim: int = 11):
        self.out_dir = output_dir
        self.airport = airport
        self.parallel = parallel
        # Load Trajectories
        input_dir = os.path.join(base_dir, f'traj_data_{traj_version}', 'raw_trajectories')
        self.trajectories_dir = os.path.join(input_dir, airport)
        print('---- Analyzing data in ', self.trajectories_dir)
        self.trajectory_files = [f for f in os.listdir(self.trajectories_dir)]
        # Make out directory
        self.min_agents = 1
        self.max_agents = None
        self.metrics = {'num_agents': {},
                        'stationary_agents': {},
                        'movement': {
                            'heading': {},
                            'aceleration': {}}}
        self.aircraft_only = True
        if (self.aircraft_only):
            print("Reporting peak hours containing only aircraft & unknown")
        os.makedirs(self.out_dir, exist_ok=True)

    def process_file(self, file):
        # Metrics to report
        agents_in_scenario = []
        # Open data file
        in_file = os.path.join(self.trajectories_dir, file)
        airport_id = in_file.split('/')[-1].split('_')[0].lower()
        data = pd.read_csv(in_file)
        if (self.aircraft_only):
            data = data[:][(data['Type'] == 0) | (data['Type'] == 2)]
        utc_time = get_time_from_file(file, airport_id)
        # Get the number of unique frames
        frames = data.Frame.unique()
        for frame in frames:
            unique_IDs = data[:][data.Frame == frame].shape[0]
            agents_in_scenario.append(unique_IDs)

        return agents_in_scenario, utc_time

    def process_dataset(self):
        if self.parallel:
            print(f"Processing files, in parallel...")
            metrics = Parallel(n_jobs=-1)(delayed(self.process_file)(f)
                                          for f in tqdm(self.trajectory_files))
            # Unpacking results (num_agents, valid_scenarios, valid_seq)
            num_agents_total = []
            agent_per_hour = {}
            for i in range(0, 24):
                agent_per_hour[i] = []
            for i in range(len(metrics)):
                res = metrics.pop()
                if res is not None:
                    num_agents_total.extend(res[0])
                    agent_per_hour[res[1]].extend(res[0])

            del metrics

        else:
            print(f"Processing files, sequentially...")
            raise NotImplemented
        # Save entire pickle object
        with open(f"{self.out_dir}/{self.airport}.pkl", 'wb') as f:
            pickle.dump(agent_per_hour, f)
        # Get mean and plot bar graph
        for hour in agent_per_hour.keys():
            if len(agent_per_hour[hour]) == 0:
                agent_per_hour[hour] = [0]
            agent_per_hour[hour] = np.array(agent_per_hour[hour]).mean()
        self.get_peak_time(agent_per_hour)
        num_agents = np.array(num_agents_total)
        self.metrics['num_agents']['all'] = np.sum(num_agents)
        self.metrics['num_agents']['min'] = num_agents.min()
        self.metrics['num_agents']['mean'] = num_agents.mean()
        self.metrics['num_agents']['max'] = num_agents.max()

    def get_peak_time(self, dictionary_count):
        gmt_times = list(dictionary_count.keys())
        frequency = dictionary_count.values()
        # Plot the bar chart
        plt.bar(gmt_times, frequency, align='center',
                alpha=0.7, color=AIRPORT_COLORMAP[self.airport])
        plt.xlabel('Hour (Local Time)')
        plt.ylabel('Agent Count')
        plt.title(f'Agent Count per Hour Across Dataset (Local Time) {self.airport}')
        plt.xticks(range(min(gmt_times), max(gmt_times) + 1, 2))
        plt.savefig(f'{self.out_dir}/agent_count_{self.airport}.png', dpi=800)
        plt.show(block=False)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--base_dir', default='./datasets/amelia', type=str, help='Input path.')
    parser.add_argument('--traj_version', default='a10v08', type=str, help='Trajectory version.')
    parser.add_argument('--output_dir', default='./output/crowdedness',
                        type=str, help='Output path.')
    parser.add_argument('--airport', default='kjfk', type=str, help='Airport to process.')
    parser.add_argument('--parallel', action="store_true",
                        default=True, help='Enable parallel computing')
    args = parser.parse_args()

    processor = DatasetProcessor(**vars(args))
    processor.process_dataset()
