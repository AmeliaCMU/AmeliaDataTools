import os
import sys
import math
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from easydict import EasyDict

RAW_IDX = EasyDict({
    'Frame':    0, 
    'ID':       1, 
    'Altitude': 2, 
    'Speed':    3, 
    'Heading':  4, 
    'Lat':      5, 
    'Lon':      6, 
    'Range':    7, 
    'Bearing':  8, 
    'Type':     9, 
    'Interp':  10, 
    'x':       11, 
    'y':       12 
})

RAW_SEQ_IDX = EasyDict({
    'Altitude': 0, 
    'Speed':    1, 
    'Heading':  2, 
    'Lat':      3, 
    'Lon':      4, 
    'Range':    5, 
    'Bearing':  6,  
    'x':        7, 
    'y':        8
})

sys.path.insert(1, '../utils/')
from common import *
from processor_utils import impute, get_time_from_file

class DatasetProcessor():
    def __init__(self, ipath: str, opath: str, airport: str, par: bool, 
                 seq_len: int = 60, skip: int = 1, dim:int = 11):
        self.OUT_DIR =  opath
        self.AIRPORT = airport
        self.parrallel = par
        # Load Trajectories
        self.TRAJECTORIES_DIR = os.path.join(ipath, airport)
        print('---- Analyzing data in ', self.TRAJECTORIES_DIR, '----')
        self.TRAJECTORY_FILES = [f for f in os.listdir(self.TRAJECTORIES_DIR)]
        # Make out directory
        self.min_agents = 1
        self.max_agents = None
        self.metrics = {'num_agents': {},
                        'stationary_agents': {}, 
                        'movement': {
                            'heading': {},
                            'aceleration': {} }}
        self.aircraft_only = True
        if(self.aircraft_only): print("Reporting peak hours containing only aircraft & unknown")
        os.makedirs(self.OUT_DIR, exist_ok=True)
    
    def process_file(self, file):
        # Metrics to report
        agents_in_scenario = []
        # Open data file
        in_file = os.path.join(self.TRAJECTORIES_DIR, file)
        airport_id = in_file.split('/')[-1].split('_')[0].lower()
        data = pd.read_csv(in_file)
        if(self.aircraft_only):
            data = data[:][(data['Type'] == 0) | (data['Type'] == 2)] 
        utc_time = get_time_from_file(file, airport_id)
        # Get the number of unique frames
        frames = data.Frame.unique()
        for frame in frames:
            unique_IDs = data[:][data.Frame == frame].shape[0]
            agents_in_scenario.append(unique_IDs)

        return agents_in_scenario, utc_time
                        
    def process_dataset(self):
        if self.parrallel:  
            print(f"Processing files, in parallel...") 
            metrics = Parallel(n_jobs=-1)(delayed(self.process_file)(f) for f in tqdm(self.TRAJECTORY_FILES))
            # Unpacking results (num_agents, valid_scenarios, valid_seq)
            num_agents_total = []
            agent_per_hour = {}
            for i in range(0,24):
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
        with open(f"{self.OUT_DIR}/{self.AIRPORT}.pkl", 'wb') as f:
            pickle.dump(agent_per_hour, f)
        # Get mean and plot bar graph
        for hour in agent_per_hour.keys():
            agent_per_hour[hour] = np.array(agent_per_hour[hour]).mean()
        self.get_peak_time(agent_per_hour)
        num_agents =  np.array(num_agents_total)
        self.metrics['num_agents']['all'] = np.sum(num_agents)
        self.metrics['num_agents']['min'] = num_agents.min()
        self.metrics['num_agents']['mean'] = num_agents.mean()
        self.metrics['num_agents']['max'] = num_agents.max()

    def save(self, print_summary = True, tag = None):
        if(print_summary):
            print('---------------Summary-----------------')
            print(f'Airport {self.AIRPORT}')
            print(f'Scenerio Count {self.metrics["scenario_count"]}')
            print(f'Sequence Count {self.metrics["sequence_count"]}')
            print(f'Num_Agents')
            print(f'|-- Min:  {self.metrics["num_agents"]["min"]}')
            print(f'|-- Max:  {self.metrics["num_agents"]["max"]}')
            print(f'|-- Mean: {self.metrics["num_agents"]["mean"]}')
            print('---------------------------------------')
        # Save Metrics
        out_path = os.path.join(self.OUT_DIR, f"{self.AIRPORT}_stats.pkl")
        if tag is not None: out_path = os.path.join(self.OUT_DIR, f"{self.AIRPORT}_stats_{tag}.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(self.metrics, f)
    
    def get_peak_time(self, dictionary_count):
        gmt_times = list(dictionary_count.keys())
        frequency = dictionary_count.values()
        # Plot the bar chart
        plt.bar(gmt_times, frequency, align='center', alpha=0.7, color = AIRPORT_COLORMAP[self.AIRPORT])
        plt.xlabel('Hour (Local Time)')
        plt.ylabel('Agent Count')
        plt.title(f'Agent Count per Hour Across Dataset (Local Time) {self.AIRPORT}')
        plt.xticks(range(min(gmt_times), max(gmt_times) + 1, 2))
        plt.savefig(f'{self.OUT_DIR}/agent_count_{self.AIRPORT}.png',dpi=800)
        plt.show(block=False)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--ipath', default='../datasets/amelia/traj_data_a10v08/raw_trajectories', type=str, help='Input path.')
    parser.add_argument('--opath', default='./crowdedness', type=str, help='Output path.')
    parser.add_argument('--airport', default='ksea', type=str, help='Airport to process.')
    parser.add_argument('--par', action="store_true", help='Enable parrallel computing')
    args = parser.parse_args()
    
    processor = DatasetProcessor(**vars(args))
    processor.process_dataset()