import os
import math
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from easydict import EasyDict
from amelia_scenes.scene_processing.src.full_processor import SceneProcessor

# Configure plotting settings.

class DatasetProcessor():
    def __init__(self, ipath: str, opath: str, airport: str, parrallel: bool, 
                 seq_len: int = 60, skip: int = 1, dim:int = 11):
        # Processor Config
        self.seq_len = seq_len
        self.skip = skip
        self.dim = dim
        self.BASE_DIR = ipath
        self.OUT_DIR =  opath
        self.AIRPORT = airport
        self.parrallel = parrallel
        self.TRAJECTORIES_DIR = os.path.join(self.BASE_DIR, 'raw_trajectories', airport)
        print('---- Analyzing data in ', self.TRAJECTORIES_DIR, '----')
        self.TRAJECTORY_FILES = [f for f in os.listdir(self.TRAJECTORIES_DIR)]
        os.makedirs(self.OUT_DIR, exist_ok=True)
        # Make out directory
        self.min_agents = 1
        self.max_agents = None
        self.metrics = {'num_agents': {},
                        'stationary_agents': {}, 
                        'movement': {
                            'heading': {},
                            'aceleration': {} }}
        self.aircraft_only = False
        os.makedirs(self.OUT_DIR, exist_ok=True)

         # TODO: provide configs as YAML files
        config = EasyDict({
            "airport": self.AIRPORT,
            "in_data_dir": os.path.join(self.BASE_DIR, 'raw_trajectories'),
            "out_data_dir": "",   
            'graph_data_dir': "../datasets/amelia/graph_data_a10v01os",
            "assets_dir": "../datasets/amelia/assets",
            "parallel": "",   
            "perc_process": 1.0,
            'overwrite': "",
            "pred_lens": [20, 50],
            "hist_len": 10,
            "skip": 1,
            "min_agents": 2, 
            "max_agents": 30,
            "min_valid_points": 2,
            "seed": 42
        })

        self.scene_process = SceneProcessor(config)
        
    
    def process_file(self, file):
        # Metrics to report
        valid_seq = 0
        valid_scenarios = 0

        # Open data file
        in_file = os.path.join(self.TRAJECTORIES_DIR, file)
        data = pd.read_csv(in_file)
        if(self.aircraft_only):
            data = data[:][data['Type'] == 0] 
        # Get the number of unique frames
        frames = data.Frame.unique().tolist()
        frame_data = []
        for frame_num in frames:
            frame = data[:][data.Frame == frame_num] 
            frame_data.append(frame)

        num_sequences = int(math.ceil((len(frames) - (self.seq_len) + 1) / self.skip))

        if num_sequences < 1:
            return 0, 0
        
        for i in range(0, num_sequences * self.skip + 1, self.skip):
            seq, agent_id, agent_type, agent_valid, agent_mask = self.scene_process.process_seq(
                frame_data=frame_data, frames=frames, seq_idx=i, airport_id=self.AIRPORT)
            
            if seq is None:
                continue

            num_agents, _, _ = seq.shape 
            valid_seq += num_agents
            valid_scenarios+=1

        return valid_scenarios, valid_seq
                        
    def process_dataset(self):
        if self.parrallel:  
            print(f"Processing files, in parallel...") 
            metrics = Parallel(n_jobs=-1)(delayed(self.process_file)(f) for f in tqdm(self.TRAJECTORY_FILES))
            valid_scenarios = 0
            valid_seq = 0
            none_moving_agents = []
            for i in range(len(metrics)):
                res = metrics.pop() 
                # Unpack results
                if res is not None:
                    valid_scenarios += res[0]
                    valid_seq += res[1]
            del metrics
            

        else:
            print(f"Processing files, sequentially...") 
            num_agents_total = []
            valid_scenarios = 0
            valid_seq = 0
            heading_change = []
            # speed_change = []
            none_moving_agents = []
            for f in tqdm(self.TRAJECTORY_FILES):
                metrics =  self.process_file(f)
                if metrics is not None:
                    (num_agents, scene_count, seq_count, 
                        heading_diff, stationary_agents_agents) = self.process_file(f)
                    num_agents_total.extend(num_agents)
                    valid_scenarios += scene_count
                    valid_seq += seq_count
                    heading_change.extend(heading_diff)
                    # speed_change.extend(speed_diff)
                    none_moving_agents.extend(stationary_agents_agents)
            del metrics
 
        self.metrics['scenario_count'] = valid_scenarios
        self.metrics['sequence_count'] = valid_seq
    

    def save(self, print_summary = True, tag = None):
        if(print_summary):
            print('---------------Summary-----------------')
            print(f'Airport {self.AIRPORT}')
            print(f'Scenerio Count {self.metrics["scenario_count"]}')
            print(f'Sequence Count {self.metrics["sequence_count"]}')
            print('---------------------------------------')
        # Save Metrics
        out_path = os.path.join(self.OUT_DIR, f"{self.AIRPORT}_stats.pkl")
        if tag is not None: out_path = os.path.join(self.OUT_DIR, f"{self.AIRPORT}_stats_{tag}.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(self.metrics, f)

    def get_histogram(self, data, metric , 
                      title = None,  color = '#773344', x_label = 'Value' , 
                      y_label = 'Frequency', limits = (None, None, None)):
        if title == None: title = metric
        num_differences = len(data)
        q_lower = np.quantile(data, 0.05)
        q_upper = np.quantile(data, 0.95)
        data = data[(data >= q_lower) & (data <= q_upper)]

        fig, histogram_plot = plt.subplots()
        freq, bins, patches = histogram_plot.hist(data, bins = 80, color= color, edgecolor = "k", linewidth=0.2 , alpha=1)
        histogram_plot.set_title(f'{title} {self.AIRPORT}')
        histogram_plot.set_xlabel(x_label)
        histogram_plot.set_ylabel(y_label)
        
        min_lim, max_lim, step = limits
        if(min_lim == None): 
            min_lim, max_lim = math.floor(min(bins)),math.floor(max(bins))
            plt.xlim([min_lim, max_lim])
        else: 
            xticks = range(math.floor(min_lim),math.ceil(max_lim),step)
            plt.xticks(xticks)
        plt.grid()
        plt.tight_layout()
        plt.show(block = False)
        plt.savefig(f'{self.OUT_DIR}/{metric}_{self.AIRPORT}.png',dpi=800)
        plt.close()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--ipath', default='../datasets/amelia/traj_data_a10v08', type=str, help='Input path.')
    parser.add_argument('--opath', default='../out/cache', type=str, help='Output path.')
    parser.add_argument('--airport', default='ksea', type=str, help='Airport to process.')
    parser.add_argument('--parrallel', action="store_true", help='Enable parrallel computing')
    args = parser.parse_args()
    
    processor = DatasetProcessor(**vars(args))
    processor.process_dataset()
    processor.save(tag = 'aircraft_only')