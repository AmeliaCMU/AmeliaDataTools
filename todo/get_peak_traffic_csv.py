import os
import math
import pickle
import numpy as np
import datetime
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from utils.processor_utils import impute, get_movement_stats, plotting_colors, get_time_from_file
from collections import Counter

class DatasetProcessor():
    def __init__(self, filepath: str, ipath: str, opath: str, parrallel: bool, 
                 seq_len: int = 30, skip: int = 1, dim:int = 9):
        self.BASE_DIR = ipath
        self.OUT_DIR =  opath
        self.parrallel = parrallel
        # Processor Config
        self.seq_len = seq_len
        self.skip = skip
        self.dim = dim
        # Load Trajectories
        self.TRAJECTORIES_DIR = os.path.join(self.BASE_DIR, 'raw_trajectories')
        self.airports = ['ksea', 'kewr', 'kmdw', 'kbos']
        print('---- Analyzing data in ', filepath, '----')
        with open(filepath, 'r') as file:
            directories = [line.strip() for line in file.readlines()]
            print(f"Processing {len(directories)} files")
        self.TRAJECTORY_FILES = directories
        # Make out directory
        self.min_agents = 1
        self.min_count  = 3600
        self.max_agents = None
        self.aircraft_only = True
        if(self.aircraft_only): print("Reporting peak hours containing only aircraft & unknown")
        self.raw_idx = {
            'Frame': 0, 'ID': 1, 'Altitude': 2, 'Speed': 3, 'Heading': 4, 'Lat': 5, 'Lon': 6, 
            'Range': 7, 'Bearing': 8, 'Type': 9, 'x': 10, 'y': 11
        }
        self.idxs = {
            'Altitude': 0, 'Speed': 1, 'Heading': 2, 'Lat': 3, 'Lon': 4, 'Range': 5, 'Bearing': 6, 
            'x': 7, 'y': 8
        }
        self.subset_size = 100
        self.peak_hours = []
        self.peak_hours.extend(list(range(6,12)))
        self.peak_hours.extend(list(range(16,20)))
        os.makedirs(self.OUT_DIR, exist_ok=True)
        
    def process_seq(self, frame_data: pd.DataFrame, frames: list, seq_idx: int) -> np.array:
        """ Processes all valid agent sequences.

        Inputs:
        -------
            frame_data[pd.DataFrame]: dataframe containing the scene's trajectory information in the 
            following format:
            <FrameID, AgentID, Altitude, Speed, Heading, Lat, Lon, Range, Bearing, AgentType, x, y>
            frames[list]: list of frames to process. 
            seq_idx[int]: current sequence index to process.  
        
        Outputs:
        --------
            seq[np.array]: numpy array containing all processed scene's sequences
            agent_id_list[list]: list with the agent IDs that were processed (TODO: confirm if these 
            are the aircraft's tail numbers?)
            agent_type_list[list]: list containing the type of agent (Aircraft = 0, Vehicle = 1, 
            Unknown=2)
        """
        seq_mask = [False, False, True, True, True, True, True, True, True, False, True, True]
        
        # All data for the current sequence: from the curr index i to i + sequence length
        seq_data = np.concatenate(frame_data[seq_idx:seq_idx + self.seq_len], axis=0)

        # IDs of agents in the current sequence
        unique_agents = np.unique(seq_data[:, 1])
        num_agents = len(unique_agents)
        num_agents_considered = 0
        seq = np.zeros((num_agents, self.seq_len, self.dim))
        agent_id_list = []
        agent_type_list = []

        if self.max_agents is not None:
            if num_agents < self.min_agents or num_agents > self.max_agents:
                return None, None, None
        
        for _, agent_id in enumerate(unique_agents):
            # Current sequence of agent with agent_id
            agent_seq = seq_data[seq_data[:, 1] == agent_id]

            # Start frame for the current sequence of the current agent reported to 0
            pad_front = frames.index(agent_seq[0, 0]) - seq_idx
            
            # End frame for the current sequence of the current agent: end of current agent 
            # path in the current sequence. It can be sequence length if the pedestrian 
            # appears in all frame of the sequence or less if it disappears earlier.
            pad_end = frames.index(agent_seq[-1, 0]) - seq_idx + 1
            
            # Exclude trajectories less then seq_len
            if pad_end - pad_front != self.seq_len:
                continue
            
            # Impute missing data using interpolation 
            agent_id_list.append(int(agent_id))
            agent_type_list.append(int(agent_seq[0, self.raw_idx['Type']]))
            
            # -----------------------------------------------------
            # TODO: debug impute
            agent_seq = impute(agent_seq, self.seq_len)[:, seq_mask]
            # -----------------------------------------------------
            
            seq[num_agents_considered, pad_front:pad_end] = agent_seq
            num_agents_considered += 1

        if num_agents_considered < self.min_agents:
            return None, None, None
            
        return seq[:num_agents_considered], agent_id_list, agent_type_list
    
    def process_file(self, file):
        # Metrics to report
        agents_in_scenario = 0
        # Open data file
        in_file = os.path.join(self.TRAJECTORIES_DIR, file)
        airport_id = in_file.split('/')[-1].split('_')[0].lower()
        utc_time = get_time_from_file(file, airport_id)
        data = pd.read_csv(in_file)
        if(self.aircraft_only): data = data[:][(data['Type'] == 0) | (data['Type'] == 2)] 
        # Get the number of unique frames
        frames = data.Frame.unique().tolist()
        frame_data = []
        for frame_num in frames:
            frame = data[:][data.Frame == frame_num] 
            frame_data.append(frame)

        num_sequences = int(math.ceil((len(frames) - (self.seq_len) + 1) / self.skip))
        if num_sequences < 1:
            return
        
        for i in range(0, num_sequences * self.skip + 1, self.skip):
            seq, agent_id, agent_types = self.process_seq(frame_data=frame_data, frames=frames, seq_idx=i)
            
            if seq is None:
                continue

            if len(seq) == 0:
                continue

            num_agents, _, _ = seq.shape
            agents_in_scenario += num_agents

        return agents_in_scenario, utc_time, file, airport_id

    def process_dataset(self):
        on_peak = {}
        off_peak = {}
        for airport in self.airports:
            on_peak[airport]  = []
            off_peak[airport] = []
        for i in tqdm(range(len(self.TRAJECTORY_FILES))):
            file = self.TRAJECTORY_FILES[i] 
            in_file = os.path.join(self.TRAJECTORIES_DIR, file)
            airport_id = in_file.split('/')[-1].split('_')[0].lower()
            utc_time = get_time_from_file(file, airport_id)
            if(utc_time in self.peak_hours):
                on_peak[airport_id].append(file)
            else:
                off_peak[airport_id].append(file)

        off_peak_subset = []
        on_peak_subset  = []

        for airport in self.airports:
            on_peak_subset.extend(on_peak[airport][:self.subset_size])
            off_peak_subset.extend(off_peak[airport][:self.subset_size])

        with open(f"test_peak_hours_subset.txt", 'w') as fp:
            fp.write('\n'.join(on_peak_subset))

        with open(f"test_off_peak_hours_subset.txt", 'w') as fp:
            fp.write('\n'.join(off_peak_subset))
        
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--ipath', default='datasets/swim', type=str, help='Input path.')
    parser.add_argument('--opath', default='./out', type=str, help='Output path.')
    parser.add_argument('--filepath', default='test.txt', type=str, help='File List.')
    parser.add_argument('--parrallel', action="store_true", help='Enable parrallel computing')
    args = parser.parse_args()
    
    processor = DatasetProcessor(**vars(args))
    processor.process_dataset()
