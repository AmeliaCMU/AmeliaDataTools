import math
import os 
import pandas as pd
import sys 
import numpy as np

sys.path.insert(1, '../utils/')
from common import *
from enum import Enum

from tqdm import tqdm
import matplotlib.pyplot as plt
AIRPORTS = AIRPORT_COLORMAP.keys()

# ------------------
# Configuration 
SEQ_LEN = 60
SKIP = 1
MIN_VALID_POINTS = 2
HIST_LEN = 10
MIN_AGENTS = 2
MAX_AGENTS = 15
PRED_LENS = [20, 50]
# ------------------

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
# Mask removes 'Frame', 'ID' and 'AgentType'
# Altitude, Speed, Heading, Lat, Lon, Range, Bearing, Type, Interp, x, y  
RAW_SEQ_MASK = [False, False, True, True, True, True, True, True, True, False, False, True, True]

# New index order after mask
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

# Swapped order to go...
# From: Altitude, Speed, Heading, Lat, Lon, Range, Bearing, x, y
#   To: Speed, Heading, Lat, Lon, Range, Bearing, x, y, z
# SEQ_ORDER = [1, 2, 3, 4, 5, 6, 7, 8, 0]
SEQ_ORDER = [
    RAW_SEQ_IDX.Speed, 
    RAW_SEQ_IDX.Heading, 
    RAW_SEQ_IDX.Lat, 
    RAW_SEQ_IDX.Lon, 
    RAW_SEQ_IDX.Range,
    RAW_SEQ_IDX.Bearing, 
    RAW_SEQ_IDX.x, 
    RAW_SEQ_IDX.y, 
    RAW_SEQ_IDX.Altitude
]
# Final index order after post-processing: 
#   Speed, Heading, Lat, Lon, Range, Bearing, Interp, x, y, z (Previously Altitude)
SEQ_IDX = EasyDict({
    'Speed':   0, 
    'Heading': 1, 
    'Lat':     2, 
    'Lon':     3, 
    'Range':   4, 
    'Bearing': 5, 
    'x':       6, 
    'y':       7, 
    'z':       8, 
})

DIM = len(RAW_SEQ_IDX.keys())

# Agent types
AGENT_TYPES = {'Aircraft': 0, 'Vehicle': 1, 'Unknown': 2}

# TODO: what's this?
MAP_IDX = [False, False, True, True, False, False, True, True, True, False]

# A bit overkill, but it's to avoid indexing errors. 
# Traj Masks
LL = np.zeros(shape=len(SEQ_ORDER)).astype(bool)
LL[SEQ_IDX.Lat] = LL[SEQ_IDX.Lon] = True

HLL = np.zeros(shape=len(SEQ_ORDER)).astype(bool)
HLL[SEQ_IDX.Lat] = HLL[SEQ_IDX.Lon] = HLL[SEQ_IDX.Heading] = True

HD = np.zeros(shape=len(SEQ_ORDER)).astype(bool)
HD[SEQ_IDX.Heading] = True

XY = np.zeros(shape=len(SEQ_ORDER)).astype(bool)
XY[SEQ_IDX.x] = XY[SEQ_IDX.y] = True

XYZ = np.zeros(shape=len(SEQ_ORDER)).astype(bool)
XYZ[SEQ_IDX.x] = XYZ[SEQ_IDX.y] = XYZ[SEQ_IDX.z] = True

# REL_XY  = [ True,  True, False, False]
# REL_XY  = [ True,  True, False]
# REL_XYZ = [ True,  True,  True, False]
# REL_HD  = [False, False, False,  True]

# REL_XY  = [ True,  True, False, False]
REL_XY  = [ True,  True, False]
REL_XYZ = [ True,  True,  True, False, False]
REL_HD  = [False, False, False,  True, False]

OUT_DIR = os.path.join(VIS_DIR, __file__.split('/')[-1].split(".")[0])
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Created output directory in: {OUT_DIR}")

class FileErrorType(Enum):
    ALL_OK = 0
    NOT_ENOUGH_SEQUENCES = 1
    INVALID_SEQUENCES = 2

class SeqErrorType(Enum):
    ALL_OK = 0
    MIN_AGENTS = 1
    MAX_AGENTS = 2
    NO_VALID_AGENTS = 3
    NOT_ENOUGH_PROCESSED_AGENTS = 4

def process_file(f: str, airport_id: str):  
    # Otherwise, shard the file and add it to the scenario list. 
    data = pd.read_csv(f)

    # Get the number of unique frames
    frames = data.Frame.unique().tolist()
    frame_data = []
    for frame_num in frames:
        frame = data[:][data.Frame == frame_num] 
        frame_data.append(frame)

    error_list = []
    num_sequences = int(math.ceil((len(frames) - (SEQ_LEN) + 1) / SKIP))
    if num_sequences < 1:
        return FileErrorType.NOT_ENOUGH_SEQUENCES, error_list

    for i in range(0, num_sequences * SKIP + 1, SKIP):
        error = process_seq(frame_data=frame_data, frames=frames, seq_idx=i, airport_id=airport_id)
        error_list.append(error)
    
    if all(error == SeqErrorType.ALL_OK for error in error_list):
        return FileErrorType.ALL_OK, error_list
    return FileErrorType.INVALID_SEQUENCES, error_list

def process_seq(frame_data: pd.DataFrame, frames: list, seq_idx: int, airport_id: str) -> np.array:
    # All data for the current sequence: from the curr index i to i + sequence length
    seq_data = np.concatenate(frame_data[seq_idx:seq_idx + SEQ_LEN], axis=0)

    # IDs of agents in the current sequence
    unique_agents = np.unique(seq_data[:, RAW_IDX.ID])
    num_agents = len(unique_agents)
    if num_agents < MIN_AGENTS:
        return SeqErrorType.MIN_AGENTS
    
    if num_agents > MAX_AGENTS:
        return SeqErrorType.MAX_AGENTS

    valid_agent_list = []
    num_agents_considered = 0

    for _, agent_id in enumerate(unique_agents):
        # Current sequence of agent with agent_id
        agent_seq = seq_data[seq_data[:, 1] == agent_id]

        # Start frame for the current sequence of the current agent reported to 0
        pad_front = frames.index(agent_seq[0, 0]) - seq_idx
        
        # End frame for the current sequence of the current agent: end of current agent path in 
        # the current sequence. It can be sequence length if the aircraft appears in all frames
        # of the sequence or less if it disappears earlier.
        pad_end = frames.index(agent_seq[-1, 0]) - seq_idx + 1
        
        # Exclude trajectories less then seq_len
        if pad_end - pad_front != SEQ_LEN:
            continue
    
        # Interpolated mask
        mask = agent_seq[:, RAW_IDX.Interp] == '[ORG]'
        agent_seq[mask, RAW_IDX.Interp]  = 1.0 # Not interpolated -->     Valid
        agent_seq[~mask, RAW_IDX.Interp] = 0.0 #     Interpolated --> Not valid
        
        valid = mask[:HIST_LEN].sum() >= MIN_VALID_POINTS
        if valid:
            for t in PRED_LENS:
                if mask[HIST_LEN:HIST_LEN+t].sum() < MIN_VALID_POINTS:
                    valid = False
                    break
        valid_agent_list.append(valid)
        
        num_agents_considered += 1
    
    # Return Nones if there aren't any valid agents
    valid_agent_list = np.asarray(valid_agent_list)
    if valid_agent_list.sum() == 0:
        return SeqErrorType.NO_VALID_AGENTS

    # Return Nones if the number of considered agents is less than the required
    if num_agents_considered < MIN_AGENTS:
        return SeqErrorType.NOT_ENOUGH_PROCESSED_AGENTS
    
    return SeqErrorType.ALL_OK

def run(traj_data_dir: str, blacklist_file: str, airport: str, num_files: int):
    
    with open(blacklist_file, 'r') as f:
        blacklist = [os.path.join(traj_data_dir, b.rstrip()) for b in f]
    
    blacklist = list(set([b for b in blacklist if airport in b]))
    if num_files == -1:
        num_files = len(blacklist)

    file_error_list, seq_error_list = [], []
    n = 0
    for f in tqdm(blacklist):
        if airport not in f:
            continue
        if n >= num_files:
            break
        ferror, serror = process_file(f, airport)
        file_error_list.append(ferror)
        seq_error_list += serror
        n += 1
    
    file_error_list = np.asarray([ferror.value for ferror in file_error_list])
    labels, counts = np.unique(file_error_list, return_counts=True)
    plt.bar(labels, counts, align='center')
    plt.title('File Error Types')
    plt.gca().set_xticks(labels)
    plt.savefig(f"{OUT_DIR}/{airport}_file_errors.png", dpi=300)
    plt.close()

    seq_error_list = np.asarray([serror.value for serror in seq_error_list])
    labels, counts = np.unique(seq_error_list, return_counts=True)
    plt.bar(labels, counts, align='center')
    plt.gca().set_xticks(labels)
    plt.title('Sequence Error Types')
    plt.savefig(f"{OUT_DIR}/{airport}_sequence_errors.png", dpi=300)
    plt.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--traj_data_dir', default='../datasets/amelia/traj_data_a10v7/raw_trajectories', type=str)
    parser.add_argument(
        "--blacklist_file", default="blacklist.txt")
    parser.add_argument(
        "--airport", default="panc", choices=AIRPORTS)
    parser.add_argument(
        "--num_files", type=int, default=-1)
    args = parser.parse_args()

    run(**vars(args))