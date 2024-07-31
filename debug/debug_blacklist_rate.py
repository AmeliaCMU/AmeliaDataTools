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

def run(traj_data_dir: str, split_type: str):
    
    for airport in AIRPORTS:
        airport_dir = os.path.join(traj_data_dir, airport)
        num_total_files = len(os.listdir(airport_dir))

        blacklist_file = os.path.join(traj_data_dir, 'blacklist', f'{airport}_{split_type}.txt')
        with open(blacklist_file, 'r') as f:
            blacklist = [os.path.join(traj_data_dir, b.rstrip()) for b in f]
        num_blacklist_files = len(blacklist)
        perc = round(100 * num_blacklist_files/num_total_files, 2)
        output_message = f"Airport {airport}:\n" + \
            f"\tTotal files: {num_total_files}\n" + \
            f"\tBlacklist files: {num_blacklist_files} ({perc}%)"
        print(output_message)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--traj_data_dir', default='../datasets/amelia/traj_data_a10v07/raw_trajectories', type=str)
    parser.add_argument("--split_type", type=str, default='day', choices=['random', 'day', 'month'])
    args = parser.parse_args()

    run(**vars(args))