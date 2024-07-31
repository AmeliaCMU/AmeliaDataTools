import os
import sys
import numpy
import pickle
from tqdm import tqdm
DIR  = f"../out/cache"

sys.path.insert(1, '../utils/')
from common import *

if __name__  == "__main__":
    scenes = 0
    sequences = 0
    tokens = 0

    for airport in AIRPORT_COLORMAP.keys():
        crowdedness_file = os.path.join(DIR, f'{airport}_stats_aircraft_only.pkl')
        with open(crowdedness_file, 'rb') as f:
            x = pickle.load(f)
        
        scenes+= x['scenario_count']
        sequences+= x['sequence_count']
        tokens += x['sequence_count'] * 60

    print("----------------------------")
    print("Scene", scenes)
    print("Sequences", sequences)
    print("Tokens", tokens)
