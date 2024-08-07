import json
import math
import os
import pandas as pd
import yaml

from easydict import EasyDict
from tqdm import tqdm


from amelia_datatools.utils.common import AIRPORT_COLORMAP, ASSET_DATA_DIR, TRAJ_DATA_DIR


class RunningStats:
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def std(self):
        return math.sqrt(self.variance())


def run_limits():
    black_list = ['Frame', 'ID', 'Type', 'Interp']
    for airport in AIRPORT_COLORMAP.keys():
        print(f"Running: {airport.upper()}")

        traj_data_dir = os.path.join(TRAJ_DATA_DIR, 'raw_trajectories', f'{airport}')
        asset_dir = os.path.join(ASSET_DATA_DIR, f'{airport}')

        limits_file = os.path.join(asset_dir, 'limits.json')
        with open(limits_file, 'r') as f:
            ref_data = EasyDict(json.load(f))

        print(json.dumps(ref_data.limits.Altitude, indent=4))

        traj_files = [os.path.join(traj_data_dir, f) for f in os.listdir(traj_data_dir)]
        print(f"\tFound {len(traj_files)} trajectory files in {traj_data_dir}")

        data = pd.read_csv(traj_files[0])

        limits = {}
        incstats = {}
        for k, v in data.items():
            if k in black_list:
                continue
            limits[k] = {
                "min": float('inf'), "max": -float('inf'), "mean": 0.0, "std": 0.0
            }
            incstats[k] = RunningStats()

        for f in tqdm(traj_files):
            data = pd.read_csv(traj_files[0])
            for k in limits.keys():
                arr = data[k].to_numpy()
                limits[k]["min"] = min(limits[k]["min"], arr.min())
                limits[k]["max"] = max(limits[k]["max"], arr.max())

                for a in arr:
                    incstats[k].push(a)

            for k in limits.keys():
                limits[k]["mean"] = incstats[k].mean()
                limits[k]["std"] = incstats[k].std()

        ref_data['limits'] = limits
        limits_file = os.path.splitext(limits_file)[0] + '.yaml'
        with open(limits_file, 'w') as f:
            yaml.dump(ref_data, f)
        print(f"\tAdding limits to reference file in: {limits_file}")


if __name__ == "__main__":
    run_limits()
