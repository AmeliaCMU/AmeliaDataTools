import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from pyproj import Transformer
from easydict import EasyDict
from tqdm import tqdm
import glob

from amelia_datatools.utils import common as C
from amelia_datatools.utils import utils
from amelia_datatools.utils.processor_utils import transform_extent


class TrajectoryProcessor():
    def __init__(
            self, airport: str, base_dir: str, traj_version: str, to_process: float, dpi: int, drop_interp: bool, output_dir: str):
        self.base_dir = base_dir

        self.out_dir = os.path.join(output_dir, utils.get_file_name(__file__))

        self.airport = airport
        self.dpi = dpi
        self.drop_interp = drop_interp
        assets_dir = os.path.join(self.base_dir, 'assets', self.airport)
        limits_file = os.path.join(assets_dir, 'limits.json')

        map_filepath = os.path.join(assets_dir, 'bkg_map.png')
        self.semantic_map = mpimg.imread(map_filepath)

        with open(limits_file, 'r') as fp:
            reference_data = EasyDict(json.load(fp))

        self.origin_crs = 'EPSG:4326'
        self.target_crs = 'EPSG:3857'

        crs = reference_data.espg_4326
        self.limits_espg_4326 = (crs.north, crs.east, crs.south, crs.west)
        self.limits_espg_3857 = transform_extent(
            self.limits_espg_4326, self.origin_crs, self.target_crs)

        self.reference_point = (reference_data.ref_lat,
                                reference_data.ref_lon, reference_data.range_scale)

        trajectories_dir = os.path.join(
            self.base_dir, f'traj_data_{traj_version}', 'raw_trajectories', self.airport)
        print(f"Analyzing data in: {trajectories_dir}")
        self.trajectory_files = [os.path.join(trajectories_dir, f)
                                 for f in os.listdir(trajectories_dir)]
        random.shuffle(self.trajectory_files)
        self.trajectory_files = self.trajectory_files[:int(len(self.trajectory_files) * to_process)]

        print(f"Plotting {to_process*100}% of data ({len(self.trajectory_files)}))")
        os.makedirs(self.out_dir, exist_ok=True)

    def plot_trajectories(self, limit_plot=True):
        north, east, south, west = self.limits_espg_4326
        projector = Transformer.from_crs(self.origin_crs, self.target_crs)
        fig, map_plot = plt.subplots(1)
        map_plot.imshow(self.semantic_map, zorder=0, extent=[west, east, south, north], alpha=1.0)

        perc_points = []
        high_interp_files = []
        for trajectory_file in tqdm(self.trajectory_files):
            data = pd.read_csv(trajectory_file)
            N = data.shape[0]
            if self.drop_interp:
                data = data[data.Interp == '[ORG]']
                n = data.shape[0]
                perc = 100 * (N-n)/N
                if perc > 50:
                    high_interp_files.append(trajectory_file.split('/')[-1])
                # print(f"Dropped {N-n} points ({round(perc, 3)}%)")
                perc_points.append(perc)

            data.dropna(subset=['Lat', 'Lon'])
            data = data[:][data['Type'] == 0]
            lon, lat = np.array(data.Lon.values), np.array(data.Lat.values)

            if self.airport in C.AIRPORT_COLORMAP.keys():
                color = C.AIRPORT_COLORMAP[self.airport]
            else:
                color = np.random.rand(3,)

            map_plot.scatter(lon, lat, c=color, s=1.50, alpha=0.40, marker=".", edgecolors="none")

        if len(perc_points):
            perc_points = np.asarray(perc_points)
            print(
                f"Dropped points %: mean: {perc_points.mean()} std: {perc_points.std()} min: {perc_points.min()} max: {perc_points.max()}")

        if len(high_interp_files):
            with open(f'{self.out_dir}/{self.airport}_interp_gt50.txt', 'w') as f:
                f.write("\n".join(high_interp_files))

        for spine in map_plot.spines.values():
            spine.set_edgecolor('white')

        if limit_plot:
            map_plot.axis([west, east, south, north])

        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        print('Saving...')
        suffix = '' if self.drop_interp else '_interp'
        plt.savefig(f'{self.out_dir}/{self.airport}{suffix}.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def heat_map(self):
        # # fig, map_plot = plt.subplots(1,figsize=(30,30))
        # map_plot.imshow(semantic_map, zorder=0, extent=[west, east, south, north], alpha=1.0)
        # Plot all XY
        Lon, Lat = [], []
        for trajectory_file in tqdm(self.trajectory_files):
            data = pd.read_csv(trajectory_file)
            data.dropna(subset=['Lat', 'Lon'])
            data = data[:][data['Type'] == 0]
            lon, lat = np.array(data.Lon.values), np.array(data.Lat.values)
            Lon.append(lon)
            Lat.append(lat)
        Lon = np.concatenate(Lon)
        Lat = np.concatenate(Lat)

        freq, xedges, yedges = np.histogram2d(x=lon, y=lat, bins=10000, density=False)
        freq = np.clip(freq, a_min=0, a_max=5)
        freq = freq.T
        x, y = np.meshgrid(xedges, yedges)
        plt.pcolormesh(x, y, freq)
        plt.axis('equal')
        print('Saving...')
        plt.savefig(f'{self.out_dir}/{self.airport}_heatmap.png', dpi=self.dpi, bbox_inches='tight')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--base_dir', default=C.DATA_DIR, type=str, help='Input path')
    parser.add_argument('--traj_version', type=str, default=C.VERSION)
    parser.add_argument('--to_process', default=1.0, type=float)
    parser.add_argument('--drop_interp', action='store_true')
    parser.add_argument('--dpi', type=int, default=C.DPI)
    parser.add_argument('--output_dir', type=str, default=C.VIS_DIR)
    args = parser.parse_args()

    airport_list = utils.get_airport_list()

    for airport in tqdm(airport_list):
        args.airport = airport
        processor = TrajectoryProcessor(**vars(args))
        processor.plot_trajectories()
