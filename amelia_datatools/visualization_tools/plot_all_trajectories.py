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
import math

from amelia_datatools.utils import common as C
import amelia_datatools.utils.utils as U
from amelia_datatools.utils.processor_utils import transform_extent
from matplotlib.colors import LogNorm
from matplotlib.colors  import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker  import LogLocator, LogFormatter


class TrajectoryProcessor():
    def __init__(
            self, airport: str, base_dir: str, traj_version: str, to_process: float, dpi: int, drop_interp: bool, output_dir: str, **kwargs):
        self.base_dir = base_dir

        self.out_dir = os.path.join(output_dir, U.get_file_name(__file__))

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
                     for f in os.listdir(trajectories_dir) if f.endswith('.csv')]
        random.shuffle(self.trajectory_files)
        self.trajectory_files = self.trajectory_files[:int(len(self.trajectory_files) * to_process)]

        print(f"Plotting {to_process*100}% of data ({len(self.trajectory_files)}))")
        os.makedirs(self.out_dir, exist_ok=True)
      
      

    def bbox_size_meters(self, north, east, south, west):
        """
        Returns (width_m, height_m) for a lat/lon bounding box.
        north, east, south, west follow the usual EPSG:4326 order (degrees).
        """
        from geographiclib.geodesic import Geodesic
        g = Geodesic.WGS84  # ellipsoidal Earth model

        # Horizontal span (west → east) measured along the northern edge
        width = g.Inverse(north, west, north, east)["s12"]

        # Vertical span (south → north) measured along the western edge
        height = g.Inverse(south, west, north, west)["s12"]

        return width, height

    def plot_traj_heatmap(
            self,
            limit_plot: bool = True,
            heatmap: bool = True,
            aircraft_only: bool = True,
            area_per_bins: int = 9500):
        
        north, east, south, west = self.limits_espg_4326
        extent_size = self.bbox_size_meters(north, east, south, west)
        area = extent_size[0] * extent_size[1]
        print(f"Area of the {self.airport} bounding box: {area / 1e6:.2f} km²")
        
        # Calculate the number of bins based on the area
        bins = area // area_per_bins  # 50 m² per bin
        bins = int(bins)  # Ensure bins is an integer
        print(f"Number of bins: {bins}")
        print(f"Area per bin: {area / bins:.2f} m²")
        
        fig, ax = plt.subplots(1, figsize=(8, 6))

        # ------------------------------------------------------------------
        # Background semantic map
        # ------------------------------------------------------------------
        ax.imshow(
            self.semantic_map,
            zorder=0,
            extent=[west, east, south, north],
            alpha=1.0,
        )

        # ------------------------------------------------------------------
        # Parse every trajectory, optionally drop interpolated points
        # ------------------------------------------------------------------
        lon_all, lat_all = [], []
        perc_points, high_interp_files = [], []
        if aircraft_only:  
            print("\033[93mShowing only aircraft\033[0m")
            
        for trajectory_file in tqdm(self.trajectory_files, desc="Reading trajectories"):
            data = pd.read_csv(trajectory_file)
            if aircraft_only:  
                clean_data = data[:][data['Type'] == 0]
                
            N = len(data)
            if N == 0:
                # File is all unknown agents, plot them
                clean_data = data
            data = clean_data

            if self.drop_interp:
                data = data[data["Interp"] == "[ORG]"]
                n = len(data)
                perc = 100 * (N - n) / N
                perc_points.append(perc)
                if perc > 50:
                    high_interp_files.append(trajectory_file.rsplit("/", 1)[-1])
            

            # keep only valid points
            
            data = data.dropna(subset=["Lat", "Lon"])
            lon_all.extend(data["Lon"].values)
            lat_all.extend(data["Lat"].values)

            # ──────────────────────────────────────────────────────────────
            # quick scatter preview (very light) when NOT using heat-map
            # ──────────────────────────────────────────────────────────────
            if not heatmap:
                color = (
                    C.AIRPORT_COLORMAP.get(self.airport, np.random.rand(3,))  # type: ignore[attr-defined]
                )
                ax.scatter(
                    data["Lon"],
                    data["Lat"],
                    s=1.0,
                    c=[color],
                    alpha=0.15,
                    marker=".",
                    edgecolors="none",
                    zorder=1,
                )

        # ------------------------------------------------------------------
        # Density heat-map
        # ------------------------------------------------------------------
        if heatmap and lon_all:
            # 2-D histogram of visits
            H, xedges, yedges = np.histogram2d(
                lon_all,
                lat_all,
                bins=bins,
                range=[[west, east], [south, north]],
            )

            x_idx = np.digitize(lon_all, xedges)            # 0-based indices
            y_idx = np.digitize(lat_all, yedges) 
            x_idx = np.clip(x_idx, 0, bins - 1)
            y_idx = np.clip(y_idx, 0, bins - 1)
            counts = H[x_idx, y_idx]       
            vmin, vmax = 1, counts.max()
            norm = LogNorm(vmin=vmin, vmax=vmax)
            cax = inset_axes(
                ax,
                width="2.5%",          # bar ≈2.5 % of the map’s width
                height="100%",         # full height of the map
                loc="lower left",
                bbox_to_anchor=(1.0, 0., 1, 1),   # (x0, y0, width, height) in Axes coords
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
            
            sc = ax.scatter(
                lon_all,
                lat_all,
                c=counts,
                cmap="afmhot",
                norm=norm,
                s=2.0,
                alpha=0.9,
                marker=".",
                edgecolors="none",
                zorder=1,
            )
        
            cbar = fig.colorbar(sc, cax=cax)
            # cbar.outline.set_visible(False) 
            # cbar.set_ticks(ticks)
            cbar.ax.tick_params(labelsize=6)
            # cbar.set_label("Frequency")

        # ------------------------------------------------------------------
        # Diagnostics on dropped interpolation points
        # ------------------------------------------------------------------
        if perc_points:
            pp = np.asarray(perc_points)
            print(
                f"Dropped points % – mean: {pp.mean():.2f}, "
                f"std: {pp.std():.2f}, min: {pp.min():.2f}, max: {pp.max():.2f}"
            )
        if high_interp_files:
            out_file = f"{self.out_dir}/{self.airport}_interp_gt50.txt"
            with open(out_file, "w") as fh:
                fh.write("\n".join(high_interp_files))

        # ------------------------------------------------------------------
        # Cosmetics
        # ------------------------------------------------------------------
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        if limit_plot:
            ax.set_xlim(west, east)
            ax.set_ylim(south, north)

        ax.set_xticks([])
        ax.set_yticks([])

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        suffix = ""
        if self.drop_interp:
            suffix = "_interp"
        if aircraft_only:
            suffix += "_aircraft_only"
        if heatmap:
            suffix += "_heatmap"
        
        out_path = f"{self.out_dir}/{self.airport}{suffix}.png"
        print(f"Saving → {out_path}")
        plt.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
    

    def plot_trajectories(self, limit_plot=True, heatmap=True):
        north, east, south, west = self.limits_espg_4326
        projector = Transformer.from_crs(self.origin_crs, self.target_crs)
        fig, map_plot = plt.subplots(1)
        map_plot.imshow(self.semantic_map, zorder=0, extent=[west, east, south, north], alpha=1.0)

        perc_points = []
        high_interp_files = []
        if heatmap:
            self.plot_traj_heatmap()
            return
        
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
            # data = data[:][data['Type'] == 0]
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

if __name__ == '__main__':
    from argparse import ArgumentParser
    airports = U.get_airport_list()
    parser = ArgumentParser()
    parser.add_argument('--base_dir', default=C.DATA_DIR, type=str, help='Input path')
    parser.add_argument('--airport', type=str, default='all', help='Airport to process',
                        choices=['all'] + airports)
    parser.add_argument('--traj_version', type=str, default=C.VERSION)
    parser.add_argument('--to_process', default=1.0, type=float)
    parser.add_argument('--drop_interp', action='store_true')
    parser.add_argument('--plot_all', action='store_true')
    parser.add_argument('--heatmap', action='store_true')
    parser.add_argument('--dpi', type=int, default=C.DPI)
    parser.add_argument('--output_dir', type=str, default=C.VIS_DIR)
    args = parser.parse_args()

    if args.airport != 'all':
        airports = [args.airport]
    for airport in airports:
        kargs = vars(args)
        kargs['airport'] = airport
        args.airport = airport
        processor = TrajectoryProcessor(**kargs)
        aircraft_only = not args.plot_all
        processor.plot_traj_heatmap(heatmap=args.heatmap, aircraft_only=aircraft_only)
