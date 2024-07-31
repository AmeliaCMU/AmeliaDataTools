import argparse
import cv2
import imageio.v2 as imageio
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import osmnx as ox
import pickle
import glob
import random

from enum import Enum
from easydict import EasyDict
from matplotlib import cm
from matplotlib import rcParams
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy import ndimage
from typing import Tuple
from tqdm import tqdm
rcParams['font.family'] = 'monospace'

AIRCRAFT = 0
VEHICLE = 1
UNKNOWN = 2

KNOTS_TO_MPS = 0.51444445
KNOTS_TO_KPH = 1.852
HOUR_TO_SECOND = 3600
KMH_TO_MS = 1/3.6

ZOOM = {
    AIRCRAFT: 0.015, 
    VEHICLE: 0.2, 
    UNKNOWN: 0.2
}

COLOR_MAP = {
    'gt_hist': '#FF5A4C',
    'gt_future': '#8B5FBF',
    'holdline': "#0DD7EF",
    'follower': "#EF0D5C",
    'leader': "#93EF0D",
    'invalid': "#000000"
}

SEQ_IDX = {
    'Speed': 0, 
    'Heading': 1, 
    'Lat': 2, 
    'Lon': 3, 
    'Range': 4, 
    'Bearing': 5, 
    'Interp': 6,
    'x': 7, 
    'y': 8,
    'z': 9, 
}

XY = [False, False, False, False, False, False, False, True, True, False]
LL = [False, False, True, True, False, False, False, False, False, False]

MAP_IDX = {
    'LatStart': 0, 
    'LonStart': 1, 
    'xStart': 2, 
    'yStart': 3, 
    'LatEnd': 4, 
    'LonEnd': 5, 
    'xEnd': 6, 
    'yEnd': 7, 
    'SemanticID': 8, 
    'OSMID': 9,
}

RUNWAY_EXTENTS = {
    "ksea": {
        "mean": 3.02, "min": 2.65, "max": 3.63
    },
    "kbos": {
        "mean": 2.27, "min": 0.8, "max": 3.29
    },
    "kmdw": {
        "mean": 1.72, "min": 1.18, "max": 2.22
    },
    "kewr": {
        "mean": 3.17, "min": 2.31, "max": 3.76
    },

    "kdca": {
        "mean": 3.17, "min": 2.31, "max": 3.76
    },
    "kjfk": {
        "mean": 3.17, "min": 2.31, "max": 3.76
    },
    "klax": {
        "mean": 3.17, "min": 2.31, "max": 3.76
    },
    "kmsy": {
        "mean": 3.17, "min": 2.31, "max": 3.76
    },
    "ksfo": {
        "mean": 3.17, "min": 2.31, "max": 3.76
    },
    "panc": {
        "mean": 3.17, "min": 2.31, "max": 3.76
    },
}

def load_assets(base_dir: str, airport: str) -> Tuple:
    asset_dir = os.path.join(base_dir, "assets")    
    raster_dir = os.path.join(asset_dir, airport)    

    raster_map_filepath = os.path.join(raster_dir, "bkg_map.png")
    raster_map = cv2.imread(raster_map_filepath)
    raster_map = cv2.resize(raster_map, (raster_map.shape[0]//2, raster_map.shape[1]//2))
    raster_map = cv2.cvtColor(raster_map, cv2.COLOR_BGR2RGB)

    map_dir = os.path.join(base_dir, "graph_data", airport)
    pickle_map_filepath = os.path.join(map_dir, "semantic_graph.pkl")
    with open(pickle_map_filepath, 'rb') as f:
        graph_pickle = pickle.load(f)
        hold_lines = graph_pickle['hold_lines']
        graph_nx = graph_pickle['graph_networkx']
        # pickle_map = temp_dict['map_infos']['all_polylines'][:]
        
    limits_filepath = os.path.join(raster_dir, 'limits.json')
    with open(limits_filepath, 'r') as fp:
        ref_data = EasyDict(json.load(fp))
    espg = ref_data.espg_4326
    limits = (espg.north, espg.east, espg.south, espg.west)

    aircraft_filepath = os.path.join(asset_dir, "ac.png")
    aircraft = imageio.imread(aircraft_filepath)

    vehicle_filepath = os.path.join(asset_dir, "vc.png")
    vehicle = imageio.imread(vehicle_filepath)

    uk_filepath = os.path.join(asset_dir, "uk.png")
    unknown =  imageio.imread(uk_filepath)
    
    agents = {AIRCRAFT: aircraft, VEHICLE: vehicle, UNKNOWN: unknown}
    return raster_map, hold_lines, graph_nx, limits, agents

def plot_agent(asset, heading, zoom = 0.015):
    img = ndimage.rotate(asset, heading) 
    img = np.fliplr(img) 
    img = OffsetImage(img, zoom=zoom)
    return img

def save(filetag, dpi=600):
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])

    # Set figure bbox around the predicted trajectory
    # plt.show(block = False)
    plt.savefig(f'{filetag}.png', dpi=dpi, bbox_inches='tight')
    plt.close()

def xy_to_ll(traj_rel, gt_abs, reference, geodesic, ego_id: int = 0, hist_len: int = 0):
    """
    Args:
        mu (tensor): tensor containing model's prediction in relative XY
        hist_abs (tensor): tensor containg past trajectory in absolute XY
        reference (Tuple): tuple containing the reference lat/lon points
        geodesic (Geodesic): geode for computing lat/lon

    Returns:
        tensor: tensor containing mu's values in lat/lon.
    """
    N, _ , _ = gt_abs.shape
    gt_xy = gt_abs[:, :, C.XY].detach().cpu().numpy()
    start_abs = gt_abs[ego_id, hist_len-1, C.XY].detach().cpu().numpy()
    start_heading = gt_abs[ego_id, hist_len-1, C.HD]
    traj_ll = torch.zeros_like(traj_rel)
    
    traj_xy_abs = transform(traj_rel.cpu().numpy(), start_abs, start_heading)
    
    # if(not np.allclose(gt_xy, traj_xy_abs)):
    #     breakpoint()
    for n in range(N):
        x = traj_xy_abs[n, :, 0]
        y = traj_xy_abs[n, :, 1]
        rang = np.sqrt(x ** 2 + y ** 2)
        bearing = np.degrees(np.arctan2(y, x))
        # lat, lon
        lat, lon = direct_wrapper(geodesic, bearing, rang, reference[0], reference[1], reference[2])
        traj_ll[n , :, 1] = lon 
        traj_ll[n , :, 0] = lat
    return traj_ll

def plot_scene(scenario: dict, assets: Tuple, filetag: str, order_list = None):
    raster_map, hold_lines, graph_map, ll_extent, agents = assets
    north, east, south, west = ll_extent

    # Save states
    breakpoint()
    fig, movement_plot = plt.subplots()
    # Display global map
    movement_plot.imshow(
        raster_map, zorder=0, extent=[west, east, south, north], alpha=0.8, cmap='gray_r') 
    
    sequences = scenario['sequences']
    agent_types = scenario['agent_types']

    N, T, D = sequences.shape 
    for n in range(N):
        # Get heading at last point of trajectory
        heading = sequences[n, -1, SEQ_IDX['Heading']]
        agent_type = agent_types[n]

        # Get ground truth sequence in lat/lon
        lat = sequences[n, :, SEQ_IDX['Lat']]
        lon = sequences[n, :, SEQ_IDX['Lon']]

        x = sequences[n, :, SEQ_IDX['x']]
        y = sequences[n, :, SEQ_IDX['y']]



        img = plot_agent(agents[agent_type], heading, ZOOM[agent_type])
    
        # Place plane on last point of ground truth sequence
        ab = AnnotationBbox(img, (lon[-1], lat[-1]), frameon=False) 
        movement_plot.add_artist(ab)
        if order_list is None:
            movement_plot.plot(lon, lat, color=COLOR_MAP['gt_hist'], lw=0.65) 
        else:
            #red is critical, green is not
            color = COLOR_MAP['follower'] if n in order_list[:5] else COLOR_MAP['leader']
            movement_plot.plot(lon, lat, color=color, lw=0.65) 
    
    # Get conflict points (Hold lines) and plot them on the map
    # hold_lines = pickle_map[pickle_map[:, MAP_IDX['SemanticID']] == 1]
    hold_lines_lon = hold_lines[:, MAP_IDX['LonStart']]
    hold_lines_lat = hold_lines[:, MAP_IDX['LatStart']]
    plt.scatter(hold_lines_lon, hold_lines_lat, color=COLOR_MAP['holdline'], s=5)

    save(filetag)


parser = argparse.ArgumentParser()
parser.add_argument("--airport", type=str, default="ksea", choices=["ksea", "kbos", "kmdw", "kewr"])
parser.add_argument("--base_dir", type=str, default="../datasets/amelia")
parser.add_argument("--out_dir", type=str, default="./out/critical")
parser.add_argument("--num_scenes", type=int, default=20)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

scenarios_dir = os.path.join(args.base_dir, "traj_data_a10v7/proc_trajectories", args.airport)
assets = load_assets(base_dir=args.base_dir, airport=args.airport)

scenarios = glob.glob(f"{scenarios_dir}/**/*.pkl", recursive=True)
random.shuffle(scenarios)
scenarios = scenarios[:args.num_scenes]
print(f"Running {len(scenarios)} scenarios")

for n, scenario in enumerate(tqdm(scenarios)):
    split = scenario.split("/")
    subdir, scenario_id = split[-2], split[-1].split('.')[0]

    outdir = os.path.join(args.out_dir, args.airport)
    os.makedirs(outdir, exist_ok=True)

    with open(scenario,'rb') as f:
        scene = pickle.load(f)
    
    filetag = os.path.join(outdir, f"{subdir}_{scenario_id}")
    plot_scene(scene, assets, filetag)