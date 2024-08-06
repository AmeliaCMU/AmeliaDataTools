import os
import pandas as pd
import numpy as np
import datetime
from datetime import timezone
import pytz
from pyproj import Transformer
from amelia_datatools.utils.common import AIRCRAFT, VEHICLE, UNKNOWN
import cv2
import json
import imageio
import pickle
from easydict import EasyDict

pd.options.mode.chained_assignment = None  # default='warn'

METERS_PER_SECOND_2_KNOTS = 1.94384

# TIME_ZONES = {
#     "ksea": 8,
#     "kewr": 5,
#     "kmdw": 6,
#     "kbos": 5
# }

GMT_TIME_ZONE = pytz.timezone('GMT')

TIME_ZONES = {
    "panc": 'US/Alaska',
    "kbos": 'America/New_York',
    "kdca": 'America/New_York',
    "kewr": 'America/New_York',
    "kjfk": 'America/New_York',
    "klax": 'America/Los_Angeles',
    "kmdw": 'America/Chicago',
    "kmsy": 'US/Central',
    "ksea": 'US/Pacific',
    "ksfo": 'America/Los_Angeles'
}


plotting_colors = {
    "ksea": "#682D63",
    "kewr": "#519872",
    "kmdw": "#0072bb",
    "kbos": "#ca3c25"
}


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def impute(seq: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    """ Imputes missing data via linear interpolation.

    Inputs
    ------
        seq[pd.DataFrame]: trajectory sequence to be imputed.
        seq_len[int]: length of the trajectory sequence.

    Output
    ------
        seq[pd.DataFrame]: trajectory sequence after imputation.
    """
    # Create a list from starting frame to ending frame in agent sequence
    conseq_frames = set(range(int(seq[0, 0]), int(seq[-1, 0])+1))
    # Create a list of the actual frames in the agent sequence. There may be missing
    # data from which we need to interpolate.
    actual_frames = set(seq[:, 0])
    # Compute the difference between the lists. The difference represents the missing
    # data points
    missing_frames = list(sorted(conseq_frames - actual_frames))
    # print(missing_frames)

    # Insert nan rows on where the missing data is. Then, interpolate.
    if len(missing_frames) > 0:
        seq = pd.DataFrame(seq)
        for missing_frame in missing_frames:
            df1 = seq[:missing_frame]
            df2 = seq[missing_frame:]
            df1.loc[missing_frame] = np.nan
            seq = pd.concat([df1, df2])

        seq = seq.interpolate(method='linear').to_numpy()[:seq_len]
    return seq


def get_movement_stats(traj: np.array, idxs: dict):
    agent_heading = traj[:, idxs['Heading']]
    # agent_speed   = traj[:, idxs['Speed']] / METERS_PER_SECOND_2_KNOTS
    x, y = traj[:, idxs['x']], traj[:, idxs['y']]
    # Calculate movement stats
    # Get wrapping value of heading difference in degrees
    heading_diff = 180 - abs(abs(agent_heading[1:] - agent_heading[:-1]) - 180)
    # speed_diff    =  agent_speed[1:] - agent_speed[:-1] # Aceleration in m/s^2
    x_rel, y_rel = x[1:] - x[:-1], y[1:] - y[:-1]
    is_stationary = np.allclose(x_rel, 0.0) and np.allclose(y_rel, 0.0)
    stats = heading_diff, is_stationary
    del x, y, agent_heading, x_rel, y_rel
    return stats


def get_time_from_file(filename, airport_id):
    time_zone = pytz.timezone(TIME_ZONES[airport_id])
    timestamp_str = os.path.splitext(filename)[0].split('_')[-1]
    timestamp = int(timestamp_str)
    utc_datetime = datetime.datetime.fromtimestamp(timestamp, tz=timezone.utc)
    local_time = pytz.utc.localize(utc_datetime, is_dst=None).astimezone(time_zone)
    return local_time.hour


def transform_extent(extent, original_crs: str, target_crs: str):
    transformer = Transformer.from_crs(original_crs, target_crs)
    north, east, south, west = extent
    xmin_trans, ymin_trans = transformer.transform(south, west)
    xmax_trans, ymax_trans = transformer.transform(north, east)
    return (ymax_trans, xmax_trans, ymin_trans, xmin_trans)


def polar_histogram(ax, data, color, bins=80, density=False, offset=0, allow_gaps=True, fill=True):
    """
    Credit: https://stackoverflow.com/questions/22562364/circular-polar-histogram-in-python

    """
    data = np.deg2rad(data)

    if not allow_gaps:
        # Force bins to partition entire circle
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    n, bins = np.histogram(data, bins=bins)
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / data.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5

    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor=color, fill=fill, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches


#
# TODO: delete deprecated function
# def load_assets(input_dir: str, airport: str) -> tuple:
#     # Graph
#     graph_data_dir = os.path.join(input_dir, "graph_data_a10v01os", airport)
#     print(f"Loading graph data from: {graph_data_dir}")
#     pickle_map_filepath = os.path.join(graph_data_dir, "semantic_graph.pkl")
#     with open(pickle_map_filepath, 'rb') as f:
#         graph_pickle = pickle.load(f)
#         hold_lines = graph_pickle['hold_lines']
#         graph_nx = graph_pickle['graph_networkx']
#         # pickle_map = temp_dict['map_infos']['all_polylines'][:]

#     assets_dir = os.path.join(input_dir, "assets")
#     print(f"Loading assets from: {assets_dir}")

#     # Map asset
#     raster_map_filepath = os.path.join(assets_dir, airport, "bkg_map.png")
#     raster_map = cv2.imread(raster_map_filepath)
#     raster_map = cv2.resize(
#         raster_map, (raster_map.shape[0]//2, raster_map.shape[1]//2))
#     raster_map = cv2.cvtColor(raster_map, cv2.COLOR_BGR2RGB)

#     # Reference file
#     limits_filepath = os.path.join(assets_dir, airport, 'limits.json')
#     with open(limits_filepath, 'r') as fp:
#         ref_data = EasyDict(json.load(fp))
#     alt = ref_data.limits.Altitude
#     espg = ref_data.espg_4326
#     limits = (espg.north, espg.east, espg.south, espg.west, alt.min, alt.max)

#     # Agent assets
#     agents = {
#         AIRCRAFT: imageio.imread(os.path.join(assets_dir, "ac.png")),
#         VEHICLE: imageio.imread(os.path.join(assets_dir, "vc.png")),
#         UNKNOWN: imageio.imread(os.path.join(assets_dir, "uk_ac.png"))
#     }
#     return raster_map, hold_lines, graph_nx, limits, agents
