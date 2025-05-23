from enum import Enum
import os
from easydict import EasyDict

# Base paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(ROOT_DIR, "../..")
ROOT_DIR = os.path.normpath(ROOT_DIR)
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

VIS_DIR = os.path.join(f"{OUTPUT_DIR}", "visualization")
STATS_DIR = os.path.join(f"{OUTPUT_DIR}", "statistics")
CACHE_DIR = os.path.join(f"{OUTPUT_DIR}", "cache")

DATA_DIR = os.path.join(ROOT_DIR, "datasets/amelia")
VERSION = "a42v01"
TRAJ_DATA_DIR = os.path.join(f"{DATA_DIR}", f"traj_data_{VERSION}")
ASSET_DATA_DIR = os.path.join(f"{DATA_DIR}", "assets")
GRAPH_DATA_DIR = os.path.join(f"{DATA_DIR}", "graph_data")

# Global variables
DPI = 300

AIRCRAFT = 0
VEHICLE = 1
UNKNOWN = 2

ZOOM = {
    AIRCRAFT: 0.015,
    VEHICLE: 0.2,
    UNKNOWN: 0.015
}


class AgentType(Enum):
    AIRCRAFT = 0
    VEHICLE = 1
    UNKNOWN = 2


AIRPORT_COLORMAP = {
    'panc': 'crimson',
    'kbos': 'lightcoral',
    'kdca': 'orangered',
    'kewr': '#2E8B57',
    'kjfk': 'limegreen',
    'klax': 'darkturquoise',
    'kmdw': 'dodgerblue',
    'kmsy': 'mediumorchid',
    'ksea': 'violet',
    'ksfo': 'deeppink',
    'katl': 'darkorange',
    'kbfi': 'darkred',
    'kpit': 'darkgreen',
    'ksan': 'darkblue',
    'kdfw': 'darkviolet',
    'kcle': 'darkcyan',
    'kmke': 'darkgoldenrod',
    'kbdl': '#ffffb3',
    'kbwi': '#bebada',
    'kclt': '#fb8072',
    'kcvg': '#80b1d3',
    'kden': '#fdb462',
    'kdtw': '#b3de69',
    'kfll': '#fccde5',
    'khou': '#d9d9d9',
    'khwd': '#bc80bd',
    'kiad': '#ccebc5',
    'kiah': '#ffed6f',
    'klas': '#8dd3c7',
    'klga': '#ffff33',
    'kmci': '#a6d854',
    'kmco': '#fb8072',
    'kmem': '#80b1d3',
    'kmia': '#fdb462',
    'kmsp': '#b3de69',
    'koak': '#fccde5',
    'kord': '#d9d9d9',
    'korl': '#bc80bd',
    'kpdx': '#ccebc5',
    'kphl': '#ffed6f',
    'kphx': '#ff7f00',
    'kpvd': '#cab2d6',
    'kpwk': '#6a3d9a',
    'ksdf': '#b15928',
    'kslc': '#1f78b4',
    'ksna': '#33a02c',
    'kstl': '#e31a1c',
    'phnl': '#fdbf6f'
}

#
AIRPORT_CROWDEDNESS = {
    "high": ["klax", "kjfk"],
    "mid": ["ksea", "kewr", "ksfo", "kbos", "kdca", "kmdw"],
    "low": ["kmsy", "panc"],
}

MOTION_PROFILE = {
    'acceleration': {
        'label': 'Acceleration',
        'unit': 'm/s²'
    },
    'speed': {
        'label': 'Speed',
        'unit': 'm/s'
    },
    'heading': {
        'label': 'Heading Change',
        'unit': 'Degrees'
    }
}

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

KNOTS_2_MS = 1/1.94384

SEQ_IDX = {
    'Speed': 0,
    'Heading': 1,
    'Lat': 2,
    'Lon': 3,
    'Range': 4,
    'Bearing': 5,
    'x': 6,
    'y': 7,
    'z': 8,
    'Interp': 9
}

XY = [False, False, False, False, False, False, True, True, False, False]
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

COLOR_MAP = {
    'gt_hist': '#FF5A4C',
    'gt_future': '#8B5FBF',
    'holdline': "#0DD7EF",
    'follower': "#EF0D5C",
    'leader': "#93EF0D",
    'invalid': "#000000"
}

AIRPORTS = {
    "kmdw": "#0072bb",
    "kewr": "#519872",
    "ksea": "#682D63",
    "kbos": "#ca3c25"
}
