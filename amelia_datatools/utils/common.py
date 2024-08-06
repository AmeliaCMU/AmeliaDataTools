from enum import Enum
import os
from easydict import EasyDict

# Base paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(ROOT_DIR, "../..")
ROOT_DIR = os.path.normpath(ROOT_DIR)
OUT_DIR = os.path.join(ROOT_DIR, "output")

VIS_DIR = os.path.join(f"{OUT_DIR}", "visualization")
STATS_DIR = os.path.join(f"{OUT_DIR}", "statistics")
CACHE_DIR = os.path.join(f"{OUT_DIR}", "cache")

DATA_DIR = os.path.join(ROOT_DIR, "datasets/amelia")
VERSION = "a10v08"
TRAJ_DATA_DIR = os.path.join(f"{DATA_DIR}", f"traj_data_{VERSION}")
ASSET_DATA_DIR = os.path.join(f"{DATA_DIR}", "assets")
GRAPH_DATA_DIR = os.path.join(f"{DATA_DIR}", "graph_data")

# Global variables
DPI = 600

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
    "panc": "crimson",
    "kbos": "lightcoral",
    "kdca": "orangered",
    "kewr": "#2E8B57",
    "kjfk": "limegreen",
    "klax": "darkturquoise",
    "kmdw": "dodgerblue",
    "kmsy": "mediumorchid",
    "ksea": "violet",
    "ksfo": "deeppink"
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
