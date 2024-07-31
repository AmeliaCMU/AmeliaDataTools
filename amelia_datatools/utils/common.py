from enum import Enum
import os
from easydict import EasyDict

# Base paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(ROOT_DIR, "../..")
ROOT_DIR = os.path.normpath(ROOT_DIR)
OUT_DIR = os.path.join(ROOT_DIR, "output")

VIS_DIR = f"{OUT_DIR}/vis"
STATS_DIR = f"{OUT_DIR}/stats"
CACHE_DIR = f"{OUT_DIR}/cache"

DATA_DIR = os.path.join(ROOT_DIR, "datasets/amelia")
VERSION = "a10v08"
TRAJ_DATA_DIR = f"{DATA_DIR}/traj_data_{VERSION}"
ASSET_DATA_DIR = f"{DATA_DIR}/assets"
GRAPH_DATA_DIR = f"{DATA_DIR}/graph_data"

# Global variables


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
