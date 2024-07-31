# --------------------------------------------------------------------------------------------------
# @name     dataset_semantics_generator.py
# @brief    Simple script to test generating semantic patches 
# --------------------------------------------------------------------------------------------------
import cv2
import json 
import numpy as np
import os
import pandas as pd
import time

from geographiclib.geodesic import Geodesic
from math import cos, sin, radians, degrees
from tqdm import tqdm

def compute_aggregated_motion(trajectory):
    """ Computes the total agregated motion for a trajectory along both, x and y. """
    x, y = trajectory.x.to_numpy(), trajectory.y.to_numpy()
    x_rel, y_rel = abs(x[1:] - x[:-1]), abs(y[1:] - y[:-1])
    return sum(x_rel) + sum(y_rel)

def bearing(geodesic, lat1, lon1, lat2, lon2, to_radians=True):
    """ Compute bearing angle between two coordinates. """
    g = geodesic.Inverse(lat1, lon1, lat2, lon2)
    if to_radians:
        return radians(g['azi1'])
    return g['azi1']

def bbox_ll_to_xy(bbox: tuple, limits: tuple, im_shape: tuple) -> tuple:
    """ Convert a bounding box from Latitude/Longitude map frame to (x, y) image frame."""
    north, east, south, west = limits
    n, e, s, w = bbox
    H, W, _ = im_shape

    # get min and max lon/lat from the provided limits
    max_lat, min_lat = max(north, south), min(north, south) 
    max_lon, min_lon = max(east, west), min(east, west)
    
    # map the lat/lon coordinates to image coordinates
    h_min = max(int(((max_lat - n) / (max_lat - min_lat)) * H), 0)
    h_max = min(int(((max_lat - s) / (max_lat - min_lat)) * H), H)
    w_min = max(int(((w - min_lon) / (max_lon - min_lon)) * W), 0)
    w_max = min(int(((e - min_lon) / (max_lon - min_lon)) * W), W)
    return h_min, h_max, w_min, w_max

def convert_ll_to_xy(lat: np.array, lon: np.array, limits: tuple, im_shape: tuple) -> tuple:
    """ Convert a bounding box from Latitude/Longitude map frame to (x, y) image frame."""
    north, east, south, west = limits
    H, W, _ = im_shape

    # get min and max lon/lat from the provided limits
    max_lat, min_lat = max(north, south), min(north, south) 
    max_lon, min_lon = max(east, west), min(east, west)
    
    X = (((lon - min_lon) / (max_lon - min_lon)) * W).astype(int)
    X[np.where(X >= W)] = W-1
    X[np.where(X < 0)] = 0

    Y = (((max_lat - lat) / (max_lat - min_lat)) * H).astype(int)
    Y[np.where(Y > H)] = H-1
    Y[np.where(Y < 0)] = 0
    return X, Y

def compute_trajectory_bbox(
    lat: np.array, lon: np.array, offset: float, obs_len: int = 10, fut_len: int =20) -> tuple: 
    """ Computes the bounding box for a given trajectory as the max distance from trajectory[0] 
    to any point in the obseverved portion. Current assumption: speed of agent will stay roughly the 
    same through the entire trajectory, so we assume that the total distance can be computed as 
                        dist = max_dist_obs * traj_len / obs_len + offset
    TODO: find more suitable way for getting the bounding boxes. 
    """
    lon_0, lat_0 = lon[0], lat[0]
    x = (fut_len + obs_len) // obs_len
    dist = np.sqrt((lon_0 - lon[1:obs_len]) ** 2 + (lat_0 - lat[1:obs_len]) ** 2).max() * x + offset
    # get coordinates for patch without rotation
    return (lat_0 + dist, lon_0 + dist, lat_0 - dist, lon_0 - dist)

def compute_rotated_patch(image: np.array, bbox: tuple, theta: float) -> np.array:
    """ Computes a rotated patch given an image, the bounding box and the rotation angle."""
    h_min, h_max, w_min, w_max = bbox
    w, h = w_max - w_min, h_max - h_min
    center_x, center_y = w // 2 + w_min, h // 2 + h_min
    H, W, C = image.shape

    # get meshgrid of patch coordinates to rotate
    x = np.linspace(w_min, w_max, num=w) - center_x
    y = np.linspace(h_min, h_max, num=h) - center_y
    X, Y = np.meshgrid(x, y)
    coords = np.asarray([X.reshape(1, -1), Y.reshape(1, -1)]).reshape(2, -1)

    # rotate the coordinates
    R = np.array(
        [[ cos(theta), sin(theta)], 
         [-sin(theta), cos(theta)]])
    rot_coords = (R @ coords).astype(int)

    # adjust coordinates
    rot_coords[0] = rot_coords[0] + center_x
    rot_coords[0, np.where(rot_coords[0] < 0)] = 0
    rot_coords[0, np.where(rot_coords[0] >= W)] = W-1

    rot_coords[1] = rot_coords[1] + center_y
    rot_coords[1, np.where(rot_coords[1] < 0)] = 0
    rot_coords[1, np.where(rot_coords[1] >= H)] = H-1

    # get patch 
    # breakpoint()
    patch = image[rot_coords[1], rot_coords[0]].reshape(h, w, C)

    # image_copy = image.copy()
    # cv2.rectangle(image_copy, (w_min, h_min), (w_max, h_max), (0, 0, 255), 10)
    # cv2.rectangle(
    #   image, (rot_coords[0].min(), rot_coords[1].min()), (rot_coords[0].max(), rot_coords[1].max()), 
    #   (0, 255, 255), 10)
    # image_res = cv2.resize(image_copy, (image.shape[0] // 10, image.shape[1] // 10))
    # patch_res = cv2.resize(patch, (128, 128))
    # cv2.imshow("imag", image_res)
    # cv2.imshow("rot", patch_res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return patch

def generate_trajectory_patches(
    geodesic: Geodesic, trajectory: pd.DataFrame, semantic_map: np.array, offset: float, 
    limits: tuple,  patch_dim: int = 128, obs_len: int = 10, fut_len: int = 20, tag: str = 'temp.png',
) -> np.array:
    """ Given a trajectory and a global map, generates its corresponding semantic patch. """
    # breakpoint()
    
    # for pd.DataFrame
    lon, lat = trajectory.Lon.to_numpy(), trajectory.Lat.to_numpy()
    
    # for np.array 
    # lat, lon = trajectory[:, 0], trajectory[:, 1]
    bbox_ll = compute_trajectory_bbox(
        lat=lat, lon=lon, offset=offset, obs_len=obs_len, fut_len=fut_len)

    # translate coordinates from map frame to image frame 
    bbox_xy = bbox_ll_to_xy(bbox=bbox_ll, limits=limits, im_shape=semantic_map.shape)

    # plot the trajectory for reference
    X, Y = convert_ll_to_xy(lat=lat, lon=lon, limits=limits, im_shape=semantic_map.shape)
    for i in range(X.shape[0]):
        y, x = Y[i], X[i]
        semantic_map = cv2.circle(semantic_map, (x, y), radius=30, color=(0, 0, 255), thickness=-1)
    resized = cv2.resize(semantic_map, (semantic_map.shape[0]//20, semantic_map.shape[1]//20))
    # cv2.imshow('Map', resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    theta = -radians(trajectory.Heading.to_numpy()[0])
    patch = compute_rotated_patch(image=semantic_map, bbox=bbox_xy, theta=theta)
    patch = cv2.resize(patch, (patch_dim, patch_dim))
    cv2.imwrite(tag+f"_theta-{round(degrees(theta), 2)}.png", patch)

    norm_patch = patch / 255.0
    # np.save(tag+".npy", norm_patch)
    return norm_patch
        
def run(
    ipath: str, opath: str, mpath: str, airport: str, ll_offset: float, num_files: int, num_trajs: int, 
    traj_len: int, patch_dim: int, show: bool, mask: bool
):
    airport_dir = os.path.join(ipath, airport) 
    trajectory_files = [os.path.join(airport_dir, f) for f in os.listdir(airport_dir)]
    assert os.path.exists(airport_dir), f"Path {airport_dir} does not exist."

    output_dir = os.path.join(opath, airport)
    os.makedirs(output_dir, exist_ok=True)
    
    reference_file = os.path.join(mpath, airport, 'limits.json')
    assert os.path.exists(reference_file), f"Reference file {reference_file} does not exist."
    
    with open(reference_file, 'r') as f:
        reference_data = json.load(f)
    semantic_map_file = os.path.join(mpath, airport, reference_data['semantic_map'])
    semantic_map = cv2.imread(semantic_map_file)
    # semantic_map = cv2.cvtColor(semantic_map, cv2.COLOR_BGR2RGB)

    # visualize for debugging 
    if show:
        resized = cv2.resize(semantic_map, (semantic_map.shape[0]//10, semantic_map.shape[1]//10))
        cv2.imshow('Map', resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    north, south = reference_data['north'], reference_data['south']
    west, east = reference_data['west'], reference_data['east']
    ll_limits = (north, east, south, west)

    geodesic = Geodesic.WGS84

    start = time.time()
    for f, trajectory_file in enumerate(tqdm(trajectory_files)):
        if f > num_files:
            break

        data = pd.read_csv(trajectory_file)
        unique_IDs = data.ID.unique()

        for i, ID in tqdm(enumerate(unique_IDs)):
            if i > num_trajs:
                break

            trajectory = data[:][data.ID == ID]
            if trajectory.shape[0] < traj_len:
                continue
            trajectory = trajectory[:traj_len]

            # TODO: remove this
            if compute_aggregated_motion(trajectory) == 0.0:
                continue
            
            smap = semantic_map.copy()
            generate_trajectory_patches(
                geodesic, trajectory, smap, ll_offset, ll_limits, patch_dim, 
                tag = os.path.join(output_dir, f"file-{f}_traj-{i}"))

    print(f"Total time {time.time() - start} seconds")

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--ipath', default='./swim/raw_trajectories/', type=str, help='Data path.')
    parser.add_argument('--opath', default='./out/semantic_patches', type=str, help='Output path.')
    parser.add_argument('--mpath', default='./swim/maps', type=str, help='Map path.')
    parser.add_argument('--airport', default='kewr', type=str, help='Airport to process.')
    parser.add_argument('--ll_offset', default=0.001, type=float, help='Lat-Lon extent offset.')
    parser.add_argument('--num_files', default=1, type=int, help='Number of files to process')
    parser.add_argument('--num_trajs', default=10, type=int, help='Number of trajectories per file.')
    parser.add_argument('--traj_len', default=30, type=int, help='Each trajectory length.')
    parser.add_argument('--patch_dim', default=16, type=int, help='Patch h, w dimension')
    parser.add_argument('--show', action='store_true', help='Visualize.')
    parser.add_argument('--mask', action='store_true', help='If true, saves mask instead of RGB.')
    args = parser.parse_args()

    run(**vars(args))