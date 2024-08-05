from amelia_datatools.utils.common import MAP_IDX, SEQ_IDX, COLOR_MAP
# import scenario_identification.scene_utils.common as C
from matplotlib.offsetbox import AnnotationBbox
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random


from amelia_datatools.utils.common import VERSION, ROOT_DIR
from amelia_datatools.utils.utils import get_file_name
from amelia_datatools.utils.processor_utils import load_assets


def plot_scene(scenario, assets, filetag, order_list=None, version='v5'):
    raster_map, hold_lines, graph_map, ll_extent, agents = assets
    print(ll_extent)
    north, east, south, west = ll_extent

    # Save states
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

        img = plot_agent(agents[agent_type], heading, C.ZOOM[agent_type])
        ab = AnnotationBbox(img, (lon[-1], lat[-1]), frameon=False)
        movement_plot.add_artist(ab)
        movement_plot.plot(lon, lat, color='blue', lw=0.65)

        if version == 'v5':
            movement_plot.scatter(lon, lat, color='red', lw=0.65, s=2)

        else:
            interp = sequences[n, :, SEQ_IDX['Interp']]

            # Place plane on last point of ground truth sequence
            if order_list is None:

                idx = np.where(interp == 0.0)[0]
                movement_plot.scatter(lon[idx], lat[idx], color='red', lw=0.65, s=2)
                idx = np.where(interp == 1.0)[0]
                movement_plot.scatter(lon[idx], lat[idx], color='orange', lw=0.65, s=3)
                idx = np.where(interp == 2.0)[0]
                movement_plot.scatter(lon[idx], lat[idx], color='yellow', lw=0.65, s=5)

    # Get conflict points (Hold lines) and plot them on the map
    # hold_lines = pickle_map[pickle_map[:, MAP_IDX['SemanticID']] == 1]
    hold_lines_lon = hold_lines[:, C.MAP_IDX['LonStart']]
    hold_lines_lat = hold_lines[:, C.MAP_IDX['LatStart']]
    plt.scatter(hold_lines_lon, hold_lines_lat, color=COLOR_MAP['holdline'], s=5)

    # save(filetag)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--airport', default="kbos", choices=["ksea", "kewr"])
    parser.add_argument("--base_dir", type=str,
                        default=f"{ROOT_DIR}/datasets/amelia")
    parser.add_argument("--traj_version", type=str, default=VERSION)
    parser.add_argument("--out_dir", type=str, default=f"{ROOT_DIR}/output")
    parser.add_argument("--debug_dir", type=str, default=f"{ROOT_DIR}/output/cache")
    parser.add_argument("--num_dirs", type=int, default=4)
    parser.add_argument("--num_scenes", type=int, default=100)
    args = parser.parse_args()
    random.seed(42)
    np.set_printoptions(suppress=True)

    out_dir = os.path.join(args.out_dir, get_file_name(__file__), args.airport)
    os.makedirs(out_dir, exist_ok=True)

    assets = load_assets(args.base_dir, args.airport)

    proc_dir = os.path.join(
        args.base_dir, f'traj_data_{args.traj_version}/proc_trajectories', args.airport)

    if args.airport == "ksea":
        ts = [f.split('.')[0].split('_')[-1] for f in os.listdir(args.debug_dir)]
        print(f"Timestamps of interest: {ts}")

        sub_dirs = [sd for sd in os.listdir(proc_dir) if sd.split('_')[-1] in ts]
        print(sub_dirs)

        for sub_dir in sub_dirs:
            timestamp = sub_dir.split('_')[-1]
            dir_ = os.path.join(proc_dir, sub_dir)
            scenarios = glob.glob(f"{dir_}/*.pkl", recursive=True)

            for scenario in scenarios:
                split = scenario.split("/")
                subdir, scenario_id = split[-2], split[-1].split('.')[0]

                with open(scenario, 'rb') as f:
                    scene = pickle.load(f)

                out = os.path.join(out_dir, timestamp)
                os.makedirs(out, exist_ok=True)

                filetag = os.path.join(out, f"{scenario_id}_{args.traj_version}")
                plot_scene(scene, assets, filetag, args.traj_version)
    else:
        sub_dirs = os.listdir(proc_dir)[:args.num_dirs]

        for sub_dir in sub_dirs:
            timestamp = sub_dir.split('_')[-1]
            dir_ = os.path.join(proc_dir, sub_dir)
            scenarios = glob.glob(f"{dir_}/*.pkl", recursive=True)[:args.num_scenes]

            for scenario in scenarios:
                split = scenario.split("/")
                subdir, scenario_id = split[-2], split[-1].split('.')[0]

                with open(scenario, 'rb') as f:
                    scene = pickle.load(f)

                out = os.path.join(out_dir, timestamp)
                os.makedirs(out, exist_ok=True)

                filetag = os.path.join(out, f"{scenario_id}_{args.traj_version}")
                plot_scene(scene, assets, filetag, args.traj_version)
