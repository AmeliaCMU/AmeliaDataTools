import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd
import random 
random.seed(4242)
np.set_printoptions(suppress=True)
from itertools import groupby

from tqdm import tqdm 

from matplotlib.offsetbox import AnnotationBbox
import scenario_identification.scene_utils.common as C

SUBDIR = __file__.split('/')[-1].split('.')[0]
INT = '[INT]'

def plot_scene(scenario, assets, filetag, order_list = None, version = 'v5'):
    raster_map, hold_lines, graph_map, ll_extent, agents = assets
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
        heading = sequences[n, -1, C.SEQ_IDX['Heading']]
        agent_type = agent_types[n]

        # Get ground truth sequence in lat/lon
        lat = sequences[n, :, C.SEQ_IDX['Lat']]
        lon = sequences[n, :, C.SEQ_IDX['Lon']]

        img = C.plot_agent(agents[agent_type], heading, C.ZOOM[agent_type])
        ab = AnnotationBbox(img, (lon[-1], lat[-1]), frameon=False) 
        movement_plot.add_artist(ab)
        movement_plot.plot(lon, lat, color='blue', lw=0.65) 

        if version == 'v5':
            movement_plot.scatter(lon, lat, color='red', lw=0.65, s=2) 

        else:
            interp = sequences[n, :, C.SEQ_IDX['Interp']]

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
    plt.scatter(hold_lines_lon, hold_lines_lat, color=C.COLOR_MAP['holdline'], s=5)

    C.save(filetag)

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--airport', default="ksea", choices=["ksea", "kewr"])
    parser.add_argument("--base_dir", type=str, default="../../datasets/amelia_v6/raw_trajectories")
    parser.add_argument("--out_dir", type=str, default="./out")
    args = parser.parse_args()

    out_dir = os.path.join(args.out_dir, SUBDIR, args.airport)
    os.makedirs(out_dir, exist_ok=True)

    map_dir = os.path.join(f"../../datasets/amelia_v5/maps/{args.airport}")    
    assets = C.load_assets(map_dir=map_dir)

    version = args.base_dir.split('/')[3].split('_')[-1]
    print(f"Version: {version}")

    base_dir = os.path.join(args.base_dir, args.airport)
    csv_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir)]

    total_agents = 0
    interp_agents = 0
    interp_T = []
    total_scenarios = 0
    curr_scenarios = 0

    N = len(csv_files)
    for n, csv_file in enumerate(csv_files):
        # get file timestamp
        timestamp = csv_file.split('/')[-1].split('.')[0].split('_')[-1]
        data = pd.read_csv(csv_file)
        
        agent_IDs = data.ID.unique().tolist()
        total_agents += len(agent_IDs)

        for agent_ID in agent_IDs:
            agent_seq = data[:][data.ID == agent_ID] 
            agent_interp = agent_seq.Interp.to_numpy()
            if INT in agent_interp:
                x = (agent_interp == INT).astype(int).tolist()
                grouped = (list(g) for _,g in groupby(enumerate(x), lambda t:t[1]))
                for g in grouped:
                    if g[0][1] == 1:
                        t = 1 if len(g) == 1 else g[-1][0] - g[0][0]
                        interp_T.append(t)
                interp_agents += 1
        
        perc_interp = round(100 * interp_agents / total_agents, 4)
        perc_files = round(100 * (n+1)/N, 4)
        print(f"Interp. agents: {perc_interp}% Interp. files: {perc_files}%", end="\r")

    perc_out = f"Percentage of interpolated agents: {perc_interp}%\n"    
    str_out = f"Total Files: {N}, Total Trajectories: {total_agents} Interp Traj: {interp_agents}\n"
    print(str_out)
    with open(f"{out_dir}/stats.txt", "w") as f:
        f.write(str_out)
        f.write(perc_out)

    interp_T = np.asarray(interp_T)
    plt.hist(interp_T, bins=60, range=(0, 60), label='Interpolated timesteps')
    plt.legend()
    plt.savefig(f"{out_dir}/{args.airport}.png", dpi=600, bbox_inches='tight')
    plt.close()

    idx = np.where(interp_T >= 10)[0]
    interp_T10 = interp_T[idx]
    plt.hist(interp_T10, bins=60, range=(10, 60), label='Interpolated timesteps')
    plt.legend()
    plt.savefig(f"{out_dir}/{args.airport}_t10.png", dpi=600, bbox_inches='tight')
    plt.close()