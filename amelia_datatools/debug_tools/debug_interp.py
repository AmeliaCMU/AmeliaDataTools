import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random 
random.seed(4242)
np.set_printoptions(suppress=True)

from tqdm import tqdm 

from matplotlib.offsetbox import AnnotationBbox
import scene_utils.common as C

SUBDIR = __file__.split('/')[-1].split('.')[0]

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
    parser.add_argument("--base_dir", type=str, default="../../datasets/amelia_v6/proc_trajectories")
    parser.add_argument("--out_dir", type=str, default="./out")
    args = parser.parse_args()

    out_dir = os.path.join(args.out_dir, SUBDIR, args.airport)
    os.makedirs(out_dir, exist_ok=True)

    map_dir = os.path.join(f"../../datasets/amelia_v5/maps/{args.airport}")    
    assets = C.load_assets(map_dir=map_dir)

    version = args.base_dir.split('/')[3].split('_')[-1]
    print(f"Version: {version}")

    base_dir = os.path.join(args.base_dir, args.airport)
    sub_dirs = os.listdir(base_dir)

    total_agents = 0
    interp_agents = 0
    interp_T = []
    total_scenarios = 0
    curr_scenarios = 0
    for sub_dir in tqdm(sub_dirs): 
        timestamp = sub_dir.split('_')[-1]
        dir_ = os.path.join(base_dir, sub_dir)
        scenarios = glob.glob(f"{dir_}/*.pkl", recursive=True)
        total_scenarios += len(scenarios)
        
        temp_t = []
        for s, scenario in enumerate(scenarios):
            split = scenario.split("/")
            subdir, scenario_id = split[-2], split[-1].split('.')[0]

            with open(scenario,'rb') as f:
                scene = pickle.load(f)
            
            seq = scene["sequences"]
            N, _, _ = seq.shape
            for n in range(N):
                interp = seq[n, :, C.SEQ_IDX["Interp"]]
                interp_idx = np.where(interp == 2.0)[0]
                
                if interp_idx.shape[0] == 1:
                    temp_t.append(1)
                    interp_agents += 1

                elif interp_idx.shape[0] > 1:
                    interp_agents += 1
                    dt = interp_idx[1:] - interp_idx[:-1]
                    mask = np.asarray([True] + (dt[1:] != dt[:-1]).tolist() + [True])
                    mask = np.where(mask == True)[0]
                    if len(mask) < 2: 
                        breakpoint()

                    for i in range(1, len(mask)):
                        t2 = interp_idx[mask[i]]
                        t1 = interp_idx[mask[i-1]]
                        temp_t.append(t2 - t1)

                else:
                    total_agents += 1

            curr_scenarios += 1
            perc_int = round(int(100 * interp_agents / total_agents), 4)
            processed_files = round(int(100 * curr_scenarios / total_scenarios), 4)
            print(f"Interpolated agents {perc_int}% Proc scenarios: {processed_files}", end="\r")
        
        if len(temp_t) > 0:
            plt.hist(np.asarray(temp_t), bins=60, range=(0, 60), label='Interpolated timesteps')
            plt.legend()
            plt.savefig(f"{out_dir}/{args.airport}_{subdir}.png", dpi=600, bbox_inches='tight')
            plt.close()
            interp_T += temp_t

    str_ = f"Total: {total_agents}, interp: {interp_agents}, scenarios: {total_scenarios}"
    print(f"Total: {total_agents}, interp: {interp_agents}, scenarios: {total_scenarios}")

    with open(f"{out_dir}/perc.txt", 'w') as f:
        f.write(str_)

    plt.hist(np.asarray(interp_T), bins=60, range=(0, 60), label='Interpolated timesteps')
    plt.legend()
    plt.savefig(f"{out_dir}/{args.airport}.png", dpi=600, bbox_inches='tight')
    plt.close()