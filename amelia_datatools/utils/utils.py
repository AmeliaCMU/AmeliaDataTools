import os
import glob
import matplotlib.pyplot as plt
import glob

import amelia_datatools.utils.common as C
from amelia_scenes.visualization.common import plot_agent
from matplotlib.offsetbox import AnnotationBbox


def save(filetag, dpi=400):
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])

    plt.savefig(f'{filetag}.png', dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_scene(scenario, assets, filetag, order_list=None):
    bkg, hold_lines, graph_nx, limits, agents = assets
    limits, ref_data = limits
    north, east, south, west, z_min, z_max = limits
    # Save states
    fig, movement_plot = plt.subplots()
    # Display global map
    movement_plot.imshow(
        bkg, zorder=0, extent=[west, east, south, north], alpha=0.8, cmap='gray_r')

    sequences = scenario['agent_sequences']
    agent_types = scenario['agent_types']

    N, T, D = sequences.shape
    for n in range(N):
        # Get heading at last point of trajectory
        heading = sequences[n, -1, C.SEQ_IDX['Heading']]
        agent_type = agent_types[n]

        # Get ground truth sequence in lat/lon
        lat = sequences[n, :, C.SEQ_IDX['Lat']]
        lon = sequences[n, :, C.SEQ_IDX['Lon']]

        img = plot_agent(agents[agent_type], heading, C.ZOOM[agent_type])

        # Place plane on last point of ground truth sequence
        ab = AnnotationBbox(img, (lon[-1], lat[-1]), frameon=False)
        movement_plot.add_artist(ab)
        if order_list is None:
            movement_plot.plot(lon, lat, color=C.COLOR_MAP['gt_hist'], lw=0.65)
        else:
            # red is critical, green is not
            color = C.COLOR_MAP['follower'] if n in order_list[:5] else C.COLOR_MAP['leader']
            movement_plot.plot(lon, lat, color=color, lw=0.65)

    # Get conflict points (Hold lines) and plot them on the map
    # hold_lines = pickle_map[pickle_map[:, MAP_IDX['SemanticID']] == 1]
    hold_lines_lon = hold_lines[:, C.MAP_IDX['LonStart']]
    hold_lines_lat = hold_lines[:, C.MAP_IDX['LatStart']]
    plt.scatter(hold_lines_lon, hold_lines_lat, color=C.COLOR_MAP['holdline'], s=5)

    save(filetag)


def get_scene_list(airport, base_dir, version):
    scene_list = []

    traj_dir = os.path.join(base_dir, f'traj_data_{version}/proc_full_scenes', f'{airport}')
    trajdirs = [os.path.join(traj_dir, f) for f in os.listdir(traj_dir)]
    for trajdir in trajdirs:
        scenarios = glob.glob(f"{trajdir}/*.pkl", recursive=True)
        scene_list += scenarios

    return scene_list, trajdirs


def get_file_name(file_path: str) -> str:
    """Extracts the file name from a file path.

    Args:
        file_path (str): The file path.

    Returns:
        str: The file name.
    """
    return os.path.basename(file_path).split(".")[0]


def save(filetag, dpi=400):
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])

    # Set figure bbox around the predicted trajectory
    # plt.show(block = False)
    plt.savefig(f'{filetag}.png', dpi=dpi, bbox_inches='tight')
    plt.close()


def get_airport_list():
    assets_dir = os.path.join(C.DATA_DIR, 'assets')
    files = glob.glob(f"{assets_dir}/*")
    airport_list = []
    for file in files:
        if os.path.isdir(file):
            airport_list.append(os.path.basename(file))
    return airport_list
