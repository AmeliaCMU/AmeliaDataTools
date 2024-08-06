import os
from tqdm import tqdm
import pickle
import amelia_datatools.utils.common as C
from amelia_datatools.utils.utils import get_scene_list, plot_scene, get_file_name
from amelia_scenes.splitting.dataset import load_assets


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--airport', default="kbos", type=str, help='ICAO Airport Code')
    parser.add_argument('--base_dir', default=C.DATA_DIR, type=str, help='Input path')
    parser.add_argument('--traj_version', type=str, default=C.VERSION)
    parser.add_argument(
        '--output_dir', default=f"{C.OUTPUT_DIR}/{get_file_name(__file__)}", type=str)
    parser.add_argument('--max_scenes', default=100, type=int)
    args = parser.parse_args()

    assets = load_assets(input_dir=args.base_dir, airport=args.airport)
    scenes, trajdirs = get_scene_list(airpoirt=args.airport, base_dir=args.base_dir,
                                      version=args.traj_version)

    if len(scenes) > args.max_scenes:
        scenes = scenes[:args.max_scenes]
        trajdirs = trajdirs[:args.max_scenes]

    for scene, trajdir in tqdm(zip(scenes, trajdirs), total=len(scenes)):
        with open(scene, 'rb') as f:
            scene_data = pickle.load(f)

        subdir = os.path.join(args.output_dir, args.airport, trajdir.split('/')[-1])
        scenario_id = scene_data['scenario_id']
        filetag = os.path.join(args.output_dir, f"{scenario_id}")
        os.makedirs(subdir, exist_ok=True)
        plot_scene(scenario=scene_data, assets=assets, filetag=filetag)
