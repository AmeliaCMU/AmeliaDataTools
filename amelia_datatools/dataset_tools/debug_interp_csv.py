from amelia_datatools.utils.utils import get_file_name
from amelia_datatools.utils.common import VERSION, DATA_DIR, OUTPUT_DIR
from itertools import groupby
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random


def debug_interpolation(base_dir: str, airport: str, output_dir: str, traj_version: str):
    random.seed(42)

    np.set_printoptions(suppress=True)

    INT = '[INT]'

    output_dir = os.path.join(output_dir, get_file_name(__file__), airport)
    os.makedirs(output_dir, exist_ok=True)

    cvs_dir = os.path.join(base_dir, f'traj_data_{traj_version}/raw_trajectories', airport)

    csv_files = [os.path.join(cvs_dir, f) for f in os.listdir(cvs_dir)]

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
                grouped = (list(g) for _, g in groupby(enumerate(x), lambda t: t[1]))
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
    with open(f"{output_dir}/stats.txt", "w") as f:
        f.write(str_out)
        f.write(perc_out)

    interp_T = np.asarray(interp_T)
    plt.hist(interp_T, bins=60, range=(0, 60), label='Interpolated timesteps')
    plt.legend()
    plt.savefig(f"{output_dir}/{airport}.png", dpi=600, bbox_inches='tight')
    plt.close()

    idx = np.where(interp_T >= 10)[0]
    interp_T10 = interp_T[idx]
    plt.hist(interp_T10, bins=60, range=(10, 60), label='Interpolated timesteps')
    plt.legend()
    plt.savefig(f"{output_dir}/{airport}_t10.png", dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--airport', default="ksea", choices=["ksea", "kewr"])
    parser.add_argument("--base_dir", type=str, default=f"{DATA_DIR}")
    parser.add_argument('--traj_version', default=VERSION)
    parser.add_argument("--output_dir", type=str, default=f"{OUTPUT_DIR}")
    args = parser.parse_args()

    debug_interpolation(**vars(args))
