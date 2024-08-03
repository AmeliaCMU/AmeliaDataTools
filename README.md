# AmeliaDataTools


























































































## How to use

<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 8a0c86b (Refactored tarajectory tools (#3))
=======
>>>>>>> 866d1a1 (Refac debug tools (#5))
### Trajectory Tools

#### Compute Agent Counts

`compute_agent_counts.py` computes the number of agents per timestep. Run the following command:

```bash
python amelia_data_tools/trajectory_tools/compute_agent_counts.py
```

The output will be saved in the `./output/stats/compute_agent_counts` directory.

#### Compute Limits

`compute_limits.py` computes the limits of the airports anad updates them. Run the following command:

```bash
python amelia_data_tools/trajectory_tools/compute_limits.py
```

The output will be saved in the `./output/cache/compute_limits` directory.

#### Compute Motion Profiles

`compute_motion_profiles.py` computes the motion profiles of the agents. Run the following command:

```bash
python amelia_data_tools/trajectory_tools/compute_motion_profiles.py --base_dir [base_dir] --traj_version [traj_version] --to_process [to_process] --drop_interp --agent_type [agent_type]
```

Where:

- `[base_dir]` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `[traj_version]` is the version of the trajectory file. By default it is `a10v08`.
- `[to_process]` is the percentage of files to process. By default it is set to `1.0`.
- `--drop_interp` is a flag to drop the interpolated points. By default it is set to `False`.

#### Compute Sequence Lengths

`compute_sequence_lengths.py` computes the sequence lengths of the agents by aerport. Run the following command:

```bash
python amelia_data_tools/trajectory_tools/compute_sequence_lengths.py
```

The output will be saved in the `./output/stats/compute_sequence_lengths` directory.

<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> refac_viz
>>>>>>> 8a0c86b (Refactored tarajectory tools (#3))
=======
>>>>>>> 866d1a1 (Refac debug tools (#5))
### Visualization Tools

#### Plot Agent Statistics

`plot_agent_stats.py` plots agents' statistics, it counts the number of agents per timesteps as well as the number of agents by type. Run the following command:

```bash
python amelia_data_tools/visualization_tools/plot_agent_statistics.py --base_dir [base_dir] --traj_version [traj_version] --dpi [dpi] --num_files [num_files]
```

Where:

- `[base_dir]` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `[traj_version]` is the version of the trajectory file. By default it is `a10v08`.
- `[dpi]` is the resolution of the image. By default it is `600`.
- `[num_files]` is the number of files to plot. By default it is set to `-1`. Which plots all the files in the directory.

The output will be saved in the `./output/visualization/plot_agent_stats` directory.

#### Plot All Trajectories

`plot_all_trajectories.py` plots the trajectories of the agents from the dataset onto the map. Run the following command:

```bash
python amelia_data_tools/visualization_tools/plot_all_trajectories.py --base_dir [base_dir] --traj_version [traj_version] --to_process [to_process] --drop_interp --dpi [dpi]
```

Where:

- `[base_dir]` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `[traj_version]` is the version of the trajectory file. By default it is `a10v08`.
- `[to_process]` is the percentage of files to plot. By default it is set to `1.0`.
- `--drop_interp` is a flag to drop the interpolated points. By default it is set to `False`.
- `[dpi]` is the resolution of the image. By default it is `600`.

The output will be saved in the `./output/visualization/plot_all_trajectories` directory.

#### Plot Average Distance

`plot_average_distance.py` plots the average distance between agents by timestamp. Run the following command:

```bash
python amelia_data_tools/visualization_tools/plot_average_distance.py --base_dir [base_dir] --traj_version [traj_version] --dpi [dpi] --num_files [num_files]
```

Where:

- `[base_dir]` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `[traj_version]` is the version of the trajectory file. By default it is `a10v08`.
- `[dpi]` is the resolution of the image. By default it is `600`.
- `[num_files]` is the number of files to plot. By default it is set to `-1`. Which plots all the files in the directory.

The output will be saved in the `./output/visualization/plot_average_distance` directory.


#### Plot Crowdeness

`plot_crowdedness.py` plots the histogram of the airports crowdedness. Run the following command:

```bash
python amelia_data_tools/visualization_tools/plot_crowdedness.py --base_dir [base_dir] --traj_version [traj_version] --dpi [dpi] --num_files [num_files] --process
```

Where:

- `[base_dir]` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `[traj_version]` is the version of the trajectory file. By default it is `a10v08`.
- `[dpi]` is the resolution of the image. By default it is `600`.
- `[num_files]` is the number of files to plot. By default it is set to `-1`. Which plots all the files in the directory.
- `--process` is a flag to process the data. By default it is set to `False`.

The output will be saved in the `./output/visualization/plot_crowdedness` directory.

#### Plot Moving or Static Interpolated Agents

`plot_interp_stats.py` plots the agents' statistics with the interpolated points. Run the following command:

```bash
python amelia_data_tools/visualization_tools/plot_interp_stats.py --base_dir [base_dir] --traj_version [traj_version] --moving --dpi [dpi] --num_files [num_files]
```

Where:

- `[base_dir]` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `[traj_version]` is the version of the trajectory file. By default it is `a10v08`.
- `--moving` is a flag to plot taking into account only the moving agents. By default it is set to `False`.
- `[dpi]` is the resolution of the image. By default it is `600`.
- `[num_files]` is the number of files to plot. By default it is set to `-1`. Which plots all the files in the directory.

The output will be saved in the `./output/visualization/plot_interp_stats` directory.

#### Plot Motion Profiles

`plot_motion_profiles.py` plots the motion profiles of the agents, they might be `acceleration`, `speed` or `heading`. Run the following command:

```bash
python amelia_data_tools/visualization_tools/plot_motion_profiles.py --base_dir [base_dir] --traj_version [traj_version] --to_process --input_path [input_path] --motion_profile [motion_profile] --drop_interp --agent_type [agent_type] --dpi [dpi] --num_files [num_files]
```

Where:

- `[base_dir]` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `[traj_version]` is the version of the trajectory file. By default it is `a10v08`.
- `[to_process]` is the percentage of files to plot. By default it is set to `1.0`.
- `[input_path]` is the path to the input file. By default it is set to `./output/cahe/compute_motion_profiles`.
- `[motion_profile]` is the motion profile to plot. By default it is set to `acceleration`. The available options are `acceleration`, `speed` and `heading`.
- `--drop_interp` is a flag to drop the interpolated points. By default it is set to `False`.
- `[agent_type]` is the type of agent to plot. By default it is set to `aircraft`. Other options are `aircraft`, `vehicle`, `unknown` and `all`.
- `[dpi]` is the resolution of the image. By default it is `600`.

The output will be saved in the `./output/visualization/plot_motion_profiles` directory.

#### Plot Moving Agents Versus Stationary Agents

`plot_moving_agent_stats.py` plots the moving agents versus the stationary agents. Run the following command:

```bash
python amelia_data_tools/visualization_tools/plot_moving_agent_stats.py --base_dir [base_dir] --traj_version [traj_version] --dpi [dpi] --num_files [num_files]
```

Where:

- `[base_dir]` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `[traj_version]` is the version of the trajectory file. By default it is `a10v08`.
- `[dpi]` is the resolution of the image. By default it is `600`.
- `[num_files]` is the number of files to plot. By default it is set to `-1`. Which plots all the files in the directory.

The output will be saved in the `./output/visualization/plot_moving_agent_stats` directory.

#### Plot Sequence Lenghts

`plot_sequence_lengths.py` plots the sequence lengths of the agents. Run the following command:

```bash
python amelia_data_tools/visualization_tools/plot_sequence_lengths.py --base_dir [base_dir] --traj_version [traj_version] --dpi [dpi] --num_files [num_files]
```

Where:

- `[base_dir]` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `[traj_version]` is the version of the trajectory file. By default it is `a10v08`.
- `[dpi]` is the resolution of the image. By default it is `600`.
- `[num_files]` is the number of files to plot. By default it is set to `-1`. Which plots all the files in the directory.
