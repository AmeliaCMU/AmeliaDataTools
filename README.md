# AmeliaDataTools

This repository contains the data tools code for the model introduced in the paper below:

***Amelia: A Large Dataset and Model for Airport Surface Movement Forecasting [[paper](https://arxiv.org/pdf/2407.21185)]***

[Ingrid Navarro](https://navars.xyz) *, [Pablo Ortega-Kral](https://paok-2001.github.io) *, [Jay Patrikar](https://www.jaypatrikar.me) *, Haichuan Wang,
Zelin Ye, Jong Hoon Park, [Jean Oh](https://cmubig.github.io/team/jean_oh/) and [Sebastian Scherer](https://theairlab.org/team/sebastian/)

## Overview

**AmeliaDataTools**: Set of tools to process and visualize the Amelia dataset. It includes tools to compute agent counts, limits, motion profiles, and sequence lengths. It also includes tools to visualize agent statistics, trajectories, average distance, crowdedness, moving or static interpolated agents, motion profiles, and moving agents versus stationary agents. It includes tools to get crowdedness and movement statistics.

## Pre-requisites

### Dataset

To run this repository, you first need to download the amelia dataset. Follow the instructions [here](https://ameliacmu.github.io/amelia-dataset/) to download the dataset.

Once downloaded, create a symbolic link into  `datasets`:

```bash
cd datasets
ln -s /path/to/amelia .
```

### Installation

Make sure that you have [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.

**Recommended:** Use the  [`install.sh`](https://github.com/AmeliaCMU/AmeliaScenes/blob/main/install.sh) to download and install the Amelia Framework:

```bash
chmod +x install.sh
./install.sh amelia
```

This will create a conda environment named `amelia` and install all dependencies.

Alternatively, refer to [`INSTALL.md`](https://github.com/AmeliaCMU/AmeliaScenes/blob/main/INSTALL.md) for manual installation.

**Note:** AmeliaDataTools requires the Amelia dataset and AmeliaScenes dependencies to run, only refer to AmeliaDataTools' and AmeliaScenes' installation.

## How to use

Activate your amelia environment (**Please follow the installation instructions above**):

```bash
conda activate amelia
```

### Trajectory Tools

#### Compute Agent Counts

`compute_agent_counts.py` computes the number of agents per timestep. Run the following command:

```bash
python amelia_datatools/trajectory_tools/compute_agent_counts.py
```

The output will be saved in the `./output/stats/compute_agent_counts` directory.

#### Compute Limits

`compute_limits.py` computes the limits of the airports and updates them. Run the following command:

```bash
python amelia_datatools/trajectory_tools/compute_limits.py
```

The output will be saved in the `./output/cache/compute_limits` directory.

#### Compute Motion Profiles

`compute_motion_profiles.py` computes the motion profiles of the agents. Run the following command:

```bash
python amelia_datatools/trajectory_tools/compute_motion_profiles.py \
        --base_dir <base_dir> \
        --traj_version <traj_version> \
        --to_process <to_process> \
        --drop_interp \
        --agent_type <agent_type>
```

Where:

- `KBOS_26_1672610400_critical_ego` and `KBOS_26_1672621200_critical_ego` are the directories with the specifications configured in the `example_kbos_critical.yaml` file.
- `kbos_scene_*.png` files are the images with the predictions generated by the model for each scene.
- `<base_dir>` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `<traj_version>` is the version of the trajectory file. By default it is `a10v08`.
- `<to_process>` is the percentage of files to process. By default it is set to `1.0`.
- `--drop_interp` is a flag to drop the interpolated points. By default it is set to `False`.

#### Compute Sequence Lengths

`compute_sequence_lengths.py` computes the sequence lengths of the agents by airport. Run the following command:

```bash
python amelia_datatools/trajectory_tools/compute_sequence_lengths.py
```

The output will be saved in the `./output/stats/compute_sequence_lengths` directory.

### Visualization Tools

#### Plot Agent Statistics

`plot_agent_stats.py` plots agents' statistics, it counts the number of agents per timesteps as well as the number of agents by type. Run the following command:

```bash
python amelia_datatools/visualization_tools/plot_agent_statistics.py \
    --base_dir <base_dir> \
    --traj_version <traj_version> \
    --dpi <dpi> \
    --num_files <num_files>
```

where:

- `<base_dir>` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `<traj_version>` is the version of the trajectory file. By default it is `a10v08`.
- `<dpi>` is the resolution of the image. By default it is `600`.
- `<num_files>` is the number of files to plot. By default it is set to `-1`. Which plots all the files in the directory.

The output will be saved in the `./output/visualization/plot_agent_stats` directory.

#### Plot All Trajectories

`plot_all_trajectories.py` plots the trajectories of the agents from the dataset onto the map. Run the following command:

```bash
python amelia_datatools/visualization_tools/plot_all_trajectories.py \
    --base_dir <base_dir> \
    --traj_version <traj_version> \
    --to_process <to_process> \
    --drop_interp \
    --dpi <dpi>
```

where:

- `<base_dir>` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `<traj_version>` is the version of the trajectory file. By default it is `a10v08`.
- `<to_process>` is the percentage of files to plot. By default it is set to `1.0`.
- `--drop_interp` is a flag to drop the interpolated points. By default it is set to `False`.
- `<dpi>` is the resolution of the image. By default it is `600`.

The output will be saved in the `./output/visualization/plot_all_trajectories` directory.

#### Plot Average Distance

`plot_average_distance.py` plots the average distance between agents by timestamp. Run the following command:

```bash
python amelia_datatools/visualization_tools/plot_average_distance.py \
    --base_dir <base_dir> \
    --traj_version <traj_version> \
    --dpi <dpi> \
    --num_files <num_files>
```

where:

- `<base_dir>` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `<traj_version>` is the version of the trajectory file. By default it is `a10v08`.
- `<dpi>` is the resolution of the image. By default it is `600`.
- `<num_files>` is the number of files to plot. By default it is set to `-1`. Which plots all the files in the directory.

The output will be saved in the `./output/visualization/plot_average_distance` directory.

#### Plot Crowdedness

`plot_crowdedness.py` plots the histogram of the airports crowdedness. Run the following command:

```bash
python amelia_datatools/visualization_tools/plot_crowdedness.py \
    --base_dir <base_dir> \
    --traj_version <traj_version> \
    --dpi <dpi> \
    --num_files <num_files> \
    --process
```

where:

- `<base_dir>` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `<traj_version>` is the version of the trajectory file. By default it is `a10v08`.
- `<dpi>` is the resolution of the image. By default it is `600`.
- `<num_files>` is the number of files to plot. By default it is set to `-1`. Which plots all the files in the directory.
- `--process` is a flag to process the data. By default it is set to `False`.

The output will be saved in the `./output/visualization/plot_crowdedness` directory.

#### Plot Moving or Static Interpolated Agents

`plot_interp_stats.py` plots the agents' statistics with the interpolated points. Run the following command:

```bash
python amelia_datatools/visualization_tools/plot_interp_stats.py \
    --base_dir <base_dir> \
    --traj_version <traj_version> \
    --moving \
    --dpi <dpi> \
    --num_files <num_files>
```

where:

- `<base_dir>` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `<traj_version>` is the version of the trajectory file. By default it is `a10v08`.
- `--moving` is a flag to plot taking into account only the moving agents. By default it is set to `False`.
- `<dpi>` is the resolution of the image. By default it is `600`.
- `<num_files>` is the number of files to plot. By default it is set to `-1`. Which plots all the files in the directory.

The output will be saved in the `./output/visualization/plot_interp_stats` directory.

#### Plot Motion Profiles

`plot_motion_profiles.py` plots the motion profiles of the agents, they might be `acceleration`, `speed` or `heading`. Run the following command:

```bash
python amelia_datatools/visualization_tools/plot_motion_profiles.py \
    --base_dir <base_dir> \
    --traj_version <traj_version> \
    --to_process \
    --input_path <input_path> \
    --motion_profile <motion_profile> \
    --drop_interp \
    --agent_type <agent_type> \
    --dpi <dpi> \
    --num_files <num_files>
```

where:

- `<base_dir>` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `<traj_version>` is the version of the trajectory file. By default it is `a10v08`.
- `<to_process>` is the percentage of files to plot. By default it is set to `1.0`.
- `<input_path>` is the path to the input file. By default it is set to `./output/cache/compute_motion_profiles`.
- `<motion_profile>` is the motion profile to plot. By default it is set to `acceleration`. The available options are `acceleration`, `speed` and `heading`.
- `--drop_interp` is a flag to drop the interpolated points. By default it is set to `False`.
- `<agent_type>` is the type of agent to plot. By default it is set to `aircraft`. Other options are `aircraft`, `vehicle`, `unknown` and `all`.
- `<dpi>` is the resolution of the image. By default it is `600`.

The output will be saved in the `./output/visualization/plot_motion_profiles` directory.

#### Plot Moving Agents Versus Stationary Agents

`plot_moving_agent_stats.py` plots the moving agents versus the stationary agents. Run the following command:

```bash
python amelia_datatools/visualization_tools/plot_moving_agent_stats.py \
    --base_dir <base_dir> \
    --traj_version <traj_version> \
    --dpi <dpi> \
    --num_files <num_files>
```

where:

- `<base_dir>` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `<traj_version>` is the version of the trajectory file. By default it is `a10v08`.
- `<dpi>` is the resolution of the image. By default it is `600`.
- `<num_files>` is the number of files to plot. By default it is set to `-1`. Which plots all the files in the directory.

The output will be saved in the `./output/visualization/plot_moving_agent_stats` directory.

#### Plot Sequence Lengths

`plot_sequence_lengths.py` plots the sequence lengths of the agents. Run the following command:

```bash
python amelia_datatools/visualization_tools/plot_sequence_lengths.py \
    --base_dir <base_dir> \
    --traj_version <traj_version> \
    --dpi <dpi> \
    --num_files <num_files>
```

where:

- `<base_dir>` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `<traj_version>` is the version of the trajectory file. By default it is `a10v08`.
- `<dpi>` is the resolution of the image. By default it is `600`.
- `<num_files>` is the number of files to plot. By default it is set to `-1`. Which plots all the files in the directory.

### Data Tools

#### Get Crowdedness

`get_peak_hour.py` gets the peak crowdedness of the airports by hour. Run the following command:

```bash
python amelia_datatools/todo/get_peak_hour.py \
  --base_dir <base_dir> \
  --traj_version <traj_version> \
  --output_dir <output_dir> \
  --airport <airport> \
  --parallel
```

where:

- `<base_dir>` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `<traj_version>` is the version of the trajectory file. By default it is `a10v08`.
- `<output_dir>` is the path to the output directory. By default it is `./output/crowdedness`.
- `<airport>` is the airport to process. By default it is set to `all`.
- `--parallel` is a flag to process the data in parallel. By default it is set to `True`.

The output will be saved in the `./output/crowdedness` directory.

#### Get Movement Statistics

`get_movement_stats.py` gets the movement statistics of the agents. Run the following command:

```bash
python amelia_datatools/todo/get_movement_stats.py \
    --base_dir <base_dir> \
    --traj_version <traj_version> \
    --output_dir <output_dir> \
    --airport <airport> \
    --parallel
```

where:

- `<base_dir>` is the path to the directory where the data is stored. By default it is `./datasets/amelia`.
- `<traj_version>` is the version of the trajectory file. By default it is `a10v08`.
- `<output_dir>` is the path to the output directory. By default it is `./output/movement`.
- `<airport>` is the airport to process. By default it is set to `all`.
- `--parallel` is a flag to process the data in parallel. By default it is set to `True`.

The output will be saved in the `./output/movement` directory.

## BibTeX

If you find our work useful in your research, please cite us!

```bibtex
@inbook{navarro2024amelia,
  author = {Ingrid Navarro and Pablo Ortega and Jay Patrikar and Haichuan Wang and Zelin Ye and Jong Hoon Park and Jean Oh and Sebastian Scherer},
  title = {AmeliaTF: A Large Model and Dataset for Airport Surface Movement Forecasting},
  booktitle = {AIAA AVIATION FORUM AND ASCEND 2024},
  chapter = {},
  pages = {},
  doi = {10.2514/6.2024-4251},
  URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2024-4251},
  eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2024-4251},
}
```
