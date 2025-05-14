import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm

from amelia_datatools.utils import common as C
from amelia_datatools.utils import utils as U
from amelia_datatools.utils import utils
from plotnine import (
    ggplot, aes, geom_col, scale_fill_manual,
    labs, theme_minimal, theme, element_text, geom_text, theme_bw
)


def plot(base_dir: str, traj_version: str, dpi: int, num_files: int,  output_dir: str):
    input_dir = os.path.join(base_dir, f"traj_data_{traj_version}", 'raw_trajectories')
    output_dir = os.path.join(f"{output_dir}", utils.get_file_name(__file__))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory in: {output_dir}")

    plt.rcParams['font.size'] = 6
    airports = U.get_airport_list()
    num_airports = len(airports)

    agent_counts = {}
    agent_types = {
        'Aircraft': [0 for _ in range(num_airports)],
        'Vehicle': [0 for _ in range(num_airports)],
        'Unknown': [0 for _ in range(num_airports)],
    }
    for i, airport in enumerate(tqdm(airports, desc="Airports")):
        airport_up = airport.upper()
        print(f"Running airport: {airport_up}")
        agent_counts[airport_up] = []
        airport_dir = os.path.join(input_dir, airport)
        traj_files = [os.path.join(airport_dir, f) for f in os.listdir(airport_dir) if f.endswith('.csv')]

        if num_files == -1:
            num_files = len(traj_files)

        for j, traj_file in enumerate(tqdm(traj_files)):
            if j > num_files:
                break

            data = pd.read_csv(traj_file)
            unique_IDs = data.ID.unique()
            agent_counts[airport_up].append(len(unique_IDs))
            for agent_ID in unique_IDs:
                traj_data = data[:][data.ID == agent_ID]
                agent_type = traj_data.Type.unique()
                if len(agent_type) > 1:
                    continue

                if agent_type == 0.0:
                    agent_types['Aircraft'][i] += 1
                elif agent_type == 1.0:
                    agent_types['Vehicle'][i] += 1
                else:
                    agent_types['Unknown'][i] += 1
    
    # Save agent_counts as cache
    cache_path = os.path.join(output_dir, "_cache")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    with open(os.path.join(cache_path, "agent_counts.pkl"), "wb") as f:
        pickle.dump(agent_counts, f)
    print(f"Agent counts cached at: {cache_path}")

    with open(os.path.join(cache_path, "agent_types.pkl"), "wb") as f:
        pickle.dump(agent_types, f)
    print(f"Agent types cached at: {cache_path}")

    # Plot unique agents
    total_agents = sum([np.asarray(c).sum() for c in agent_counts.values()])
    df = pd.DataFrame({
        "airport": list(agent_counts.keys()),
        "count":   [np.asarray(c).sum() for c in agent_counts.values()]
    })
    df["percentage"] = df["count"] / total_agents * 100
    airport_colors = {k: C.AIRPORT_COLORMAP[k.lower()] for k in df.airport}
    # --------- construct the plot ----------
    # Sort df and set airport as a categorical with the sorted order to ensure plotnine respects the order
    df = df.sort_values("count", ascending=False)
    df["airport"] = pd.Categorical(df["airport"], categories=df["airport"], ordered=True)
    
    g = (
        ggplot(df, aes(x="airport", y="count", fill="airport"))
        + geom_col(show_legend=False)
        + scale_fill_manual(values=airport_colors)
        + labs(
            x="Airport (ICAO)",
            y="# of Unique Agents",
            title="Unique Agents Per Airport"
        )
        + theme_bw()
        + theme(
            axis_text_x=element_text(rotation=45, color="dimgray", size=6),
            axis_text_y=element_text(color="dimgray",size=6),
            axis_title  =element_text(color="dimgray"),
            plot_title  =element_text(color="dimgray"),
            panel_grid_major_x=element_text(alpha=0),  # disables vertical grid lines
            panel_grid_minor_x=element_text(alpha=0),   # disables minor vertical grid lines
            # panel_grid_major_y=element_text(alpha=0),  # disables horizontal grid lines
        )
        + geom_col(show_legend=False)
        + geom_text(
            aes(label="count"),
            va='bottom',
            format_string='{:.0f}',
            size=3.5,
            color='dimgray'
        )
    )
    g.save(
        os.path.join(output_dir, "unique_agents.png"), height=3, width=7, dpi=dpi 
    )
    
    
    # Plot agent type
    fig, ax = plt.subplots()
    fontcolor = 'dimgray'
    airports = agent_counts.keys()
    bar_colors = [C.AIRPORT_COLORMAP[airport.lower()] for airport in airports]

    # the label locations
    x = np.arange(len(airports))
    # the width of the bars
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    color = {'Aircraft': 'darkblue', 'Vehicle': 'darkred', 'Unknown': 'darkgreen'}
    for multiplier, (attribute, measurement) in enumerate(agent_types.items()):
        offset = width * multiplier
        rects = ax.bar(
            x + offset, measurement, width, label=attribute, alpha=0.6, color=color[attribute], align='center')
        ax.bar_label(rects, padding=6, fontsize=5, color=fontcolor, label_type='edge')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(x + width, airports)
    ax.legend(loc='upper right', ncols=3)
    # ax.set_ylim(0, 250)

    ax.set_ylabel('Num. of Agents', color=fontcolor, fontsize=10)
    ax.set_title('Total Num. of Agents per Type', color=fontcolor, fontsize=15)
    ax.tick_params(color=fontcolor, labelcolor=fontcolor)
    for spine in ax.spines.values():
        spine.set_edgecolor(fontcolor)

    # ax.legend()
    plt.savefig(f"{output_dir}/agent_types.png", dpi=dpi, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '--base_dir', default=C.DATA_DIR, type=str, help='Input path')
    parser.add_argument('--traj_version', default=C.VERSION, type=str)
    parser.add_argument('--dpi', type=int, default=C.DPI)
    parser.add_argument('--num_files', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=C.VIS_DIR)
    args = parser.parse_args()

    plot(**vars(args))
