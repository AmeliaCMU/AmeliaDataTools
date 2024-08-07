import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import math

from amelia_datatools.utils import common as C


def get_single_histogram(data, airport, color='#773344', bins=179, ax=None, limits=(None, None, None), title=None,
                         x_label='Value', y_label='Frequency', alpha=0.25):
    ax.set_title(f'{title}')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # data = data[data != 0]

    if (alpha == 1):
        freq, bins, patches = ax.hist(data, bins=bins, color=color,
                                      alpha=alpha, density=False, label=airport, zorder=10)
    else:
        freq, bins, patches = ax.hist(data, bins=bins, color=color,
                                      alpha=alpha, density=False, label=airport, zorder=0)
    min_lim, max_lim, step = limits
    if (min_lim == None):
        min_lim, max_lim = math.floor(min(bins)), math.floor(max(bins))
        plt.xlim([min_lim, max_lim])
    else:
        plt.xlim([min_lim, max_lim])
        xticks = np.arange(math.floor(min_lim), math.ceil(max_lim), step)
        plt.xticks(xticks)


def create_histograms():
    out_dir = f"{C.OUTPUT_DIR}/count"
    os.makedirs(out_dir, exist_ok=True)
    airports = C.AIRPORT_COLORMAP.keys()
    target = 'kbos'
    plotting_colors = {
        "ksea": "#682D63",
        "kewr": "#519872",
        "kmdw": "#0072bb",
        "kbos": "#ca3c25"}
    name = {'total': 'Total', '0': 'Aircraft', '1': 'Vehicle', '2': 'Unknown'}
    for key, value in name.items():
        fig, histogram_plot = plt.subplots()
        for airport in airports:
            if (airport == target):
                alpha = 1.0
            else:
                alpha = 0.35
            file = f"{out_dir}/{airport}.pkl"
            try:
                with open(file, 'rb') as f:
                    data = pickle.load(f)
            except:
                continue
            # breakpoint()
            arr = data[key]

            get_single_histogram(
                arr,
                airport=airport,
                color=plotting_colors[airport],
                ax=histogram_plot,
                title=f'{value}',
                x_label=f'Sequence Length',
                alpha=alpha,
                # bins = (arr.max() // 10)
                bins=int(arr.max())
            )

        plt.grid()
        plt.tight_layout()
        plt.legend()
        plt.show(block=False)
        plt.savefig(f'{out_dir}/all_{key}.png', dpi=C.DPI)
        plt.close()


if __name__ == '__main__':
    create_histograms()
