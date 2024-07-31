import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def impute(seq, seq_len):
    """ Imputes missing data by doing simple linear interpolation. """
    # TODO: figure out best way to fill in missing data. For now; linear interpolation
    # Create a list from starting frame to ending frame in agent sequence
    conseq_frames = set(range(int(seq[0, 0]), int(seq[-1, 0])+1))
    # Create a list of the actual frames in the agent sequence. There may be missing 
    # data from which we need to interpolate.
    actual_frames = set(seq[:, 0])
    # Compute the difference between the lists. The difference represents the missing
    # data points
    missing_frames = list(sorted(conseq_frames - actual_frames))
    # print(missing_frames)

    # Insert nan rows on where the missing data is. Then, interpolate. 
    if len(missing_frames) > 0:
        seq = pd.DataFrame(seq)
        for missing_frame in missing_frames:
            df1 = seq[:missing_frame]
            df2 = seq[missing_frame:]
            df1.loc[missing_frame] = np.nan
            seq = pd.concat([df1, df2])

        seq = seq.interpolate(method='linear').to_numpy()[:seq_len]
    return seq

def plot_scene(seq, soc_seq, idx, filename):
    N, T, D = seq.shape

    fig, axs = plt.subplots(2, N)#, figsize=(2*5, (N+1) * 10))
    for n in range(N):
        x, y = seq[n, :, idx]
        axs[0, n].plot(x, y, color="blue")

        x, y = soc_seq[n, n, :, 0], soc_seq[n, n, :, 1]
        axs[1, n].plot(x, y, color="green")

        for nn in range(N):
            if nn == n:
                continue
            
            x, y = seq[nn, :, idx]
            axs[0, n].plot(x, y, color="red")

            sx, sy = soc_seq[n, nn, :, 0], soc_seq[n, nn, :, 1]
            axs[1, n].plot(sx, sy, color="orange")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    # plt.show()
    # plt.close()