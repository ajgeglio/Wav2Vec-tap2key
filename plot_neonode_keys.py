import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def concat_keys(key_dirs):# not used currently
    all_keys = np.empty([0,6], dtype=str)
    for i in range(len(key_dirs)):
        key1 =  np.loadtxt(key_dirs[i], delimiter=',', dtype=str)
        key1 = np.char.strip(key1)
        all_keys = np.concatenate([all_keys, key1])
    return all_keys

def plot_keys():
    key_dir = '/work/ajgeglio/Tap_Data/13.key_centers/key-tap-locs.csv'
    raw_dir = '/work/ajgeglio/Tap_Data/13.key_centers/qwerty_locations.csv'
    key_loc = np.array(pd.read_csv(key_dir,header=None))
    key_loc = key_loc[key_loc[:,2] != 'none']
    raw_loc = np.array(pd.read_csv(raw_dir,header=None))
    fig, ax = plt.subplots(dpi=600)
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 3500, 500)
    minor_ticks = np.arange(0, 3500, 50)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # # And a corresponding grid
    # ax.grid(which='both')
    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', 3250))
    ax.spines['bottom'].set_position(('data', 1750))
    ax.axis('equal')
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlim(3500,-100)
    ax.set_ylim(2600,0)
    z = key_loc[:,0]
    y = key_loc[:,1]
    n = key_loc[:,2]

    zr = raw_loc[:,3]
    yr = raw_loc[:,4]
    nr = raw_loc[:,6]

    # ax.scatter(z, y, marker='s', sizes = [300],color='lightskyblue')
    ax.scatter(z, y, marker='.', sizes = [5], color='k')
    # ax.scatter(zr, yr, marker='.', sizes = [3], color='red')

    # for i, txt in enumerate(n):
    #     ax.annotate(txt, (z[i], y[i]))
    plt.savefig(f"./key_locs2.png")


plot_keys()