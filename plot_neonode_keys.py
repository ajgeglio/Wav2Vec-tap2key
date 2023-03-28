import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def concat_keys(key_dirs):
    all_keys = np.empty([0,6], dtype=str)
    for i in range(len(key_dirs)):
        key1 =  np.loadtxt(key_dirs[i], delimiter=',', dtype=str)
        key1 = np.char.strip(key1)
        all_keys = np.concatenate([all_keys, key1])
    return all_keys

def plot_keys():
    # key_dirs = [
    #         # '/home/ajgeglio/FutureGroup/Tap_Data/abc/real.csv', 
    #         # '/home/ajgeglio/FutureGroup/Tap_Data/abc/real (copy).csv',
    #         # '/home/ajgeglio/FutureGroup/Tap_Data/abc/real (another copy).csv', 
    #         '/home/ajgeglio/FutureGroup/Tap_Data/abc/real (3rd copy).csv'
    #         ]
    # all_keys = concat_keys(key_dirs)
    key_dir = '/work/ajgeglio/Tap_Data/abc/key_label_location.csv'
    raw_dir = '/work/ajgeglio/Tap_Data/abc/qwerty_locations.csv'
    key_loc = np.array(pd.read_csv(key_dir,header=None))
    raw_loc = np.array(pd.read_csv(raw_dir,header=None))
    fig, ax = plt.subplots()
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
    ax.axis('equal')
    # ax.invert_xaxis()
    ax.set_xlim(3250,250)
    ax.set_ylim(1500,500)
    z = key_loc[:,0]
    y = key_loc[:,1]
    n = key_loc[:,2]

    zr = raw_loc[:,3]
    yr = raw_loc[:,4]
    nr = raw_loc[:,6]

    ax.scatter(z, y, marker='s', sizes = [300],color='lightskyblue')
    ax.scatter(z, y, marker='.', sizes = [5], color='k')
    ax.scatter(zr, yr, marker='.', sizes = [3], color='red')

    for i, txt in enumerate(n):
        ax.annotate(txt, (z[i], y[i]))
    plt.savefig(f"key_locs2.png")


