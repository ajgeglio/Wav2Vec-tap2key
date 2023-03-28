import numpy as np
import random
import os
import pandas as pd
from datasets import Audio, Dataset, DatasetDict, load_from_disk
import argparse
from timeit import default_timer as stopwatch
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_sample(example, title):
    tmp_sample = example['audio']['array']
    plt.plot(tmp_sample)
    plt.savefig(f"/home/ajgeglio/FutureGroup/{title}.png")

def plot_random_sample(dataset):
    fig, ax = plt.subplots(4,1, figsize=(15,12), tight_layout=True)
    subset = dataset.sort('key')
    time_x = np.linspace(0, sample_len / fs, num=sample_len)
    idx_ = np.random.randint(0,6192)
    for r in range(ax.shape[0]):
        # for c in range(ax.shape[1]):
        sample = subset['train'][idx_+r]['audio']['array']
        letter = subset['train'][idx_+r]['key']
        for chan in range(sample.shape[0]):
            ax[r].plot(time_x, sample[chan], label=f"chan {chan}")

            ax[r].set_title(f"idx {idx_+r} key {letter}", y=0.8)
    plt.legend()
    # plt.xticks(rotation=90)
    plt.savefig(f"/home/ajgeglio/FutureGroup/Sample_plots/sample_{idx_}.png")

def plot_split(dataset, title):
    # Three way split
    fig, ax = plt.subplots(figsize=(10,7))
    counts_train = np.unique(dataset['train']['label'], return_counts=True)
    counts_valid = np.unique(dataset['validation']['label'], return_counts=True)
    counts_eval = np.unique(dataset['evaluation']['label'], return_counts=True)
    tr_lbls = [id2label[str(i)] for i in counts_train[0]]
    va_lbls = [id2label[str(i)] for i in counts_test[0]]
    ev_lbls = [id2label[str(i)] for i in counts_test[0]]
    plt.barh(tr_lbls, width=counts_train[1], label='Train')
    plt.barh(va_lbls, width=counts_test[1], label='Test')
    plt.xlabel('count')
    plt.legend()
    # plt.xticks(rotation=90)
    plt.title(f"Total Sample count: {(counts_train[1]+counts_valid[1]+counts_eval[1]).sum()}")
    plt.savefig(f"/home/ajgeglio/FutureGroup/{title}.png")


if __name__ == "__main__":
    start_time = stopwatch()
    t = time.localtime()
    current_time = time.strftime("%b-%d-%Y-%H:%M", t)

    parser = argparse.ArgumentParser(description='plotting using matplotlib')
    # Logical operators for configuration of data
    parser.add_argument('--dataset_dir', help="directory of the hugging face dataset", dest="dataset_dir", default='/work/ajgeglio/Tap_Data/11.All_Dataset_96k')
    parser.add_argument('--plot_split', help='plot the dataset dev split', action="store_true")
    parser.add_argument('--plot_sample', help='plot signal', action="store_true")
    parser.add_argument('--plot_performance', help='plot performance', action="store_true")
    args = parser.parse_args()


    tap_dataset = load_from_disk(args.dataset_dir)

    if args.plot_split:
        plot_split(tap_dataset, f'split_{current_time}')

    if args.plot_sample:
        example = tap_dataset[64]
        plot_sample(example, f'sample_{current_time}')

    if args.plot_performance:
        fig, ax = plt.subplots(3,1,figsize=(10,7), tight_layout=True)
        perf1 = [0.74, 0.77, 0.78, 0.82, 0.85, 0.85, 0.81]
        key1 = [
                '50% data use + avr channel', 
                '60% data use + avr channel', 
                '70% data use + avr channel', 
                '80% data use + avr channel', 
                '90% data use + avr channel',
                '100% data use + avr channel', 
                'Before group data collection'
                ]

        perf2 = [0.63, 0.87, 0.94, 0.95]
        key2 = [
                'oversample + max absolute',
                'oversample + avr channel', 
                'oversample + flatten to 8x len',
                'oversample + interleave to 8x len',
                ]

        perf3 = [0.95, 0.95, 0.94, 0.86, 0.86]
        key3 = ['48 khz, len 32,768',
                '24 khz, len 16,384',
                '16 khz, len 10,928',
                '16 khz, top only',
                '16 khz, bottom only']
        df1 = pd.DataFrame(np.c_[perf1, key1], columns=["perf", "key"]).astype({'perf': 'float32'})
        df2 = pd.DataFrame(np.c_[perf2, key2], columns=["perf", "key"]).astype({'perf': 'float32'})
        df3 = pd.DataFrame(np.c_[perf3, key3], columns=["perf", "key"]).astype({'perf': 'float32'})
        title = 'perf_summary'
        ax[0].plot(df1.perf, df1.key, marker='x',zorder=4, color='k', markersize=10)
        ax[0].barh(width=df1.perf, y=df1.key, color='cornflowerblue', zorder=3, edgecolor='k')
        ax[0].set_xlim(0.5,1)
        ax[0].set_title('Avr channel, varying amounts of data')
        ax[1].plot(df2.perf, df2.key, marker='x',zorder=4, color='k', markersize=10)
        ax[1].barh(width=df2.perf, y=df2.key, color='cornflowerblue', zorder=3, edgecolor='k')
        ax[1].set_xlim(0.5,1)  
        ax[1].set_title('Using all data, testing differnt channel configuration')       
        ax[2].plot(df3.perf, df3.key, marker='x',zorder=4, color='k', markersize=10)
        ax[2].barh(width=df3.perf, y=df3.key, color='cornflowerblue', zorder=3, edgecolor='k')
        ax[2].set_xlim(0.5,1)  
        ax[2].set_title('Interleaving channels, testing differnt sample rates')   
        plt.savefig(f"{title}.png")
        # plt.xlabel('Experiment')
        # plt.ylabel('F1 score')
        # props = dict(boxstyle='round', alpha=0.7)
        # ax.text(1,0.9,'Testing with Average channel \nand varying proportion of data', verticalalignment='center', bbox=props)
        # plt.xticks(rotation=0)
        # # rect = patches.Rectangle((0.5, 0.95), 6, -1, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        # for x, y, p in zip(df.index, df.perf, df.key):
        #     plt.text(x-0.12, 0.51, p, rotation=90, size=14,style='oblique', alpha=0.7)
        # plt.title(f"Total Sample count: {(counts_train[1]+counts_valid[1]+counts_eval[1]).sum()}")