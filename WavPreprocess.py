# -*- coding: utf-8 -*-
"""
Preprocess Recorded tap data with tap detection and neonode mapping to key labels

The first arg.INPUT is the directory of the data. To work properly the folder must be:
    01.acoustic --- this is the folder where the original recordings are
    02.neonode --- this is where the raw neonode csv is

The current os.walk function requires the following folder structure:

    01.acoustic --- this is the folder where the original recordings are
    02.neonode --- this is where the raw neonode csv is


TO RUN... 

##########################  EXAMPLE ######################################

WHEN YOU WANT TO JUST PLOT WITHOUT SAMPLING - RECCOMENDED TO DO FIRST
python3 WavPreprocess.py --dir '/work/ajgeglio/Tap_Data/Tony_01_25_23_data' 
--idx 9 --latency -0.14 --peak_height 0.0025 --plot

WHEN YOU WANT TO SAMPLE AND STORE FILES
python3 WavPreprocess.py --dir '/work/ajgeglio/Tap_Data/Tony_01_25_23_data' 
--idx 9 --latency -0.14 --peak_height 0.0025 --plot --selectivity 0.134 --sample

#####################################################################

The OUTPUT for each recording sould be a .wav file for each tap and 1 csv file of label data

Created on Mon Oct 24 12:37:03 2022

@author: Anthony.Geglio
"""

import os
import pandas as pd
import numpy as np
import datetime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('default')
from scipy.io import wavfile
from scipy import signal
from sklearn.neighbors import KNeighborsClassifier
import argparse
import re
from timeit import default_timer as stopwatch

def list_files(dir_):
    roots_ = []
    files_ = []
    for root, dirs, files in os.walk(dir_):
        roots_.append(root)
        files_.append(files)
    roots_.sort()
    files_ = [item for sublist in files_ for item in sublist]
    wav_files = [x for x in files_ if '.wav' in x]
    wav_files.sort()
    neonode_files = [x for x in files_ if 'nnode.csv' in x]
    return roots_, wav_files, neonode_files

def list_paths(dir_):
    wav_file_paths = []
    neonode_file_paths = []
    for dirpath, subdirs, files in os.walk(dir_):
        for x in files:
            if re.findall('\d+\.wav', x):
                wav_file_paths.append(os.path.join(dirpath, x))
            elif x.endswith("nnode.csv"):
                neonode_file_paths.append(os.path.join(dirpath, x))
    wav_file_paths.sort()
    # print(wav_file_paths)
    # quit()
    neonode_file_paths.sort()
    return wav_file_paths, neonode_file_paths

def dir_to_array2(wav_file_list, idx):
    fs, snd = wavfile.read(wav_file_list[idx])
    time_x = np.linspace(0, len(snd) / fs, num=len(snd))
    init = wav_file_list[idx]
    init = re.findall(r'(\d+)',init)[-7:]
    init = f"{init[6]}-{init[4]}-{init[5]} {init[0]}:{init[1]}:{init[2]}.{init[3]}"
    init = datetime.datetime.strptime(init, "%Y-%m-%d %H:%M:%S.%f")
    print("sound file start: ", init)
    snd = snd / 2.**32
    return fs, snd, time_x, init

def dir_to_node_loc():
# if you need to combine neonode files. Requires nnode file to end in nnode.csv
    if len(neonode_files)>1:
        df = pd.read_csv(f"{dir_}/02.neonode/{neonode_files[0]}", header=None)
        for i in range(1,len(neonode_files)):
            df_i = pd.read_csv(f"{dir_}/02.neonode/{neonode_files[i]}", header=None)
            df = pd.concat([df,df_i])
        df1 = df
    elif len(neonode_files)==0:
        print('Try renaming neonode file with *nnode.csv')
        quit()
    else: df1 = pd.read_csv(f"{dir_}/02.neonode/{neonode_files[0]}", header=None)
    df1.isetitem(0,pd.to_datetime(df1.iloc[:,0], format="%Y-%m-%d %H:%M:%S.%f"))
    df1 = df1[(df1.iloc[:,2]== '   Down')]  
    # df1['diff'] = (df1.iloc[:,0] - ini_record)/np.timedelta64(1,'s')
    diff_ = (df1.iloc[:,0] - ini_record)/np.timedelta64(1,'s')
    df1 = pd.concat([df1,diff_], axis=1)
    df1 = df1[df1.iloc[:,6]>0]
    df1 = df1[df1.iloc[:,6]<60]
    # df1 = df1[df1[5]<150]
    # d0 = np.abs(ar0).min()
    df1['label'] = knn.predict(list(zip(df1[3],df1[4])))
    df1 = df1[df1['label'] != 'none']
    ar = np.array(df1.iloc[:,[0,1,2,6,3,4,5,7]])
    # np.savetxt(f'{root}/{name_time}_nnode.csv', ar, fmt='%s', delimiter=',')
    return ar

def plot_avr_snd(   avr_signal, # input averaged sound time-domain matrix
                    linewidth):
    fig, ax = plt.subplots(1,1, sharex=True,sharey=True, figsize = (18,3), 
                           tight_layout=True, dpi=400)
    plt.suptitle(f"File start: {str(name_time)}")
    plt.subplots_adjust(hspace=0.1)
    ax.set_title('Average of 8 Channels', y=1.0, pad=-14)
    ax.plot(time_x, avr_signal, linewidth=linewidth)
    ax.vlines(peaks_times - behind/fs, -max_/4, max_/4, color='orange', linewidth=linewidth)
    ax.vlines(peaks_times + forward/fs ,-max_/4 ,max_/4, color='orange', linewidth=linewidth)
    ax.vlines(node_times, -max_/6, max_/6, color='red', linewidth = linewidth)
    ax.vlines(peaks_times, -max_/8, max_/8, color='k', linewidth = linewidth/2)
    ax.hlines(args.peak_height, start,stop, color='k', linewidth = linewidth)
    sample_num = []
    for i in range(len(peaks)):
        ax.annotate(i+1, (peaks_times[i], 0),fontsize=2)
        sample_num.append(i+1)
    for j, txt in enumerate(node_labels):
        ax.annotate((j+1,txt), (node_times[j], np.random.uniform(0.003, max_/2)),fontsize=1.5)
    ax.set_xlim(start,stop)
    plt.savefig(f"{dir_}/03.plots/{name_time}_avechan.png")    
    print(f"Plotted: {name_time} {len(node_times)} neonode taps")


def reshape3_(data): #takes input of average signal or 8 channels
    i = 1
    n = 0
    # This is how I control the selectivity (is the sample near a neonode label)
    for p, t in zip(peaks, peaks_times): #peaks is an index, p/fs is the time in decimal seconds
        ar0 = np.array(t - node_times) #only taking the ones near a label
        d0 = np.abs(ar0).min()
        try:
            if d0 <= args.selectivity:
                b = np.take(data, np.arange(p-behind,p+forward), axis=0)
                wavfile.write(f'{dir_}/05.wav_samples/{name_time}_{i:0>3}_8chan.wav', fs, b)
                n+=1
            else: print(f"did not create sample {i:0>3} is not near a label")
        except: 
            print(f'sample {i:0>3} not enough data in window')
            pass
        i+=1
    print(f"Sampled: {name_time} \n {n} files created, {i-1} peaks detected")

def plot_keys():
    fig, ax = plt.subplots()
    z = key_loc[:,0]
    y = key_loc[:,1]
    n = key_loc[:,2]
    ax.scatter(z, y)
    for i, txt in enumerate(n):
        ax.annotate(txt, (z[i], y[i]))
    plt.savefig(f"key_locs.png")

# Not used anymore
def label_encode(labels):
    encode_dic = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 
        'j':9, 'k':10, 'l':11, 'm':12, 'n':13, 'o':14, 'p':15, 'q':16, 'r':17, 's':18,
        't':19, 'u':20, 'v':21, 'w':22, 'x':23, 'y':24, 'z':25, '|':26, '_1':27, 
        '_2':28, '_3':29, '_4':30, '_5':31, '_6':32, 'none':33}
    return [encode_dic[label] for label in labels]

####################### HIGH PASS FILTER ##############################
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a
    
# Currently taking in the average signal
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y
######################################################################
if __name__ == "__main__":

    start_time = stopwatch()

    parser = argparse.ArgumentParser(description='Creates tap samples and labels from a wavfile of person recorded typing sentences\
                                                  and associated neonode data')
    parser.add_argument('--dir', help="directory of original wav files", required=True, dest="dir_")
    parser.add_argument('--idx',type=int, required=True, help=' integer index of .wav file', dest = "idx_")
    parser.add_argument('--plot', required=False, help='save plot to visually check labels', action="store_true")
    parser.add_argument('--latency',type=float, required=False, help='latency of the record start - used to line up with neonode labels', dest = "latency_", default=-0.15)
    parser.add_argument('--selectivity',type=float, required=False, help='only select samples this close to a node label (seconds)', dest = "selectivity", default=.1)
    parser.add_argument('--peak_height',type=float, required=True, help='the amplitude height threshold for tap detection', dest = "peak_height", default=.0025)
    parser.add_argument('--sample', help="generate samples and labels", required=False, action="store_true")
    args = parser.parse_args()
    
    # Sort out the Files
    dir_ = args.dir_
    # Center locations of key taps
    key_dir = '/work/ajgeglio/Tap_Data/13.key_centers/key-tap-locs.csv'
    if not os.path.exists(dir_ + '/' + "03.plots"):
        os.makedirs(dir_ + '/' + "03.plots")
    if not os.path.exists(dir_ + '/' + "04.labels"):
        os.makedirs(dir_ + '/' + "04.labels")
    if not os.path.exists(dir_ + '/' + "05.wav_samples"):
        os.makedirs(dir_ + '/' + "05.wav_samples")
    
    _, wav_files, neonode_files = list_files(dir_)
    wav_file_paths, neonode_file_paths = list_paths(dir_)

    ################# MAP NEONODE LOCATION TO KEY LABEL ################
    # ## this is for labeling. KNN maps the xy node loc to a key label. 
    # ## Then you can use predict to map future nnode hits
    key_loc = np.array(pd.read_csv(key_dir,header=None))
    # plot_keys(dir_)
    # quit()
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(key_loc[:,:2],key_loc[:,2])

    #%%
    ##################### DEFINE RAW 8-CHANNEL MATRIX ###########################
    fs, wav_matrix, time_x, ini_record = dir_to_array2( wav_file_list = wav_file_paths,
                                                        idx = args.idx_) # in the list of 01.acoustic

    name_time = str(ini_record).replace(':','.').replace(' ','.')
    ini_record = ini_record - datetime.timedelta(seconds = args.latency_)
    #################### Sound Attributes ##########################
    record_len = len(time_x)/fs
    end_record = ini_record + datetime.timedelta(seconds = record_len)
    #################### neonode data for plotting ###############################
    neonode_arr = dir_to_node_loc()
    node_times= neonode_arr[:,3]
    node_labels = neonode_arr[:,7]
    # enter the start and end in seconds
    # There is usually noise at the beginning which is why I clip at start
    start, stop = 0.05, record_len
    # start, stop = 0.05, 4
    s = int(start*fs)
    e = int(stop*fs)
    behind = 2192
    forward = 6000
    print(f"Time window being sampled: {forward+behind} seconds")
    chan_8 = wav_matrix[s:e]
    time_x = time_x[s:e]
    trace = chan_8.mean(1)
    trace = butter_highpass_filter(data=trace, cutoff=1200, fs=fs, order=3)
    max_ = trace.max()
    # peak_height = args.peak_height

    ############## PEAK DETECTION ##############################
    # peaks = signal.find_peaks(  trace, 
    #                             threshold = max_/64,
    #                             distance=22000, # 4 taps per second is 0.25s*96000=24000 
    #                             height=max_/24,
    #                             prominence =max_/24
    #                             )[0]
    peaks = signal.find_peaks(trace, 
                            #   threshold = max_/16,
                              distance=12000, # 4 taps per second is 0.25s*96000=24000 
                              height = args.peak_height,
                            #   prominence =max_/2
                              )[0]
    print(f'Detected {len(peaks)} peaks')
    peaks_times = time_x[peaks]
    ###################### PLOTTING ##################################
    if args.plot:
        plot_avr_snd(avr_signal = trace, linewidth = 0.3)
    
    if args.sample:

        labels = np.c_[neonode_arr[:,[0,7,4,5,6]]]
        reshape3_(chan_8) # Saves out all of the tap samples
        if not os.path.exists(f'{dir_}/04.labels/{name_time}_label.csv'):
            np.savetxt(f'{dir_}/04.labels/{name_time}_label.csv', labels, fmt='%s', delimiter=',')
            print(len(neonode_arr[:,7]), "labels created")
        else: print('DID NOT OVERWRITE LABEL FILE')
    
    print(f"TOTAL TIME: {stopwatch() - start_time:.2f}")