'''
* A flexible program to generate Hugging Face Datasets
* Uses the labels (.csv) and samples (.wav) of captured taps with WavPreprocess.py
* This program a lot of flexibility to allow for experimentation with different data shapes, such as interleaving the channels, stacking, max-absulute value from each channel, and channel average. 
* These are defined in mapping functions. 
* Also, you can specify the proportion of total data for additional experimentation.
* The interleaving of the 8-channels were found to be the most effective representation, so this was incorperated into the preprocess
function in the training program 'Wav2vec2-tap2key'.

EXAMPLE

python create_dataset.py --sample_rate 16_000 --reshape_interleave --oversample

python create_dataset.py --oversample --encode --sample_rate 16_000

Note - In order to use a mapping function like reshape_interleave, you must specify a sample_rate, otherwise the dataset
only maps to the file location which is much more efficient, but does not contain the audio feature.

'''

import numpy as np
import random
import os
from datasets import Audio, Dataset, DatasetDict,  interleave_datasets, load_dataset
from audiomentations import Compose, AddGaussianNoise, Gain, PitchShift, TimeStretch, Shift
from transformers import AutoFeatureExtractor
import argparse
from timeit import default_timer as stopwatch
import matplotlib.pyplot as plt

def list_samples_labels(dir_):
    roots_ = []
    files_ = []
    for root, dirs, files in os.walk(dir_):
        roots_.append(root)
        files_.append(files)
    files_ = [item for sublist in files_ for item in sublist]
    label_files = [roots_[0]+'/'+x for x in files_ if 'label.csv' in x]
    label_files.sort()
    tap_files = [roots_[0]+'/'+x for x in files_ if '8chan.wav' in x]
    tap_files.sort()
    return label_files, tap_files

def concat_labels(label_files):
    label = np.empty([0,2], dtype=str)
    for i in range(len(label_files)):
        label1 =  np.loadtxt(label_files[i], delimiter=',', dtype=str)[:,:2]
        label = np.concatenate([label,label1])
    return label

def create_dataset(data_dir, test_ds_dir, seed_):
    label_files, tap_files = list_samples_labels(data_dir)
    label = concat_labels(label_files)
    ds = Dataset.from_dict(
        {"audio": tap_files, 'label': label[:,1]})
    ds = ds.class_encode_column("label")      
    # split twice and combine
    train_set = ds.train_test_split(    shuffle = True, 
                                        seed = seed_, 
                                        stratify_by_column ='label',
                                        test_size=0.3)
    test_set = train_set['test'].train_test_split(  shuffle = True, 
                                                    seed = seed_, 
                                                    stratify_by_column ='label',
                                                    test_size=0.5)
    ds_dev = DatasetDict({  'train'     : train_set['train'],
                            'validation': test_set['train']})
    ds_test = DatasetDict({ 'test'      : test_set['test']})
        
    print(f"Saving test dataset separately to: {test_ds_dir}")
    ds_test.save_to_disk(test_ds_dir)
    return ds_dev

def label_encoder(ds):
    try:
        labels = ds.features["label"].names
    except:
        labels = ds['train'].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    return label2id, id2label
    
def oversample_interleave(tap_dataset, seed_):
        dataset = tap_dataset['train']
        ds0 = dataset.filter(lambda dataset: dataset["label"]==0)
        ds1 = dataset.filter(lambda dataset: dataset["label"]==1)
        ds2 = dataset.filter(lambda dataset: dataset["label"]==2)
        ds3 = dataset.filter(lambda dataset: dataset["label"]==3)
        ds4 = dataset.filter(lambda dataset: dataset["label"]==4)
        ds5 = dataset.filter(lambda dataset: dataset["label"]==5)
        ds6 = dataset.filter(lambda dataset: dataset["label"]==6)
        ds7 = dataset.filter(lambda dataset: dataset["label"]==7)
        ds8 = dataset.filter(lambda dataset: dataset["label"]==8)
        ds9 = dataset.filter(lambda dataset: dataset["label"]==9)
        ds10 = dataset.filter(lambda dataset: dataset["label"]==10)
        ds11 = dataset.filter(lambda dataset: dataset["label"]==11)
        ds12 = dataset.filter(lambda dataset: dataset["label"]==12)
        ds13 = dataset.filter(lambda dataset: dataset["label"]==13)
        ds14 = dataset.filter(lambda dataset: dataset["label"]==14)
        ds15 = dataset.filter(lambda dataset: dataset["label"]==15)
        ds16 = dataset.filter(lambda dataset: dataset["label"]==16)
        ds17 = dataset.filter(lambda dataset: dataset["label"]==17)
        ds18 = dataset.filter(lambda dataset: dataset["label"]==18)
        ds19 = dataset.filter(lambda dataset: dataset["label"]==19)
        ds20 = dataset.filter(lambda dataset: dataset["label"]==20)
        ds21 = dataset.filter(lambda dataset: dataset["label"]==21)
        ds22 = dataset.filter(lambda dataset: dataset["label"]==22)
        ds23 = dataset.filter(lambda dataset: dataset["label"]==23)
        ds24 = dataset.filter(lambda dataset: dataset["label"]==24)
        ds25 = dataset.filter(lambda dataset: dataset["label"]==25)
        ds26 = dataset.filter(lambda dataset: dataset["label"]==26)
        ds27 = dataset.filter(lambda dataset: dataset["label"]==27)
        ds28 = dataset.filter(lambda dataset: dataset["label"]==28)
        ds29 = dataset.filter(lambda dataset: dataset["label"]==29)
        ds30 = dataset.filter(lambda dataset: dataset["label"]==30)
        ds31 = dataset.filter(lambda dataset: dataset["label"]==31)
        ds32 = dataset.filter(lambda dataset: dataset["label"]==32)
        ds33 = dataset.filter(lambda dataset: dataset["label"]==33)

        ds_oversample = interleave_datasets([ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8, ds9, ds10, ds11, ds12,
                                        ds13, ds14, ds15, ds16, ds17, ds18, ds19, ds20, ds21, ds22, ds23, ds24, ds25, 
                                        ds26, ds27, ds28, ds29, ds30, ds31, ds32, ds33], 
                                        probabilities=None, seed=seed_, stopping_strategy = 'all_exhausted')
        # ds_oversample = ds_oversample.map(augment_taps)
        # tap_dataset = tap_dataset
        ds_oversample = DatasetDict({       'train'         : ds_oversample,
                                            'validation'    : tap_dataset['validation'],
                                            # 'test'          : tap_dataset['test']
                                               })
        return ds_oversample


def preprocess_function(examples):
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    max_duration = 1  # seconds
    # audio_arrays = [x["array"] for x in examples["audio"]]
    audio_arrays = [np.reshape(x["array"], order='F', newshape=-1) for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True, 
    )
    return inputs

# This is the map function that interleaves the 8 channels into one channel 8x longer
def reshape_interleave(example):
    x = example["audio"]['array']
    new_x = np.reshape(x, newshape=-1, order="F")
    # print(new_x.shape)
    example["audio"]['array'] = new_x
    return example


def augment_taps(example, fs):
    # do augmentation dataset 
    augmentation = Compose([
            AddGaussianNoise(min_amplitude=0.00001, max_amplitude=0.001, p=0.8),
            # PitchShift(min_semitones=-1, max_semitones=1, p=0.8),
            # Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.8),
            # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.8)
    ])
    x = np.array(example['input_values'])

    x_augmented = augmentation(x, sample_rate=fs)
    example['input_values'] = x_augmented
    return example

def reshape_c_style_top(example):
    x = example["audio"]['array'][4:,:]
    new_x = np.reshape(x.T, newshape=-1).T
    example["audio"]['array'] = new_x
    return example

def reshape_c_style_bottom(example):
    x = example["audio"]['array'][0:4,:]
    new_x = np.reshape(x.T, newshape=-1).T
    example["audio"]['array'] = new_x
    return example

def reshape_stack(example):
    x = example["audio"]['array'] 
    new_x = np.reshape(x, newshape=-1)
    example["audio"]['array'] = new_x
    return example

# This is the map function that takes the max absolute of the channels
def channel_maxabs(example):
    x = example["audio"]['array']
    new_x = x[np.argmax(np.abs(x), axis=0), np.arange(x.shape[1])]
    example["audio"]['array'] = new_x
    return example

def plot_sample(example, title):
    tmp_sample = example['audio']['array']
    plt.plot(tmp_sample)
    plt.savefig(f"/home/ajgeglio/FutureGroup/{title}.png")

if __name__ == "__main__":
    start_time = stopwatch()

    parser = argparse.ArgumentParser(description='Creates a dataset of taps from a wavfile of recorded typed sentences and associated neonode file')
    # Logical operators for configuration of data
    parser.add_argument('--presplit', help='Used if samples are stored in separate directories for training and testing', action="store_true")
    parser.add_argument("--oversample", help="oversample class samples until 'all-exausted' to create class balance dataset", action="store_true")
    parser.add_argument('--test_size',type=float, help='validation/evaluation dataset split', dest = "ts_",default=0.5)
    parser.add_argument('--encode', help='maps and saves dataset with autofeature extractor (currently this is done in the training and inference programs)', action="store_true")
    parser.add_argument('--seed',type=int, help='set random seed', dest = "seed_",default=42)
    parser.add_argument('--sample_rate',type=int, help='sample rate to cast audio (khz)', dest = "fs_",default=None)
    parser.add_argument('--plot_average_channel', help='plot signal after average channel', action="store_true")
    # Mapping functions for past experiments
    parser.add_argument('--prop',type=float, help='proportion of total data to use', dest = "prop_",default=1.0)
    parser.add_argument('--reshape_interleave', help='reshapes an n_samp x 8-channel matrix with numpy reshape, order = F', action="store_true")
    parser.add_argument('--reshape_c_style_top', help='reshapes the 8-channel data with numpy reshape, only top mics', action="store_true")
    parser.add_argument('--reshape_c_style_bottom', help='reshapes the 8-channel data with numpy reshape, only bottom mics', action="store_true")
    parser.add_argument("--reshape_stack", help="flattens 8-channels by stacking to 8x length", action="store_true")
    parser.add_argument('--max_absolute', help='maps 8-channel dataset to a 1 channel ds with the max absolute amplitude value', action="store_true")
    # Taps and label files
    parser.add_argument('--data_dir', help="directory of all 96k samples and labels", required=False, dest="data_dir", default='/work/ajgeglio/Tap_Data/00.All_TapsLabels_96k')
    parser.add_argument('--train_dir96k', help="directory of train samples and labels", required=False, dest="train_dir96k", default='/work/ajgeglio/Tap_Data/01.Train_TapsLabels_96k')
    parser.add_argument('--test_dir96k', help="directory of train samples and labels", required=False, dest="test_dir96k", default='/work/ajgeglio/Tap_Data/02.Test_TapsLabels_96k')
    # Datastes Directories
    parser.add_argument('--oversampled_dir', help="directory of the oversampled dataset", dest="interleave_dir", default='/work/ajgeglio/Tap_Data/10.Oversampled_Dataset')
    parser.add_argument('--save_96k_all', help="directory of dictionary dataset with train+validation+evaluation sets", required=False, dest="save_96k_all", default = '/work/ajgeglio/Tap_Data/11.All_Dataset_96k')
    parser.add_argument('--save_tr96k', help="directory of a training dataset only", required=False, dest="save_tr96k", default = '/work/ajgeglio/Tap_Data/03.Train_Dataset_96k')
    parser.add_argument('--save_te96k', help="directory of a dictionary dataset with validation+evaluation sets", required=False, dest="save_te96k", default = '/work/ajgeglio/Tap_Data/04.Test_Dataset_96k')

    args = parser.parse_args()

    if not args.presplit:
        label_files, tap_files = list_samples_labels(args.data_dir)
        label = concat_labels(label_files)
        if args.prop_ < 1.0:
            idx = list(range(len(label)))
            random.seed(args.seed_)
            randIdx = random.sample(idx,int(args.prop_*len(idx)))
            tap_files = np.array(tap_files)[randIdx]
            label = np.array(label)[randIdx]
            print("new sample size: ", len(tap_files))
            print(len(label))
            
        tap_dataset = Dataset.from_dict({"audio": tap_files, 'label': label[:,1]})
        if args.fs_ != None:
            print(f"casting to {args.fs_} khz")
            tap_dataset = tap_dataset.cast_column("audio", Audio(mono=False, sampling_rate = args.fs_))
        ############## To use the mapping functions #####################
        if args.reshape_interleave:
            print("Interleaving audio")
            example = tap_dataset[64]
            tmp = reshape_interleave(example)
            plot_sample(tmp, 'sample_reshape-interleave')
            tap_dataset = tap_dataset.map(reshape_interleave)
            sample = tap_dataset[64]['audio']['array']
            print("New Shape: ", sample.shape)
        if args.reshape_c_style_top:
            example = tap_dataset[64]
            tmp = reshape_c_style_top(example)
            plot_sample(tmp, 'sample_reshape-interleave-top')
            tap_dataset = tap_dataset.map(reshape_c_style_top)
            sample = tap_dataset[0]['audio']['array']
            print("New Shape: ", sample.shape)
        if args.reshape_c_style_bottom:
            example = tap_dataset[64]
            tmp = reshape_c_style_bottom(example)
            plot_sample(tmp, 'sample_reshape-interleave-bottom')
            tap_dataset = tap_dataset.map(reshape_c_style_top)
            sample = tap_dataset[0]['audio']['array']
            print("New Shape: ", sample.shape)
        if args.reshape_stack:
            example = tap_dataset[64]
            tmp = reshape_stack(example)
            plot_sample(tmp, 'sample_reshape-stack')
            tap_dataset = tap_dataset.map(reshape_stack)
            sample = tap_dataset[0]['audio']['array']
            print("New Shape: ", sample.shape)
        if args.max_absolute:
            example = tap_dataset[64]
            tmp = channel_maxabs(example)
            plot_sample(tmp, 'sample_max_absolute')
            tap_dataset = tap_dataset.map(channel_maxabs)
            new_sample = tap_dataset[0]['audio']['array']
            print("New Shape: ", new_sample.shape)
        if args.plot_average_channel:
            example = tap_dataset[64]["audio"]['array']
            new_x = np.average(example, axis=0)
            print(new_x.shape)
            plt.plot(new_x)
            plt.savefig(f"/home/ajgeglio/FutureGroup/sample_avr_channel.png")
            quit()
        ##################################################################

        tap_dataset = tap_dataset.class_encode_column("label")      
        # split twice and combine
        train_set = tap_dataset.train_test_split(   shuffle = True, 
                                                    seed = args.seed_, 
                                                    stratify_by_column ='label',
                                                    test_size=0.3)
        test_set = train_set['test'].train_test_split(  shuffle = True, 
                                                        seed = args.seed_, 
                                                        stratify_by_column ='label',
                                                        test_size=args.ts_)
        tap_dataset = DatasetDict({
            'train'         : train_set['train'],
            'validation'    : test_set['train']})
        
        # print(f"Saving 3-split 96k dataset to disk, directory: {args.save_96k_all}")
        # tap_dataset.save_to_disk(args.save_96k_all)
        print(f"Saving test dataset separately to: {args.save_te96k}")
        tap_dataset_test =  test_set['test']
        tap_dataset_test.save_to_disk(args.save_te96k)
        
        
    if args.presplit:
        label_files, tap_files = list_samples_labels(args.train_dir96k)
        label = concat_labels(label_files)
        label_files_test, tap_files_test = list_samples_labels(args.test_dir96k)
        label_test = concat_labels(label_files_test)
        tap_dataset = Dataset.from_dict(   {"audio": tap_files, 
                                            'label': label[:,1]})
        tap_dataset_test = Dataset.from_dict(   {"audio": tap_files_test, 
                                                'label': label_test[:,1]})
        tap_dataset = tap_dataset.class_encode_column("label").shuffle(seed=args.seed_)
        tap_dataset_test = tap_dataset_test.class_encode_column("label")
        tap_dataset_test = tap_dataset_test.train_test_split(   test_size = args.ts_, 
                                                                stratify_by_column ='label', 
                                                                shuffle = True, 
                                                                seed = args.seed_)
        # print(f"Saving training dataset to disk, directory: {args.save_tr96k}")
        # tap_dataset.save_to_disk(args.save_tr96k)
        # print(f"Saving test dataset to disk, directory: {args.save_te96k}")
        # tap_dataset_test.save_to_disk(args.save_te96k)

    if args.oversample:
        tap_dataset = oversample_interleave(tap_dataset, args.seed_)
        # print(f'Saving oversampled dataset to disk, {args.interleave_dir}')
        # tap_dataset.save_to_disk(args.interleave_dir)

    if args.encode:
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        # tap_dataset = tap_dataset.cast_column("audio", Audio(mono=True))
        encoded_dataset = tap_dataset.map(preprocess_function, remove_columns=["audio"], batched=True)
        encoded_dataset_test = tap_dataset_test.map(preprocess_function, remove_columns=["audio"], batched=True)
        print(f"Saving encoded dataset to disk, /work/ajgeglio/Tap_Data/09.Encoded_Dataset")
        encoded_dataset.save_to_disk('/work/ajgeglio/Tap_Data/09.Encoded_Dataset')
        encoded_dataset_test.save_to_disk('/work/ajgeglio/Tap_Data/09.Encoded_Dataset_Test')
   
    print('DATASET DESCRIPTION #################################')
    print(tap_dataset)
    print('Sound Features #######################################')
    try:
        fs = tap_dataset['train'].features['audio'].sampling_rate
        print("Sampling Rate: ", fs)
        wavform = tap_dataset['train'][0]['audio']
        print("Waveform", wavform)
        print("Waveform shape", wavform['array'].shape)
    except:
        try:
            fs = args.fs_
            print("Sampling Rate: ", fs)
            wavform = tap_dataset['train'][0]['audio']
            print("Waveform", wavform)
        except:
            fs = args.fs_
            print("Sampling Rate: ", fs)
            wavform = tap_dataset[0]['audio']
            print("Waveform", wavform)

    print('\n')

    print(f"TOTAL TIME: {stopwatch() - start_time:.2f}")