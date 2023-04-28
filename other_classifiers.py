import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC as SVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from datasets import Audio, Dataset, load_from_disk
from timeit import default_timer as stopwatch
import time
import argparse
import os
# Tensorflow
from tensorflow import keras
import tensorflow as tf
from keras.layers import GaussianNoise
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback, ModelCheckpoint

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

def extract_features(file_path):
    # audio, sample_rate = librosa.load(file_path)
    sample_rate, audio = wavfile.read(file_path)
    audio = np.swapaxes(audio, 1,0)
    # print(audio.shape)
    # quit()
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def preprocess_function(examples):
    max_duration = 5  # seconds
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True, 
    )
    return inputs

def train_rf(X_train, y_train, X_test, y_test):
    start_time = stopwatch()
    model = RandomForestClassifier(n_estimators=200)
    # model = AdaBoostClassifier(n_estimators=100)
    # model = SVM(kernel = 'linear')
    model.fit(X_train, y_train)
    # accuracy = model.score(X_test, y_test)
    print(f"TRAIN TIME: {stopwatch() - start_time:.2f}")
    start_time = stopwatch()
    y_pred = model.predict(X_test)
    f1_score_ = f1_score(y_test,y_pred, average = 'weighted')
    print(f"INFERENCE TIME: {stopwatch() - start_time:.2f}")
    print(classification_report(y_test, y_pred))
    print("Weighted F1_score:", f1_score_)
    # return model, f1_score_
    
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

def basic_convnet():
    model = keras.Sequential()
    # model.add(GaussianNoise(0.01, input_shape=(audio_len,1)))
    model.add(keras.layers.Conv1D(512, kernel_size=128,strides=4, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv1D(512, kernel_size=128,strides=4, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv1D(128, kernel_size=64,strides=2, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(34, activation='sigmoid'))
    print(model.summary())
    # quit()
    return model

if __name__ == "__main__":
    start_time = stopwatch()
    t = time.localtime()
    current_time = time.strftime("%b-%d-%Y-%H:%M", t)
    name_time = str(current_time).replace(':','.').replace(' ','.')
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    parser = argparse.ArgumentParser(description='Creates a dataset of taps from a wavfile of recorded typed sentences and associated neonode file')
    parser.add_argument('--dir', help="directory of the samples and labels", dest="dir_", default='/work/ajgeglio/Tap_Data/00.All_TapsLabels_96k')
    parser.add_argument('--encoded_dir', help="directory of the samples and labels", dest="ds_", default='/work/ajgeglio/Tap_Data/09.Encoded_Dataset')
    parser.add_argument('--encoded_test_dir', help="directory of the samples and labels", dest="ds_test", default='/work/ajgeglio/Tap_Data/09.Encoded_Dataset_Test')
    parser.add_argument('--test_size',type=float, help='test size 0-1', dest = "ts_",default=0.2)
    parser.add_argument('--seed',type=int, help='set random seed', dest = "seed_",default=42)
    parser.add_argument("--create_dataset_manual", help="create own dataset", action="store_true")
    parser.add_argument("--convnet", help="simple 1-d sequential keras model", action="store_true")
    parser.add_argument("--test_convnet", help="test keras model", action="store_true")
    parser.add_argument("--train_rf", help="use random forests", action="store_true")
    parser.add_argument("--all_ds_to_numpy", help="use the saved dataset convert to numpy", action="store_true")
    parser.add_argument("--all_ds_to_tf", help="use the saved dataset convert to numpy", action="store_true")
    args = parser.parse_args()

 
    if args.all_ds_to_numpy:
        dataset = load_from_disk(args.ds_)
        test_dataset = load_from_disk(args.ds_test)
        print('loaded dataset')
        print(dataset)
        # val = dataset['validation']['input_values']
        # valY = dataset['validation']['label']
        X_train = dataset['train']['input_values']
        y_train = dataset['train']['label']
        X_test = test_dataset['input_values']
        y_test = test_dataset['label']

    if args.all_ds_to_tf:
        ds_train = load_from_disk(args.ds_)['train']
        audio_len = len(ds_train[0]['input_values'])
        ds_val = load_from_disk(args.ds_)['validation']
        tf_ds_train = ds_train.to_tf_dataset(   columns=["input_values"],
                                                label_cols=["label"],
                                                batch_size=32,
                                                shuffle=True)

        tf_ds_val = ds_val.to_tf_dataset(   columns=["input_values"],
                                            label_cols=["label"],
                                            batch_size=32,
                                            shuffle=True)
        print(tf_ds_train)
        
        # quit()

    if args.create_dataset_manual:
        label_files, tap_files = list_samples_labels(args.dir_)
        label = concat_labels(label_files)
        # Load the audio files and extract features
        file_paths = label_files # list of file paths
        labels = label[:,1] # list of corresponding labels
        features = np.array([extract_features(f) for f in file_paths])
        nsamples, nx, ny = features.shape
        d2_features = features.reshape((nsamples,nx*ny))
        print(features.shape)
        print(labels.shape)
    
    if args.convnet:
        # Define the CNN model
        model = basic_convnet()
        early_stopping_monitor = EarlyStopping( monitor='val_loss',
                                                min_delta=0,
                                                patience=16,
                                                verbose=0,
                                                mode='auto',
                                                baseline=None,
                                                restore_best_weights=True)
        model_path = f'./convnet/{name_time}'
        callbacks = [ModelCheckpoint(filepath=model_path, save_best_only=True)]
        # quit()
        # Compile the model
        model.compile(  optimizer='adam', 
                        loss="sparse_categorical_crossentropy", 
                        # metrics=['accuracy']
                        metrics=["sparse_categorical_accuracy"]
                        )

        # Train the model
        history = model.fit(tf_ds_train, validation_data=tf_ds_val, epochs=100, callbacks=[early_stopping_monitor, callbacks])
        # pd.DataFrame(history).to_csv(f"history.csv")
        # model.save("basic_convnet")

        # Evaluate the model
    if args.test_convnet:
        ds_test = load_from_disk(args.ds_test)
        print(ds_test)
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        model_path = f'./convnet/Apr-07-2023-18.29/'
        model = keras.models.load_model(model_path)
        # test_loss, test_acc = model.evaluate(tf_ds_test)
        # print('Test accuracy:', test_acc)
        X_test = ds_test['input_values']
        y_test = ds_test['label']
        predict_x=model.predict(X_test) 
        y_pred = np.argmax(predict_x,axis=1)
        label2id, id2label = label_encoder(ds_test)
        labels_ = label2id.keys()
        print(classification_report(y_test, y_pred, target_names=labels_))
        # print(cm)
    
    if args.train_rf:
        
        train_rf(X_train, y_train, X_test, y_test)
        
        
