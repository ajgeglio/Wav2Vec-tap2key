import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.svm import SVC as SVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from datasets import Audio, Dataset, load_from_disk
from timeit import default_timer as stopwatch
import time
from tensorflow import keras
import tensorflow as tf
from keras.layers import GaussianNoise
import argparse
import os

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

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.ts_)
    model = RandomForestClassifier(n_estimators=100)
    # model = AdaBoostClassifier(n_estimators=100)
    # model = SVM(kernel = 'linear')
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    f1_score_ = f1_score(y_test,y_pred, average = 'weighted')
    print(classification_report(y_test, y_pred))
    return model, f1_score_
    
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
    model.add(GaussianNoise(0.01, input_shape=(audio_len,1)))
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description='Creates a dataset of taps from a wavfile of recorded typed sentences and associated neonode file')
    parser.add_argument('--dir', help="directory of the samples and labels", dest="dir_", default='/work/ajgeglio/Tap_Data/00.All_TapsLabels_96k')
    parser.add_argument('--encoded_dir', help="directory of the samples and labels", dest="ds_", default='/work/ajgeglio/Tap_Data/09.Encoded_Dataset')
    parser.add_argument('--test_size',type=float, help='test size 0-1', dest = "ts_",default=0.2)
    parser.add_argument('--seed',type=int, help='set random seed', dest = "seed_",default=42)
    parser.add_argument("--create_dataset_manual", help="create own dataset", action="store_true")
    parser.add_argument("--convnet", help="simple 1-d sequential keras model", action="store_true")
    parser.add_argument("--test_convnet", help="test keras model", action="store_true")
    parser.add_argument("--rf", help="use random forests", action="store_true")
    parser.add_argument("--all_ds_to_numpy", help="use the saved dataset convert to numpy", action="store_true")
    parser.add_argument("--all_ds_to_tf", help="use the saved dataset convert to numpy", action="store_true")
    args = parser.parse_args()

 
    if args.all_ds_to_numpy:
        dataset = load_from_disk(args.ds_)
        print('loaded dataset')
        print(dataset)
        validation = dataset['validation']
        training = dataset['train']
        test = dataset['test']

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
        ds_test = load_from_disk(args.ds_)['test']
        # tf_ds_test = ds_test.to_tf_dataset(   columns=["input_values"],
        #                                         label_cols=["label"],
        #                                         batch_size=32,
        #                                         shuffle=True)
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
    
    if args.rf:
        model, f1_score_ = train_model(d2_features, labels)
        print("Weighted F1_score:", f1_score_)

    '''
(Wav2vec) ajgeglio@cheetah:~/FutureGroup$ python rf_classifier.py
(16369, 40, 8)
(16369,)
              precision    recall  f1-score   support

          _1       0.79      0.60      0.68        50
          _2       0.65      0.89      0.75        99
          _3       0.44      0.25      0.32        48
          _4       0.50      0.66      0.57        80
          _5       0.60      0.79      0.68       146
          _6       0.68      0.69      0.69        84
           a       0.53      0.89      0.67       152
           b       0.70      0.54      0.61        68
           c       0.59      0.65      0.62        85
           d       0.67      0.45      0.54        65
           e       0.66      0.91      0.77       190
           f       0.76      0.47      0.58        74
           g       0.79      0.23      0.36        64
           h       0.64      0.70      0.67       122
           i       0.54      0.77      0.63       149
           j       0.71      0.15      0.25        65
           k       0.67      0.19      0.30        63
           l       0.88      0.28      0.43        82
           m       0.83      0.42      0.56        83
           n       0.58      0.64      0.61        84
        none       0.85      0.93      0.89        81
           o       0.60      0.89      0.72       203
           p       0.86      0.71      0.77        95
           q       0.77      0.42      0.54        65
           r       0.69      0.54      0.60       114
           s       0.55      0.68      0.61       142
           t       0.59      0.80      0.68       128
           u       0.65      0.75      0.70       139
           v       0.75      0.30      0.42        61
           w       0.85      0.18      0.29        62
           x       0.79      0.46      0.58        71
           y       0.70      0.75      0.72       107
           z       0.73      0.26      0.38        92
           |       0.98      0.85      0.91        61

    accuracy                           0.64      3274
   macro avg       0.69      0.58      0.59      3274
weighted avg       0.67      0.64      0.62      3274

Weighted F1_score: 0.6200965251904987
    '''