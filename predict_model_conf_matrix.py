from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from transformers import AutoModelForAudioClassification, Trainer, TrainingArguments
from transformers import AutoFeatureExtractor
from datasets import Audio, Dataset, load_from_disk
from timeit import default_timer as stopwatch
import time
import evaluate
import matplotlib.pyplot as plt
import os
import argparse
import torch
import numpy as np

# My Functions
from create_dataset import label_encoder
from Wav2vec2_tap2key import compute_metrics

def preprocess_function(examples):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    max_duration = 8  # seconds
    # This does encoding and interleaving of channels in the same function
    audio_arrays = [np.reshape(x["array"], order='F', newshape=-1) for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        truncation=True, 
    )
    return inputs


if __name__== '__main__':
    start = stopwatch()
    t = time.localtime()
    current_time = time.strftime("%b-%d-%Y-%H:%M", t)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    parser = argparse.ArgumentParser(description='Run Evaluation prediction on trained Wav2Vec2 model and output\
                                        a performance report and confusion matrix plot of the results')
    parser.add_argument('--dataset_dir', help="directory of the evaluation dataset", dest="dataset_dir", default='/work/ajgeglio/Tap_Data/04.Test_Dataset_96k')
    args = parser.parse_args()

     ####### LOAD DATASET ##############
    try:
        tap_dataset_test = load_from_disk(args.dataset_dir)['evaluation']
    except:
        tap_dataset_test = load_from_disk(args.dataset_dir)
        tap_dataset_test = tap_dataset_test.cast_column("audio", Audio(sampling_rate = 16000, mono=False))

    ############# PRINT BASIC DATA PARAMETERS ####################
    print("############# WAVEFORM ###################")    
    fs = tap_dataset_test.features['audio'].sampling_rate
    wavform = tap_dataset_test[64]['audio']['array']
    reshape_wavform = np.reshape(wavform, order='F', newshape=-1)
    sample_time = wavform.shape[1]/fs
    try: np.isclose(sample_time, 0.085375)
    except: 
        print("Data casted incorrectly to not reflect actual sample time window")
        quit()
    reshape_time = len(reshape_wavform)/fs
    print("Sample rate:", fs,
        "\nReal sample time(s):", sample_time, 
        "\nReshaped sample time(s):", reshape_time, 
        "\nwaveform shapes:","original-->", wavform.shape, "reshaped-->", reshape_wavform.shape)
    # print("###########################################") 
    label2id, id2label = label_encoder(tap_dataset_test)
    num_labels = len(id2label)
    labels_ = label2id.keys()

    # max_eval_size = 2456
    # prop = num_rows/max_eval_size
    # print(f"{prop*100:0.0f}p of largest eval set")

    ############## Define the path to the saved checkpoint file for the best models ###############################
    ##          Before Group data collection
    # model_checkpoint = "/work/ajgeglio/pretrained_models/checkpoint-3922_Feb-16-2023-17:03_tap2key/checkpoint-18500"
    # model_checkpoint = '/work/ajgeglio/pretrained_models/checkpoint-3922_Feb-23-2023-17:54_tap2key/checkpoint-4224'
    #           (New Group Data and not oversampled)
    # model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-06-2023-12:34_tap2key/checkpoint-9256'
    #           (New Group Data and oversampled (interleave sampling, all exausted))
    # model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-07-2023-11:42_tap2key/checkpoint-7098'
    ##          50% of data
    # model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-13-2023-13:38_tap2key/checkpoint-4224'
    ##          60% of data
    # model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-13-2023-18:04_tap2key/checkpoint-4982'
    ##          70% of data
    # model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-14-2023-09:54_tap2key/checkpoint-7378'
    ##          80% of data
    # model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-14-2023-14:15_tap2key/checkpoint-5254'
    ##          90% of data
    # model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-15-2023-08:26_tap2key/checkpoint-6560'
    ##          Interleaved channel data to 8x length, oversampled data (all exausted), used 100% of data collected to date
    # model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-16-2023-10:51_tap2key/checkpoint-8124'- did not increase max_duration
    # model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-18-2023-11:13_tap2key/checkpoint-6422'
    ##          Flatten signal to 8x len, oversampled (all exausted), used 100%
    # model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-19-2023-18:40_tap2key/checkpoint-6084'
    ##          Max absolute value from each channel, oversampled data (all exausted), used 100% of data collected to date
    # model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-16-2023-17:50_tap2key/checkpoint-6600' 
    ##          100% of data, average channel, oversampled early stop. Repeated experiment twice
    # model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-16-2023-23:19_tap2key/checkpoint-8631'
    # model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-21-2023-13:11_tap2key/checkpoint-6929'
    ##          All data, interleave chan, 48k
    # model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-22-2023-19:43_tap2key/checkpoint-3612'
    ##          All data, oversampled, 24 khz, interleave chan, len 16,384
    # model_checkpoint =  '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-23-2023-09:08_tap2key/checkpoint-6216'
    ##          All data, oversampled, 16 khz, interleave chan, len 10928
    # model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-23-2023-15:49_tap2key/checkpoint-6468'
    ##          Top Microphones, 16 khz, oversampled
    # model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-24-2023-11:11_tap2key/checkpoint-6720'
    ##          Bottom Microphones, 16 khz, oversampled
    # model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-24-2023-18:40_tap2key/checkpoint-7476'
    ##          Using Data augmentation
    # model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Apr-08-2023-17:48_tap2key/checkpoint-5124'
    ##          New re-labeled dataset
    model_checkpoint = '/work/ajgeglio/pretrained_models/wav2vec2-base_Apr-27-2023-11:38_tap2key/checkpoint-5525'
    batch_size = 64
    encoded_test_dataset = tap_dataset_test.map(preprocess_function, remove_columns=["audio"], batched=True)

    # print(encoded_test_dataset)
    model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, 
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label)

    f1_metric = evaluate.load("f1")
    # Load the training args used to train the checkpoint
    args = torch.load(model_checkpoint+"/training_args.bin")
    # print(args)
    # Load the trainer used to train the checkpoint
    trainer = Trainer(model=model, args=args)

    logits, y_test , metrics = trainer.predict(encoded_test_dataset)
    y_pred = logits.argmax(-1)
    print(f"TOTAL TIME: {stopwatch() - start:.2f}")
    # predicted_labels = [model.config.id2label[id] for id in y_pred.squeeze().tolist()]
    # print(predicted_labels)
    cm = confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred, target_names=labels_))
    print(cm)

    disp = ConfusionMatrixDisplay.from_predictions( y_test, y_pred, display_labels=labels_)
    fig = disp.ax_.get_figure() 
    fig.set_figwidth(14)
    fig.set_figheight(14)  
    # plt.savefig(f'confusion_matrix_interleave_16khz_bottom.png')