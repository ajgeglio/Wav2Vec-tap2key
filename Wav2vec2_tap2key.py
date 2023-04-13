'''
Wav2vec2_tap2key does supervised training and evaluation on audio data with labels. 
The model is a transformer based on Wav2vec 2.0 architecture with the Facebook/Wav2Vec2-base weight initialialization. 
The fine tuning is done on an added 34 class layer based on the virtual table-top keyboard keys.

By default it creates a dataset from the samples and labels and interleaves 8-channel audio and casts it to 16 khz

'''
from huggingface_hub import notebook_login
import evaluate
from datasets import Audio, Dataset, load_from_disk, DatasetDict,  interleave_datasets, load_dataset
# from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import Wav2Vec2Model, Wav2Vec2Config
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, IntervalStrategy
from timeit import default_timer as stopwatch
import time
import argparse
import os
import numpy as np
# from huggingface_hub import HfApi
from create_dataset import label_encoder, list_samples_labels, concat_labels, oversample_interleave, create_dataset, augment_taps

'''
Wav2vec preprocess step to create features from audio samples...

Examples are fed to the feature_extractor with the argument truncation=True, 
as well as the maximum sample length. This will ensure that very long inputs 
like the ones in the _silence_ class can be safely batched.

https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/audio_classification.ipynb#scrollTo=qUtxmoMvqml1

The feature extractor will return a list of numpy arays for each example:

'''

def preprocess_function(examples):
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
'''
Here, we need to define a function for how to compute the metrics from the predictions, which will just use 
the metric we loaded earlier. The only preprocessing we have to do is to take the argmax of our predicted logits:
'''
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return f1_metric.compute(predictions=predictions, references=eval_pred.label_ids, average="weighted")

if __name__== '__main__':
    
    start_time = stopwatch()
    t = time.localtime()
    current_time = time.strftime("%b-%d-%Y-%H:%M", t)
    name_time = str(current_time).replace(':','.').replace(' ','.')
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    f1_metric = evaluate.load("f1")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model_checkpoint = "facebook/wav2vec2-base"
    batch_size = 64
    '''
    To apply the preprocess function on all samples in our dataset, we just use the map method of our dataset object we created earlier. 
    * This will apply the function on all the elements of all the splits in dataset, so our training, validation and 
    * testing data will be preprocessed in one single command.
    '''
    parser = argparse.ArgumentParser(description='Wav2vec2_tap2key does training and evaluaiton of the tap sample\
            dataset that has already been created or can create dataset. The model is a transformer based on Wav2vec2 architecture with\
            Wav2Vec2-base weight initialialization. The fine tuning is done on 34 classes based on the virtual\
            table top keyboard and audio sample classification is supervised based on the labels created with\
            WavePreprocess.py')
    parser.add_argument('--dataset_dir', help="directory of the hugging face dataset", dest="dataset_dir", default='/work/ajgeglio/Tap_Data/09.Encoded_Dataset')
    parser.add_argument('--data_dir', help="directory of the taps/labels", dest="data_dir", default='/work/ajgeglio/Tap_Data/00.All_TapsLabels_96k')
    parser.add_argument('--seed',type=int, help='set random seed', dest = "seed_",default=42)
    parser.add_argument("--load_dataset", help="load hugging face dataset saved to disk", action="store_true")
    parser.add_argument('--save_te96k', help="directory to save evaluation dataset", dest="save_te96k", default = '/work/ajgeglio/Tap_Data/04.Test_Dataset_96k')
    parser.add_argument("--oversample", help="oversample hugging face dataset", action="store_true")
    parser.add_argument('--sample_rate',type=int, help='sample rate to cast audio (khz)', dest = "fs_",default=16_000)
    parser.add_argument("--early_stop", help="early stopping using evaluation loss", action="store_true")
    parser.add_argument("--train", help="run training epochs", action="store_true")
    parser.add_argument("--early_patience", type=int, help="number of worse evals before early stopping", default=16)
    parser.add_argument("--epochs", type=int, help="number of epochs to run", default=150)
    args = parser.parse_args()

    if args.load_dataset:
        ########### Load alreay created HF DATASETS ######################
        print('loading hugging face dataset created earlier')
        encoded_dataset = load_from_disk(args.dataset_dir)
        # For testing an augmentation mapping
        # encoded_dataset['train'] = encoded_dataset['train'].map(augment_taps, args.fs_)
        
        tap_dataset = tap_dataset.cast_column("audio", Audio(sampling_rate = args.fs_, mono=False))
        # tap_dataset_test = load_from_disk(args.test_set_dir)

    if not args.load_dataset:
        print("Creating Dataset")
        s2 = stopwatch()
        tap_dataset = create_dataset(args.data_dir, args.save_te96k, args.seed_)
        tap_dataset = tap_dataset.cast_column("audio", Audio(sampling_rate = args.fs_, mono=False))
        example = tap_dataset['train'][64]
        # train_dataset = tap_dataset['train'].to_iterable_dataset()
        # train_dataset = train_dataset.map(reshape_c_style)
        encoded_dataset = tap_dataset.map(preprocess_function, remove_columns=["audio"], batched=True)
        print(encoded_dataset)
        print(f"DATASET + FEATURE CREATION TIME: {stopwatch() - s2:.2f}")

    label2id, id2label = label_encoder(encoded_dataset)
    num_labels = len(id2label)

    if args.oversample:
        s4 = stopwatch()
        tap_dataset = oversample_interleave(encoded_dataset, args.seed_)
        print(f"OVERSAMPLING TIME: {stopwatch() - s4:.2f}")

    ############# PRINT BASIC DATA PARAMETERS ####################
    print("############# WAVEFORM ###################")  
  
    # Define a generator function that yields the examples from an iterable dataset
    def tap_generator():
        for example in train_dataset.take(1):
            yield example
    # example = Dataset.from_generator(tap_generator, features=tap_dataset['train'].features)
    # if created the dataset
    try:
        fs = example['audio']['sampling_rate']
        wavform = example['audio']['array']
        wavform_shape = wavform.shape
        reshape_wavform = np.reshape(wavform, order='F', newshape=-1).shape
        sample_time = wavform.shape[0]/fs
        reshape_time = len(reshape_wavform)/fs
    # if we loaded the dataset
    except:
        wavform = encoded_dataset['validation']['input_values'][64]
        fs = args.fs_
        sample_time = len(wavform)/8/fs
        reshape_wavform = len(wavform)
        wavform_shape = int(len(wavform)/8*(96000/args.fs_)) 
        reshape_time = len(wavform)/fs
    assert np.isclose(sample_time, 0.085375, rtol=0.001), "Data casted incorrectly to not reflect actual sample time window"
    print("Sample rate:", fs,
        "\nReal sample time(s):", sample_time, 
        "\nReshaped sample time(s):", reshape_time, 
        "\nwaveform shapes:","original-->", wavform_shape, "-reshaped-->", reshape_wavform)
    print("###########################################")  

    ############################################# MODEL ############################################################

    if args.train:
        '''
        ######################## Training the model ###################################

        Now that our data is ready, we can download the pretrained model and fine-tune it. 
        For classification we use the AutoModelForAudioClassification class. 
        Like with the feature extractor, the from_pretrained method will download and cache the model for us. 
        As the label ids and the number of labels are dataset dependent, we pass num_labels, label2id, 
        and id2label alongside the model_checkpoint here:
        '''
        model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, 
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            )

        '''
        To instantiate a Trainer, we will need to define the training configuration and the evaluation metric. 
        The most important is the TrainingArguments, which is a class that contains all the attributes to customize the training. 
        It requires one folder name, which will be used to save the checkpoints of the model, and all other arguments are optional:
        '''
        model_name = f"{model_checkpoint.split('/')[-1]}"

        ######## Checkpoint Save Strategy #############3
        
        # Set up arguments for early stopping
        callbacks = None
        if args.early_stop:
            callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_patience)]

        args = TrainingArguments(
            f"/work/ajgeglio/pretrained_models/{model_name}_{current_time}_tap2key",
            logging_steps=10,
            evaluation_strategy = IntervalStrategy.EPOCH,
            save_strategy = IntervalStrategy.EPOCH,
            learning_rate=3e-5,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=args.epochs,
            warmup_ratio=0.1,
            save_total_limit = 3,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            gradient_checkpointing=True,
            fp16_full_eval=True,
            fp16=True
            # push_to_hub=True,
        )

        '''
        Here we set the evaluation to be done at the end of each epoch, tweak the learning rate, use the batch_size defined at the top 
        of the notebook and customize the number of epochs for training, as well as the weight decay. Since the best model might not 
        be the one at the end of training, we ask the Trainer to load the best model it saved (according to metric_name) at the end of training.

        The last argument push_to_hub allows the Trainer to push the model to the Hub regularly during training. 
        Remove it if you didn't follow the installation steps at the top of the notebook. If you want to save your model 
        locally with a name that is different from the name of the repository, or if you want to push your model under an organization 
        and not your name space, use the hub_model_id argument to set the repo name (it needs to be the full name, including your namespace: 
        for instance "anton-l/wav2vec2-finetuned-ks" or "huggingface/anton-l/wav2vec2-finetuned-ks").
    
        Then we just need to pass all of this along with our datasets to the Trainer:
        '''

        trainer = Trainer(
            model,
            args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            tokenizer=feature_extractor,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            )

        trainer.train()
        # We can check with the evaluate method that our Trainer did reload the best model properly (if it was not the last one):
        trainer.evaluate()
        # You can now upload the result of the training to the Hub, just execute this instruction:
        
        # trainer.push_to_hub()
    print(f"TOTAL TIME: {stopwatch() - start_time:.2f}")

