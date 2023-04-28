# Wav2Vec-tap2key
### Optimizing the Wav2Vec2 base model with a 34-classs classification layer representing tap locations.

Data: Audio from test users tapping on a surface with that has a graphical keyboard with 34 key locations


The facebook/wav2vec2-base pretrained transformer
https://huggingface.co/facebook/wav2vec2-base

Pretrained Weights
* Wav2Vec2-Base
* Facebook's Wav2Vec2

The base model pretrained on 16kHz sampled speech audio. This will determine the models assumption of your sample rate

Note: This model does not have a tokenizer as it was pretrained on audio alone. 
In order to use this model speech recognition, a tokenizer should be created and the model should be fine-tuned on labeled text data. 
Check out this blog for more in-detail explanation of how to fine-tune the model.

Paper

Authors: Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli


# **** Python Files ****

# 1) WavPreprocess.py - Used for raw sound file processing
* Creates tap samples in wave format (.wav) and associated labels (.csv) from a wavfile that has recorded typing sounds
* Requires an associated neonode file which is used for labeling

### NOTE - The current os.walk function requires the following folder structure:
*    path/to/original/recordings/01.acoustic --- this is the folder where the original recordings are
*    path/to/original/recordings/02.neonode --- this is where the raw neonode csv is

## EXAMPLE USAGE 

### First, it is reccomended to plot without generating samples...
    python3 WavPreprocess.py --dir 'path/to/original/recordings' --idx 9 --latency -0.14 --peak_height 0.0025 --plot
* The --dir argument is the path to the directory not the specific folder with the .wav samples, i.e. /work/ajgeglio/Tap_Data/Tony_01_27_23_data
* The --latency argument allows you to adjust the recording start time so to line up with the neonode data
* The --peak_height argument adjusts what taps are detected from the wave file
* The --idx is the index of the recording in the filepath
* It should be noted that the generated plot is showing you the filtered averaged signal using a Butterworth Highpass Filter on the average of 8 channels. This is better for tap detection. The actual data that is sampled is the raw, 8-channel data of length 8192 at a sample rate of 96,000, or, 0.085 second windows.

### When the plot it generates which shows you the tap detection and sample windows, you can then perform the sample generation
    python3 WavPreprocess.py --dir 'path/to/original/recordings' --idx 9 --latency -0.14 --peak_height 0.0025 --selectivity 0.134 --sample
* The --selectivity argument allows you to ignore samples that do not coincide with a neonode samples. It is in seconds.
* The --sample argument is going to generate the samples. 
* The OUTPUT after sampling sould be a .wav file for each tap and 1 csv file of labels.

# 2) create_dataset.py - Used for Datasets Generation

* A flexible program to generate Hugging Face Datasets
* Uses the labels (.csv) and samples (.wav) of captured taps with WavPreprocess.py
* This program a lot of flexibility to allow for experimentation with different data shapes, such as interleaving the channels, stacking, max-absulute value from each channel, and channel average. 
* These are defined in mapping functions. 
* Also, you can specify the proportion of total data for additional experimentation.
* The interleaving of the 8-channels were found to be the most effective representation, so this was incorperated into the preprocess
function in the training program 'Wav2vec2-tap2key'.

## EXAMPLE USAGE

    python create_dataset.py --all_data --sample_rate 16_000 --reshape_interleave --oversample

* NOTE - In order to use a mapping function like reshape_interleave, you must specify a sample_rate, otherwise the dataset only maps to the file location which is much more efficient, but does not carry the data in memory for doing such calculations.
* --oversampling argument uses Hugging Face "interleave_datasets" with the arguments probabilities=None, stopping_strategy = 'all_exausted'
* --reshape_interleave argument reshapes the data using numpy.reshape(order="F", newshape=-1) which results in a 1-channel audio signal by flattening out the (n_channel) x (n_samples) array with the "F" order. See numpy reshape documentation.

# 3) Wav2vec2_tap2key.py - Training the model

* Wav2vec2_tap2key does supervised training and evaluation on audio data with labels. 
* The model is a transformer based on Wav2vec 2.0 architecture with the Facebook/Wav2Vec2-base weight initialialization. 
* By default the program creates a dataset from the samples and labels, interleaves 8-channel audio to 1-channel 8x length, and it casts it to 16 khz audio.
* The AutoFeatureExtractor is used and the max_duration is set to 1-second. 
* The fine tuning is done on an added 34 class layer based on the virtual table-top keyboard keys.


## EXAMPLE USAGE

    python3 Wav2Vec2_tap2key.py --oversample --train --early_stop 
* --oversampling argument uses Hugging Face "interleave_datasets" with the arguments probabilities=None, stopping_strategy = 'all_exausted'.
  * Each class is separated into a dataset and each dataset is randomly sampled until all classes run out of samples
  * The resulting dataset has a class balance with size (n number of samples in most populated class) x (n classes)
* --train argument executes the training loop
  * default traing args saves only 2 checkpoints and loads best model at the end
* --early_stop argument uses an early stopping callback with a default patience of 16 epochs with worse eval performance


# 4) predict_model_conf_matrix.py - Used for model evaluation

* Evaluates the holdout test set based using a saved model checkpoint
* Has a function to output a confution matrix

## EXAMPLE USAGE
  python 3 --dataset_dir path/to/test/dataset

# **** Setting up Python environment with conda****

% conda create -n Wav2Vec-tap2key python=3.8
% conda activate Wav2Vec-tap2key

For machine without GPU:
% conda install pytorch torchvision torchaudio -c pytorch

For machine with GPU:
% conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

% conda install -c huggingface transformers (had an error with this last time - https://github.com/huggingface/tokenizers/issues/585)
(conda uninstall transformers)

So instead use: 
% pip install transformers==4.26.0
* Note - Whatever version of transformers is used for training must be the same version used for inference 

% conda install -c huggingface -c conda-forge datasets=2.10.1
or
% pip install evaluate==0.4.0

% conda install scikit-learn=1.0.2
% conda install matplotlib
% conda install -c conda-forge librosa=0.9.2


Needed for counting physical CPU cores:
% pip install psutil

