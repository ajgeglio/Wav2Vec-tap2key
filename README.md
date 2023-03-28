This is for the Wav2Vec base model with a 34-classs classification layer trained on surface key taps. 

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
#############################################################################################
1) WavPreprocess.py - DATA PREPROCESSING
Creates tap samples (.wav) and labels (.csv) from a wavfile of recorded typing sentences and associated neonode data
usage: WavPreprocess.py [-h] --dir DIR_ --idx IDX_ [--plot] [--latency LATENCY_] [--selectivity SELECTIVITY] --peak_height PEAK_HEIGHT [--sample]
optional arguments:
  -h, --help            show this help message and exit
  --dir DIR_            directory of original wav files
  --idx IDX_            integer index of .wav file
  --plot                save plot to visually QC labels
  --latency LATENCY_    adjustment to the neonode timing
  --selectivity SELECTIVITY
                        only select samples this close to a node label (seconds)
  --peak_height PEAK_HEIGHT
                        the height threshold for tap detection
  --sample              generate samples and labels

The current os.walk function requires the following folder structure:
    01.acoustic --- this is the folder where the original recordings are
    02.neonode --- this is where the raw neonode csv is

EXAMPLE USAGE 

WHEN YOU WANT TO JUST PLOT WITHOUT SAMPLING
python3 WavPreprocess.py --dir '/work/ajgeglio/Tap_Data/Tony_01_25_23_data' 
--idx 9 --latency -0.14 --peak_height 0.0025

WHEN YOU WANT TO SAMPLE AND STORE FILES
python3 WavPreprocess.py --dir '/work/ajgeglio/Tap_Data/Tony_01_25_23_data' 
--idx 9 --latency -0.14 --peak_height 0.0025 --plot --selectivity 0.134 --sample

The OUTPUT for each recording sould be a .wav file for each tap and 1 csv file of label data.
###########################################################################################
2) create_dataset.py - DATASET GENERATION
usage: create_dataset.py [-h] [--all_data] [--presplit] [--oversample] [--test_size TS_] [--prop PROP_] [--reshape_c_style] [--reshape_stack]
                         [--max_absolute] [--seed SEED_] [--plot_average_channel] [--all_dir96k ALL_DIR96K] [--train_dir96k TRAIN_DIR96K]
                         [--test_dir96k TEST_DIR96K] [--oversampled_dir INTERLEAVE_DIR] [--save_96k_all SAVE_96K_ALL] [--save_tr96k SAVE_TR96K]
                         [--save_te96k SAVE_TE96K]

Creates a dataset of taps from a wavfile of recorded typed sentences and associated neonode file
optional arguments:
  -h, --help            show this help message and exit
  --all_data            Used if 1 directory has all samples
  --presplit            Used if TRAIN and VALIDATION+EVALUATION samples are in separate directories
  --oversample          oversample class samples until 'all-exausted' to create class balance dataset
  --test_size TS_       validation/evaluation dataset split
  --prop PROP_          proportion of total data to use
  --reshape_c_style     reshapes the 8-channel data with numpy reshape, order = C
  --reshape_stack       flattens 8-channels by stacking to 8x length
  --max_absolute        maps 8-channel dataset to a 1 channel ds with the max absolute amplitude value
  --seed SEED_          set random seed
  --plot_average_channel
                        plot signal after average channel
  --all_dir96k ALL_DIR96K
                        directory of all 96k samples and labels
  --train_dir96k TRAIN_DIR96K
                        directory of train samples and labels
  --test_dir96k TEST_DIR96K
                        directory of train samples and labels
  --oversampled_dir INTERLEAVE_DIR
                        directory of the oversampled dataset
  --save_96k_all SAVE_96K_ALL
                        directory of dictionary dataset with train+validation+evaluation sets
  --save_tr96k SAVE_TR96K
                        directory of a training dataset only
  --save_te96k SAVE_TE96K
                        directory of a dictionary dataset with validation+evaluation sets
###########################################################################################
3) Wav2vec2_tap2key.py - DATASET GENERATION + MODEL TRAINER 
usage: Wav2vec2_tap2key.py [-h] [--dataset_dir DATASET_DIR] [--early_stop] [--early_patience EARLY_PATIENCE] [--epochs EPOCHS]

Wav2vec2_tap2key does training and evaluaiton of the tap sample dataset that has already been created. 
The model is a transformer based on Wav2vec2 architecture with Wav2Vec2-base weight initialialization. 
The fine tuning is done on 34 classes based on the virtual table-top keyboard and audio sample classification 
is supervised based on the labels created with WavePreprocess.py
optional arguments:
  -h, --help            show this help message and exit
  --dataset_dir DATASET_DIR
                        directory of the hugging face dataset
  --early_stop          early stopping using evaluation loss
  --early_patience EARLY_PATIENCE
                        number of worse evals before early stopping
  --epochs EPOCHS       number of epochs to run
##########################################################################################
4) predict_model_conf_matrix - EVALUATE
usage: predict_model_conf_matrix.py [-h] [--dataset_dir DATASET_DIR]

Run Evaluation prediction on trained Wav2Vec2 model and output a performance report and confusion matrix plot of the results

optional arguments:
  -h, --help            show this help message and exit
  --dataset_dir DATASET_DIR
                        directory of the evaluation dataset

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
% pip install psutil# Wav2Vec-tap2key
# Wav2Vec-tap2key
