#!/bin/bash
#SBATCH --job-name="unip1"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=30Gb
#SBATCH -p high
#SBATCH --gres=gpu:1                      # ask for a gpu resource
#SBATCH -C intel

module load Tensorflow-gpu/1.12.0-foss-2017a-Python-3.6.4
module load scikit-learn/0.19.1-foss-2017a-Python-3.6.4
module load dynet/2.1-foss-2017a-Python-3.6.4-GPU-CUDA-9.0.176


# compiling decoders

python setup.py build_ext --inplace


# training and running model 

python kiperwasser_main.py --results_folder /homedtic/lperez/UniParse/saved_models/kiperwasser_en_ptb_pieced --logging_file logging.log --do_training True --train_file /homedtic/lperez/UniParse/datasets/PTB_SD_3_3_0/train.gold.pieced.conll --dev_file /homedtic/lperez/UniParse/datasets/PTB_SD_3_3_0/dev.gold.pieced.conll --test_file /homedtic/lperez/UniParse/datasets/PTB_SD_3_3_0/test.gold.pieced.conll --output_file output_ptb_pieced.output --model_file model_ptb_pieced.model --vocab_file vocab_ptb_pieced.pkl --dynet-devices GPU:0
