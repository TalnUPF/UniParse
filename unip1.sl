#!/bin/bash
#SBATCH --job-name="unip1"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=100Gb
#SBATCH -p high
#SBATCH --gres=gpu:3                      # ask for a gpu resource
#SBATCH -C intel

module load Tensorflow-gpu/1.12.0-foss-2017a-Python-3.6.4
module load scikit-learn/0.19.1-foss-2017a-Python-3.6.4
module load dynet/2.1-foss-2017a-Python-3.6.4-GPU-CUDA-9.0.176


# compiling decoders

python setup.py build_ext --inplace


# training and running model 

python kiperwasser_main.py --results_folder /homedtic/lperez/UniParse/saved_models/kiperwasser_en_1b_unip1 --logging_file logging.log --do_training True --train_file /homedtic/lperez/UniParse/datasets/1-billion-benchmark/1B_train.conllu --dev_file /homedtic/lperez/UniParse/datasets/1-billion-benchmark/1B_dev.conllu --test_file /homedtic/lperez/UniParse/datasets/1-billion-benchmark/1B_test.conllu --output_file output_1b3.output --model_file model_1b3.model --vocab_file vocab.pkl --dynet-devices GPU:0,GPU:1,GPU:2
