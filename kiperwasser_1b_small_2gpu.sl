#!/bin/bash
#SBATCH --job-name="1bsmall"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=20Gb
#SBATCH -p high
#SBATCH --gres=gpu:2

module load Tensorflow-gpu/1.12.0-foss-2017a-Python-3.6.4
module load scikit-learn/0.19.1-foss-2017a-Python-3.6.4
module load dynet/2.1-foss-2017a-Python-3.6.4-GPU-CUDA-9.0.176


# compiling decoders

python setup.py build_ext --inplace


# dynet config

#--dynet-gpus=1                            # Specify how many GPUs you want to use, if DyNet is compiled with CUDA.
#--dynet-devices=CPU,GPU:1,GPU:3,GPU:0      # Specify the CPU/GPU devices that you want to use.
#--dynet_mem=8000                           # DyNet runs by default with 512MB of memory, which is split evenly for the forward and backward steps, parameter storage as well as scratch use. This will be expanded automatically every time one of the pools runs out of memory.
#--dynet-profiling=2                       # Will output information about the amount of time/memory used by each node in the graph. Profile level with 0, 1 and 2.


# training params

dataset_version=small
do_training=True
big_dataset=True


# dataset and results folders and files

dataset_folder=/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/conll_bpe_$dataset_version
train_file=$dataset_folder/1b_train.bpe.conllu
dev_file=$dataset_folder/1b_dev.bpe.conllu
test_file=$dataset_folder/1b_test.bpe.conllu

results_folder=/homedtic/lperez/UniParse/saved_models/1b/small_2_gpu
output_file=$results_folder/output.out
logging_file=$results_folder/logging.log

model_file=model_1b.bpe.$dataset_version.model
vocab_file=vocab_1b.bpe.$dataset_version.pkl


# running the code

python kiperwasser_main.py --dynet_mem 8000 --dynet-gpus 2 --dynet-profiling 2 --results_folder $results_folder --logging_file $logging_file --do_training $do_training --train_file $train_file --dev_file $dev_file --test_file $test_file --output_file $output_file --model_file $model_file --vocab_file $vocab_file --big_dataset $big_dataset

