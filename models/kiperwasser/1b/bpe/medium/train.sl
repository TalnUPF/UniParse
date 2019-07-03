#!/bin/bash
#SBATCH --job-name="1bmedium"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=20Gb
#SBATCH -p high
#SBATCH --gres=gpu:2

module load Tensorflow-gpu/1.12.0-foss-2017a-Python-3.6.4
module load scikit-learn/0.19.1-foss-2017a-Python-3.6.4
module load dynet/2.1-foss-2017a-Python-3.6.4-GPU-CUDA-9.0.176


# dataset

dataset_folder=/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/conll_bpe_medium
train_file=$dataset_folder/1b_train.bpe.conllu
dev_file=/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/conll_bpe_mini/1b_dev.bpe.conllu
test_file=$dataset_folder/1b_test.bpe.conllu


# results folder

results_folder=$(pwd)
output_file=$results_folder/output.out
logging_file=$results_folder/logging.log
model_file=$results_folder/model.model
vocab_file=$results_folder/vocab.pkl


# running the code

do_training=True
big_dataset=True
only_words=False

cd /homedtic/lperez/UniParse

python setup.py build_ext --inplace  # compiling decoders

python kiperwasser_main.py --dynet_mem 8000 \
                           --dynet-gpus 2 \
                           --results_folder $results_folder \
                           --logging_file $logging_file \
                           --do_training $do_training \
                           --train_file $train_file \
                           --dev_file $dev_file \
                           --test_file $test_file \
                           --output_file $output_file \
                           --model_file $model_file \
                           --vocab_file $vocab_file \
                           --big_dataset $big_dataset \
                           --only_words $only_words