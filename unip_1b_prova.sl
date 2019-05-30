#!/bin/bash
#SBATCH --job-name="1bprova"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=20Gb
#SBATCH -p high
#SBATCH --gres=gpu:1

module load Tensorflow-gpu/1.12.0-foss-2017a-Python-3.6.4
module load scikit-learn/0.19.1-foss-2017a-Python-3.6.4
module load dynet/2.1-foss-2017a-Python-3.6.4-GPU-CUDA-9.0.176
# module load sentencepiece/0.1.81-foss-2017a-Python-3.6.4
# module load conllu/1.3.1-foss-2017a-Python-3.6.4


# compiling decoders

python setup.py build_ext --inplace


# training and running model 


dataset_version=mini
dataset_folder=/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/conll_bpe_$dataset_version

train_file=$dataset_folder/1b_train.bpe.conllu
dev_file=$dataset_folder/1b_dev.bpe.conllu
test_file=$dataset_folder/1b_test.bpe.conllu

results_folder=/homedtic/lperez/UniParse/saved_models/kiperwasser_en_1B_$dataset_version
output_file=$results_folder/output_1B_bpe_$dataset_version.output
logging_file=logging.log

do_training=True

model_file=model_1B.bpe.$dataset_version.model
vocab_file=vocab_1B.bpe.$dataset_version.pkl

dynet_devices=GPU:0
dynet_mem=8000

big_dataset=False
dev_mode=True

python kiperwasser_main.py --dev_mode $dev_mode --results_folder $results_folder --logging_file $logging_file --do_training $do_training --train_file $train_file --dev_file $dev_file --test_file $test_file --output_file $output_file --model_file $model_file --vocab_file $vocab_file --dynet-devices $dynet_devices --dynet-mem $dynet_mem --big_dataset $big_dataset


# testing in local (remember: there is no dev mode if you choose big_dataset = True)
# python kiperwasser_main.py --results_folder /home/lpmayos/code/UniParse/saved_models/kiperwasser_en_1B_mini --logging_file logging.log --do_training True --train_file /home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/conll_bpe_mini/1b_train.bpe.conllu --dev_file /home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/conll_bpe_mini/1b_dev.bpe.conllu --test_file /home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/conll_bpe_mini/1b_test.bpe.conllu --output_file output_1B_mini_bpe.output --model_file model_1B_mini_bpe.model --vocab_file vocab_1B_mini_bpe.pkl --big_dataset True

