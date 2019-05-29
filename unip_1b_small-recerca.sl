#!/bin/bash
#SBATCH --job-name="1bsmall"
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

python kiperwasser_main.py --results_folder /homedtic/lperez/UniParse/saved_models/kiperwasser_en_1B_small --logging_file logging.log --do_training True --train_file /homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/conll_bpe_small/1b_train.conllu --dev_file /homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/conll_bpe_small/1b_dev.conllu --test_file /homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/conll_bpe_small/1b_test.conllu --output_file output_1B_bpe.output --model_file model_1B_bpe.model --vocab_file vocab_1B_bpe.pkl --dynet-devices GPU:0 --dynet-mem 80000 --big_dataset True
