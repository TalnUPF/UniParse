#!/bin/bash
#SBATCH --job-name="unip_ptb"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=50Gb
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

python kiperwasser_main.py --results_folder /homedtic/lperez/UniParse/saved_models/kiperwasser_en_ptb_BPE_by_sets --logging_file logging.log --do_training True --train_file /homedtic/lperez/UniParse/datasets/PTB_SD_3_3_0/train.gold.bpe.conll --dev_file /homedtic/lperez/UniParse/datasets/PTB_SD_3_3_0/dev.gold.bpe.conll --test_file /homedtic/lperez/UniParse/datasets/PTB_SD_3_3_0/test.gold.bpe.conll --output_file output_ptb_bpe.output --model_file model_bpe.model --vocab_file vocab_bpe.pkl --big_dataset True


# testing in local
# python kiperwasser_main.py --results_folder /home/lpmayos/code/UniParse/saved_models/prova --logging_file logging.log --do_training True --train_file /home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/train.gold.bpe.conll --dev_file /home/lpmayos/code//UniParse/datasets/PTB_SD_3_3_0/dev.gold.bpe.conll --test_file /home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/test.gold.bpe.conll --output_file output_1B_mini_bpe.output --model_file model_1B_mini_bpe.model --vocab_file vocab_1B_mini_bpe.pkl --big_dataset True --dev_mode False
