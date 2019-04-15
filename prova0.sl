#!/bin/bash
#SBATCH --job-name="prova0"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=32Gb
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

python kiperwasser_main.py --results_folder /homedtic/lperez/UniParse/saved_models/prova0 --logging_file logging.log --do_training True --train_file /homedtic/lperez/UniParse/datasets/PTB_SD_3_3_0/train.gold.bpe.conll --dev_file /homedtic/lperez/UniParse/datasets/PTB_SD_3_3_0/dev.gold.bpe.conll --test_file /homedtic/lperez/UniParse/datasets/PTB_SD_3_3_0/test.gold.bpe.conll --output_file output_ptb_bpe.output --model_file model_bpe.model --vocab_file vocab_bpe.pkl
