#!/bin/bash
#SBATCH --job-name="uniparse"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p short
#SBATCH --mem=32Gb
#SBATCH --gres=gpu:1                      # ask for a gpu resource


module load dynet/2.1-foss-2017a-Python-3.6.4
module load  scikit-learn/0.19.1-foss-2017a-Python-3.6.4
module load Tensorflow-gpu/1.5.0-foss-2017a-Python-3.6.4


# compiling decoders

python setup.py build_ext --inplace


# training and running model 

python kiperwasser_main.py --results_folder /homedtic/lperez/UniParse/saved_models/kiperwasser_en_ud_test_hpc --logging_file logging.log --do_training True --train_file /homedtic/lperez/UniParse/datasets/ud2.1/en-ud-train.conllu --dev_file /homedtic/lperez/UniParse/datasets/ud2.1/en-ud-dev.conllu --test_file /homedtic/lperez/UniParse/datasets/ud2.1/en-ud-test.conllu --output_file prova.output --model_file model.model --vocab_file vocab.pkl --dev_mode True --epochs 2 --dynet-devices GPU:0
