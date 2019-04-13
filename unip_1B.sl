#!/bin/bash
#SBATCH --job-name="unip1B"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=32Gb
#SBATCH -p short
#SBATCH --gres=gpu:2

module load Tensorflow-gpu/1.12.0-foss-2017a-Python-3.6.4
module load scikit-learn/0.19.1-foss-2017a-Python-3.6.4
module load dynet/2.1-foss-2017a-Python-3.6.4-GPU-CUDA-9.0.176
# module load sentencepiece/0.1.81-foss-2017a-Python-3.6.4
# module load conllu/1.3.1-foss-2017a-Python-3.6.4


# compiling decoders

python setup.py build_ext --inplace


# training and running model 

python kiperwasser_main.py --results_folder /homedtic/lperez/UniParse/saved_models/kiperwasser_en_1B --logging_file logging.log --do_training True --train_file /homedtic/lperez/UniParse/datasets/1B/train.pieced.conll --dev_file /homedtic/lperez/UniParse/datasets/1B/dev.pieced.conll --test_file /homedtic/lperez/UniParse/datasets/1B/test.pieced.conll --output_file output_1B_pieced.output --model_file model_1B_pieced.model --vocab_file vocab_1B_pieced.pkl --dynet-devices GPU:0,GPU:1
