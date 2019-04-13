#!/bin/bash
#SBATCH --job-name="bpe"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p medium
#SBATCH --mem=30Gb

module load sentencepiece/0.1.81-foss-2017a-Python-3.6.4
python datasets/bpe_demo/parse1b_pipeline.py
