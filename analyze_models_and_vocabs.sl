#!/bin/bash
#SBATCH --job-name="analyze"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=10Gb
#SBATCH -p high


module load Python/3.6.4-foss-2017a


python analyze_models_and_vocabs.py
