#!/bin/bash
sentencepiece_build_folder="/home/lpmayos/code/sentencepiece-master/build"
test_folder="/home/lpmayos/code/UniParse/datasets/bpe_test"
model_name="m"

cd $sentencepiece_build_folder

echo "training..."
spm_train --input=$test_folder/babau1.txt,$test_folder/babau2.txt,$test_folder/babau3.txt --model_prefix=$model_name --vocab_size=80000 --character_coverage=1.0 --model_type=bpe
mv $model_name.model $test_folder
mv $model_name.vocab $test_folder

echo "encoding..."
spm_encode --model=$test_folder/$model_name.model --output_format=piece < $test_folder/babau4.txt > $test_folder/babau4_encoded.txt

echo "decoding..."
spm_decode --model=$test_folder/$model_name.model --input_format=piece < $test_folder/babau4_encoded.txt > $test_folder/babau4_decoded.txt
