#!/bin/bash


ROOT_DIR=`cd ../.. &&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR'/bert'
DATA_DIR=$ROOT_DIR'/data'
OUT_DIR=$ROOT_DIR/results
mkdir -p $OUT_DIR

module load cuda/9.0
module load cudnn/7.3.0-cuda9
module load python/3.6.2
source $ROOT_DIR/env/bin/activate

INPUTS=( "wikitext-2.headless.txt" "wikitext-103.headless.txt" "bookcorpus.txt" )
OUTPUTS=( "tf_bert_pos_wikitext.tfrecord" "tf_bert_pos_wikitext-103.tfrecord" "tf_bert_pos_bookcorpus.tfrecord" )
index=$1
INPUT_FILE="${DATA_DIR}/${INPUTS[$index]}"
OUTPUT_FILE="${OUT_DIR}/${OUTPUTS[$index]}"
VOCAB_FILE="${DATA_DIR}/vocab.txt"

cd $SRC_PATH && python3 create_pretraining_data_pos.py \
  --input_file=$INPUT_FILE \
  --output_file=$OUTPUT_FILE \
  --vocab_file=$VOCAB_FILE \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5