#!/bin/bash

ROOT_DIR=`cd ../../.. &&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR'/bert'
DATA_ROOT="/mnt/lustre/projects/da33/trang/masked-lm"
module load python/3.6.2
module load cuda/9.0
module load cudnn/7.3.0-cuda9
source $ROOT_DIR/env/bin/activate

language=$1
index=$2

DATA_DIR=$DATA_ROOT"/wikidump/"$language"-raw"
INPUT="lower_tok_wiki"
DATASET="wikidump-"$language
VOCAB_FILE=$DATA_DIR"/"$language"_vocab.txt"
seq_len=128
echo "Processing shard ${index}"
INPUT_FILE="${DATA_DIR}/${INPUT}_${index}"
OUT_DIR="${DATA_ROOT}/train/wikidump-${language}-${seq_len}"
OUTPUT_FILE="train-${index}.tfrecord"
DEV_FILE="dev-${index}.tfrecord"
mkdir -p $OUT_DIR

cd $SRC_PATH && python3 create_pretraining_data.py \
--input_file=$INPUT_FILE \
--output_file=$OUT_DIR/$OUTPUT_FILE \
--dev_file=$OUT_DIR/$DEV_FILE \
--vocab_file=$VOCAB_FILE \
--do_lower_case=True \
--max_seq_length=$seq_len \
--max_predictions_per_seq=20 \
--masked_lm_prob=0.15 \
--random_seed=12345 \
--dupe_factor=1