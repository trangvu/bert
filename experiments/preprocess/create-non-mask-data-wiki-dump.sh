#!/bin/bash

ROOT_DIR=`cd ../../../.. &&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR'/bert'
DATA_ROOT="/home/xvuthith/da33/trang/masked-lm"
DATA_DIR=$DATA_ROOT"/data"
#
module load python/3.6.2
module load cuda/9.0
module load cudnn/7.3.0-cuda9
source $ROOT_DIR/env/bin/activate

INPUT="wiki_tok"
DATASET="wikidump-en"
VOCAB_FILE="/home/xvuthith/da33/trang/masked-lm/pretrained/cased_L-12_H-768_A-12/vocab.txt"
index=$1
seq_len=$2
echo "Processing shard ${index}"
INPUT_FILE="${DATA_ROOT}/wikidump/en-raw/${INPUT}_${index}"
OUT_DIR="${DATA_ROOT}/train/wikidump-en-${seq_len}"
OUTPUT_FILE="train-${index}.tfrecord"
DEV_FILE="dev-${index}.tfrecord"
mkdir -p "${DATA_ROOT}/train/wikidump-en-${seq_len}"

cd $SRC_PATH && python3 create_pretraining_data.py \
--input_file=$INPUT_FILE \
--output_file=$OUT_DIR/$OUTPUT_FILE \
--dev_file=$OUT_DIR/$DEV_FILE \
--vocab_file=$VOCAB_FILE \
--do_lower_case=True \
--max_seq_length=$seq_len \
--max_predictions_per_seq=80 \
--masked_lm_prob=0.15 \
--random_seed=12345 \
--dupe_factor=1