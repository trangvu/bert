#!/bin/bash

ROOT_DIR=`cd ../../../.. &&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR'/bert-pertub'
DATA_ROOT="/scratch/da33/trang/masked-lm"
DATA_DIR=$DATA_ROOT"/data"
#
#module load python/3.6.2
#source $ROOT_DIR/env/bin/activate

INPUT="wiki_tok"
DATASET="wikidump-en"
OUTPUT_FILE="train"
VOCAB_FILE="${DATA_DIR}/bert_vocab_uncase.txt"
index=$1
seq_len=$2
echo "Processing shard ${index}"
INPUT_FILE="${DATA_ROOT}/wikidump/en-raw/${INPUT}_${index}"
OUT_DIR="${DATA_ROOT}/train/wikidump-en-${seq_len}/${DATASET}-shard-${index}"
mkdir -p "${DATA_ROOT}/train/wikidump-en-${seq_len}"
set -x
rm -r -f $OUT_DIR

cd $SRC_PATH && python3 create_pretraining_data.py \
  --input_file=$INPUT_FILE \
  --output_dir=$OUT_DIR \
  --output_file=$OUTPUT_FILE \
  --vocab_file=$VOCAB_FILE  \
  --do_lower_case \
  --max_seq_length=$seq_len \
  --max_predictions_per_seq=80 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=1 \
  --mask_strategy=none \
  --max_shard_size=500000 \
  --dev_size=10 \
  --meta_dev_size=10 \
  --dev_file="dev" \
  --in_memory