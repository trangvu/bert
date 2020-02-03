#!/bin/bash

ROOT_DIR=`cd ../.. &&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR'/bert'
DATA_DIR=$ROOT_DIR'/data/glue-sim'

module load python/3.6.2
module load cuda/10.0
module load cudnn/7.6.5-cuda10.1
source $ROOT_DIR/env/bin/activate

INPUT="train"
DATASET="wikibook-tf-128"
VOCAB_FILE=$SRC_PATH'/config/en_uncase_vocab.txt'
EXP_NAME=$1
INPUT_PATTERN=$2
INPUT_FILE=`echo $DATA_DIR/*${INPUT_PATTERN}.tfrecord |  sed -r 's/[ ]+/,/g'`
OUT_DIR="${ROOT_DIR}/models/${EXP_NAME}"
mkdir -p $OUT_DIR

CONFIG_FILE=$SRC_PATH'/config/base_bert_config.json'
BERT_BASE_DIR=$ROOT_DIR'/models/bert_base_uncased/bert_model.ckpt'
shift
shift
PARAMS="$@"

set -x
cd $SRC_PATH && python3 run_pretraining.py \
--input_file=$INPUT_FILE \
--output_dir=$OUT_DIR \
--do_train=True \
--do_eval=True \
--do_lower_case \
--bert_config_file=$CONFIG_FILE \
--train_batch_size=32 \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--num_train_steps=16000 \
--num_warmup_steps=100 \
--learning_rate=5e-5 \
--vocab_file=$VOCAB_FILE \
--init_checkpoint=$BERT_BASE_DIR \
$PARAMS
