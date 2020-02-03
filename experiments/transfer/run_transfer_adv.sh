#!/bin/bash

ROOT_DIR=`cd ../.. &&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR'/bert'

module load python/3.6.2
module load cuda/10.0
module load cudnn/7.6.5-cuda10.1
source $ROOT_DIR/env/bin/activate

INPUT="train"
EXP_NAME=$1
LANGUAGE=$2

DATASET="wikidump-$LANGUAGE-128"
VOCAB_FILE=$SRC_PATH'/config/'$LANGUAGE'_uncase_vocab.txt'
DATA_DIR=$ROOT_DIR'/train/'$DATASET
INPUT_FILE=`echo $DATA_DIR/train* |  sed -r 's/[ ]+/,/g'`
OUT_DIR="${ROOT_DIR}/models/${EXP_NAME}-${LANGUAGE}"
mkdir -p $OUT_DIR

CONFIG_FILE=$SRC_PATH'/config/base_bert_config.json'
TEACHER_CONFIG_FILE=$SRC_PATH'/config/base_teacher_config.json'
BERT_BASE_DIR=$ROOT_DIR'/models/bert_base_uncased/bert_model.ckpt'
shift
shift
PARAMS="$@"

set -x
cd $SRC_PATH && python3 run_adversarial_transfer.py \
--input_file=$INPUT_FILE \
--output_dir=$OUT_DIR \
--do_train=True \
--do_eval=True \
--do_lower_case \
--bert_config_file=$CONFIG_FILE \
--teacher_config_file=$TEACHER_CONFIG_FILE \
--train_batch_size=32 \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--num_train_steps=500000 \
--num_warmup_steps=50000 \
--learning_rate=5e-5 \
--vocab_file=$VOCAB_FILE \
--teacher_update_rate=0.7 \
--teacher_rate_update_step=5000 \
--teacher_rate_decay=0.963 \
--teacher_learning_rate=5e-5 \
--init_checkpoint=$BERT_BASE_DIR \
$PARAMS
