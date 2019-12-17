#!/bin/bash

ROOT_DIR=`cd ../.. &&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR'/bert'
DATA_DIR=$ROOT_DIR'/results'
CONFIG_DIR=$SRC_PATH'/config'
OUT_DIR=$ROOT_DIR/results
mkdir -p $OUT_DIR

module load cuda/9.0
module load cudnn/7.3.0-cuda9
module load python/3.6.2
source $ROOT_DIR/env/bin/activate

TRAINING_DATA=( "tf_bert_pos_wikitext.tfrecord" "tf_bert_pos_wikitext-103.tfrecord" "tf_bert_pos_bookcorpus.tfrecord" )
EXPS=( wiki2 wiki103 bookcorpus)
index=$1

EXP_NAME=${EXPS[$index]}
INPUT_FILE=${DATA_DIR}/${TRAINING_DATA[$index]}

config_idx=$2
CONFIGS=( "base" "large" )
BATCH_SIZES=( 32 6 )
CONFIG=${CONFIGS[$config_idx]}
BATCH_SIZE=${BATCH_SIZES[$config_idx]}

OUTPUT=$OUT_DIR/$EXP_NAME"-pos-"$CONFIG"-"$DATE

set -x
cd $SRC_PATH && python3 run_pretraining.py \
  --input_file=$INPUT_FILE \
  --output_dir=$OUTPUT \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$CONFIG_DIR/${CONFIG}_bert_config.json \
  --train_batch_size=$BATCH_SIZE \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=1000000 \
  --num_warmup_steps=10000 \
  --learning_rate=1e-4
#  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt