#!/bin/bash

ROOT_DIR=`cd ../.. &&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR'/bert'
DATA_DIR=$ROOT_DIR'/bert/ner/data'

module load python/3.6.2
module load cuda/10.0
module load cudnn/7.6.5-cuda10.1
source $ROOT_DIR/env/bin/activate

INPUT="train"
DATASET="wikibook-tf-128"
VOCAB_FILE=$SRC_PATH'/config/en_uncase_vocab.txt'
EXP_NAME=$1
MODEL_DIR=$2
OUT_DIR="${ROOT_DIR}/results/twitter/${EXP_NAME}"
mkdir -p $OUT_DIR

CONFIG_FILE=$SRC_PATH'/config/base_bert_config.json'

set -x
cd $SRC_PATH && python3 run_ner.py \
--data_dir=$DATA_DIR \
--output_dir=$OUT_DIR \
--do_train=True \
--do_eval=True \
--do_predict=True \
--do_lower_case \
--bert_config_file=$CONFIG_FILE \
--train_batch_size=32 \
--max_seq_length=128 \
--num_train_epochs=3 \
--num_warmup_steps=100 \
--learning_rate=5e-5 \
--vocab_file=$VOCAB_FILE \
--init_checkpoint=$MODEL_DIR