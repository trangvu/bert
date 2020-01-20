#!/bin/bash
ROOT_DIR=`cd ../.. &&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR'/bert'

module load python/3.6.2
module load cuda/10.0
module load cudnn/7.6.5-cuda10.1
source $ROOT_DIR/env/bin/activate

OUT_DIR=$ROOT_DIR/results
mkdir -p $OUT_DIR
index=$1
MODEL_DIR=$2
EXP_NAME=$3
TASKS=( MNLI QQP QNLI "SST-2" CoLA "STS-B" MRPC RTE WNLI )
LEARNING_RATES=( 3 3 3 3 3 10 3 10 3 )

TASK_NAME=${TASKS[$index]}
LEARNING_RATE=${LEARNING_RATES[$index]}
MODEL_NAME="bert_model.ckpt"
OUTPUT=$OUT_DIR/$EXP_NAME/"run-"$DATE/$TASK_NAME
VOCAB_FILE=$SRC_PATH'/config/en_uncase_vocab.txt'
CONFIG_FILE=$SRC_PATH'/config/small_bert_config.json'


set -x

echo "Train classifier "TASK_NAME" with BERT base model "
cd $SRC_PATH && python3 run_classifier.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/$TASK_NAME \
  --vocab_file=$VOCAB_FILE \
  --bert_config_file=$CONFIG_FILE \
  --init_checkpoint=MODEL_DIR/model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=3e-4 \
  --num_train_epochs=$LEARNING_RATE \
  --output_dir=$OUTPUT


echo "Evaluate classifier "$EXP_NAME
MODEL_CKP=`cat $OUTPUT/checkpoint | grep "^model_checkpoint_path" | cut -d' ' -f2  | tr -d '"'`

cd $SRC_PATH && python run_classifier.py \
  --task_name=$TASK_NAME \
  --do_predict=true \
  --data_dir=$GLUE_DIR/$TASK_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT/$MODEL_CKP \
  --max_seq_length=128 \
  --output_dir=$OUTPUT