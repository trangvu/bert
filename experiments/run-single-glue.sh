#!/bin/bash
module load cuda/10.1
module load cudnn/7.6.5-cuda10.1
#module load anaconda/2019.03-Python3.7-gcc5
#conda activate jiant
module load gcc/5.4.0
export BERT_BASE_DIR=/scratch/da33/trang/masked-lm/models/bert_base_uncased
export GLUE_DIR=/project/da33/data_nlp/natural_language_understanding
ROOT_DIR=`cd ../.. &&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR'/bert'

OUT_DIR=$ROOT_DIR/results
mkdir -p $OUT_DIR
index=$1
EXP_NAME=${2:-glue_exp}
TASKS=( MNLI QQP QNLI "SST-2" CoLA "STS-B" MRPC RTE WNLI )

TASK_NAME=${TASKS[$index]}
MODEL_NAME="bert_model.ckpt"
OUTPUT=$OUT_DIR/$EXP_NAME/"run-"$DATE/$TASK_NAME

set -x

echo "Train classifier "TASK_NAME" with BERT base model "
cd $SRC_PATH && python3 run_classifier.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/$TASK_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
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