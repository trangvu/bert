#!/bin/bash
ROOT_DIR=/scratch/da33/trang/masked-lm
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR'/bert'

module load python/3.6.2
module load cuda/10.0
module load cudnn/7.6.5-cuda10.1
source $ROOT_DIR/env/bin/activate


MODEL_DIR=$ROOT_DIR'/models/bert_base_uncased/bert_model.ckpt'
OUT_DIR=$ROOT_DIR'/models/eng-mnli'
TASK_NAME='MNLI'
LEARNING_RATE=2e-5
TRAINING_EPOCHS=3
MODEL_NAME="bert_model.ckpt"
OUTPUT=$OUT_DIR/$TASK_NAME
VOCAB_FILE=$SRC_PATH'/config/en_uncase_vocab.txt'
CONFIG_FILE=$SRC_PATH'/config/base_bert_config.json'


set -x
    echo "Train classifier "TASK_NAME" with BERT base model "
    cd $SRC_PATH && python3 run_classifier.py \
      --task_name=$TASK_NAME \
      --do_train=true \
      --do_eval=true \
      --do_predict=true \
      --data_dir=$GLUE_DIR/$TASK_NAME \
      --vocab_file=$VOCAB_FILE \
      --bert_config_file=$CONFIG_FILE \
      --init_checkpoint=$MODEL_DIR \
      --max_seq_length=128 \
      --train_batch_size=32 \
      --learning_rate=$LEARNING_RATE \
      --do_lower_case=true \
      --num_train_epochs=$TRAINING_EPOCHS \
      --freeze_embedding=True \
      --output_dir=$OUTPUT
