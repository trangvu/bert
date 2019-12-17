#!/bin/bash



ROOT_DIR=`cd ../.. &&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`
DATA_DIR=$ROOT_DIR'/data'

SRC_PATH=$ROOT_DIR'/bert'

OUT_DIR=$ROOT_DIR/results
mkdir -p $OUT_DIR

GLUE_DIR=$DATA_DIR/glue_data
SQUAD_DIR=$DATA_DIR/squad-2.0


module load cuda/9.0
module load cudnn/7.3.0-cuda9
module load python/3.6.2
source $ROOT_DIR/env/bin/activate

EXPS=( "SQUAD-2.0" )
index=0

EXP_NAME=${EXPS[$index]}
VOCAB_FILE="${DATA_DIR}/vocab.txt"
MODEL_NAME="model.ckpt"
MODEL_NAME="model.ckpt"
CONFIGS=( "base" "large" )
MASKS=( "rand" "pos" )
SRCS=( "wiki2" "wiki103" "book" "orig" )

mask_index=$1
config_index=$2
src_index=$3
#CONFIG=${CONFIGS[$config_index]}
BERT_BASE_DIR=$ROOT_DIR/models/${MASKS[$mask_index]}'_'${CONFIGS[$config_index]}'_'${SRCS[$src_index]}
CONFIG_DIR=$BERT_BASE_DIR
OUTPUT=$OUT_DIR/$EXP_NAME"-"${MASKS[$mask_index]}'-'${CONFIGS[$config_index]}'-'${SRCS[$src_index]}"-"$DATE

echo "Train classifier "$EXP_NAME
echo "Model "${MASKS[$mask_index]}'_'${CONFIGS[$config_index]}'_'${SRCS[$src_index]}

cd $SRC_PATH && python3 run_squad.py \
  --vocab_file=$VOCAB_FILE \
  --bert_config_file=$CONFIG_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/$MODEL_NAME \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=6 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=$OUTPUT \
  --version_2_with_negative=True

echo "Evaluate classifier "$EXP_NAME
python3 $SQUAD_DIR/evaluate-v2.0.py $SQUAD_DIR/dev-v2.0.json $OUTPUT/predictions.json --na-prob-file $OUTPUT/null_odds.json