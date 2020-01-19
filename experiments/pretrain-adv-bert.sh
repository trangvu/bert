ROOT_DIR=`cd ../.. &&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR'/bert'
DATA_ROOT=$ROOT_DIR'/train'
DATA_DIR=$DATA_ROOT
#
module load python/3.6.2
module load cuda/9.0
module load cudnn/7.3.0-cuda9
source $ROOT_DIR/env/bin/activate

INPUT="train"
DATASET="en-general-128"
VOCAB_FILE=$SRC_PATH'/config/en_uncase_vocab.txt'
EXP_NAME=$1
INPUT_FILE=`echo $DATA_DIR/$DATASET/train-* |  sed -r 's/[ ]+/,/g'`
OUT_DIR="${ROOT_DIR}/models/${EXP_NAME}"
mkdir -p $OUT_DIR

CONFIG_FILE=$SRC_PATH'/config/small_bert_config.json'
TEACHER_CONFIG_FILE=$SRC_PATH'/config/small_tearcher_config.json'
shift
PARAMS="$@"

set -x

cd $SRC_PATH && python3 run_adversarial_pretraining.py \
--input_file=$INPUT_FILE \
--output_dir=$OUT_DIR \
--do_train=True \
--do_eval=True \
--bert_config_file=$CONFIG_FILE \
--teacher_config_file=$TEACHER_CONFIG_FILE\
--train_batch_size=128 \
--max_seq_length=128 \
--max_predictions_per_seq=80 \
--num_train_steps=62500 \
--num_warmup_steps=10000 \
--learning_rate=5e-4 \
--vocab_file=$VOCAB_FILE \
$PARAMS
