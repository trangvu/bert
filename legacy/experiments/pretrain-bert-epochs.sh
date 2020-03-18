ROOT_DIR=`cd ../.. &&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR'/bert'
DATA_ROOT=$ROOT_DIR'/train'
DATA_DIR=$DATA_ROOT
#
module load python/3.6.2
module load cuda/10.0
module load cudnn/7.6.5-cuda10.1
source $ROOT_DIR/env/bin/activate

INPUT="train"
DATASET="wikibook-tf-128"
VOCAB_FILE=$SRC_PATH'/config/en_uncase_vocab.txt'
EXP_NAME=$1
INPUT_FILE=$DATA_DIR'/'$DATASET'/train-13.tfrecord'
OUT_DIR="${ROOT_DIR}/models/${EXP_NAME}"
mkdir -p $OUT_DIR

CONFIG_FILE=$SRC_PATH'/config/small_bert_config.json'
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
--train_batch_size=128 \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--num_train_steps=100000 \
--num_warmup_steps=10000 \
--learning_rate=5e-4 \
--vocab_file=$VOCAB_FILE \
$PARAMS
