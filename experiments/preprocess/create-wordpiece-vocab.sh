#!/bin/bash


ROOT_DIR=`cd ../.. &&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR'/wikiextractor'
DATA_ROOT="/home/xvuthith/da33/trang/masked-lm"
DATA_DIR=$DATA_ROOT"/wikidump"

module load python/3.6.2
source $ROOT_DIR/env/bin/activate
language=$1
OUTPUT_DIR=$DATA_ROOT/wikidump/$language"-raw"

python learn_wordpiece.py --files $DATA_DIR/joints.txt --out $OUTPUT_DIR --name $language'_vocab.txt'