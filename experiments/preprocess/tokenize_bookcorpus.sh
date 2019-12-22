#!/bin/bash
ROOT_DIR=`cd ../../../.. &&pwd`
#module load python/3.6.2
#source $ROOT_DIR/apex_env/bin/activate
mosesdecoder=/home/xvuthith/da33_scratch/tools/mosesdecoder

DATA_DIR="/home/xvuthith/da33_scratch/trang/masked-lm/data/sharded_bookcorpus"
LANGUAGE="en"
for index in 1 2 3 4 5 6 7 8 9 ; do
  echo "Process "$index
  TRAIN_FILE=$DATA_DIR"/bookcorpus_${index}"
  OUT_FILE=$DATA_DIR"/bookcorpus_tok_${index}"
  cat $TRAIN_FILE | \
  $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $LANGUAGE | \
  $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $LANGUAGE > $OUT_FILE
done