#!/bin/bash
TASK_SCRIPT=$1
EXP_NAME=$2
MODEL_DIR=$3
TASK_NAME=$4
RUN_NAME=$5

ROOT_DIR=`cd ../../.. &&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`
OUTPUT_DIR=$ROOT_DIR/results/$EXP_NAME/$RUN_NAME
mkdir -p $OUTPUT_DIR
echo "Evaluate $1 $2 on GLUE"
set -x
EVAL_RESULT=$RUN_NAME','$TASK_NAME
for (( index=1; index<=10; index+=1 )); do
  echo "RUN ${index}"
  OUTPUT_DIR_RUN=$OUTPUT_DIR"/run-${index}"
  bash ../$TASK_SCRIPT $MODEL_DIR $EXP_NAME $OUTPUT_DIR_RUN
  ACC=`cat $OUTPUT_DIR_RUN'/'$TASK_NAME'/eval_results.txt' | grep eval_accuracy | cut -d ' ' -f 3`
  if [ -z "$var" ]
  then
    ACC=`cat $OUTPUT_DIR_RUN'/'$TASK_NAME'/eval_results.txt' | grep pearson | cut -d ' ' -f 3`
  fi
  EVAL_RESULT=$EVAL_RESULT','$ACC
done
echo $EVAL_RESULT >> $OUTPUT_DIR'/eval_results.csv'