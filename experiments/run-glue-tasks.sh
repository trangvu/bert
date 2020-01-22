#!/bin/bash

EXP_NAME=$1
MODEL_DIR=$2
OUTPUT_DIR=$3
SUBMIT_DIR=$OUTPUT_DIR'/'$EXP_NAME'_submission'
mkdir -p $SUBMIT_DIR
for (( index=0; index<=8; index+=1 )); do
    ./run-single-glue.sh $index $MODEL_DIR $EXP_NAME $OUTPUT_DIR
done

echo "Collect results to $SUBMIT_DIR"
TASKS=( MNLI QQP QNLI "SST-2" CoLA "STS-B" MRPC RTE WNLI )
set -x
for (( index=0; index<=8; index+=1 )); do
    TASK_NAME=${TASKS[$index]}
    cp $OUTPUT_DIR'/'$TASK_NAME'/'$TASK_NAME'.tsv' $SUBMIT_DIR
done
cp $OUTPUT_DIR'/'$TASK_NAME'/MNLI-m.tsv' $SUBMIT_DIR
cp $OUTPUT_DIR'/'$TASK_NAME'/MNLI-mm.tsv' $SUBMIT_DIR
zip $SUBMIT_DIR'.zip' $SUBMIT_DIR