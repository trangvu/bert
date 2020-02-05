#!/bin/bash

MODEL_DIR='/scratch/da33/trang/masked-lm/models/bert_base_uncased/bert_model.ckpt'
OUTPUT_DIR='/scratch/da33/trang/masked-lm/results/bert_base_uncased/glue'
SUBMIT_DIR=$OUTPUT_DIR'/'$EXP_NAME'submission'
mkdir -p $SUBMIT_DIR
for (( index=0; index<=8; index+=1 )); do
    ./run-single-glue.sh $index $MODEL_DIR glue $OUTPUT_DIR
done

echo "Collect results to $SUBMIT_DIR"
TASKS=( MNLI QQP QNLI "SST-2" CoLA "STS-B" MRPC RTE WNLI )
set -x
for (( i=0; i<=8; i+=1 )); do
    TASK_NAME=${TASKS[$i]}
    cp $OUTPUT_DIR'/'$TASK_NAME'/'$TASK_NAME'.tsv' $SUBMIT_DIR
done
cp $OUTPUT_DIR'/MNLI/MNLI-m.tsv' $SUBMIT_DIR
cp $OUTPUT_DIR'/MNLI/MNLI-mm.tsv' $SUBMIT_DIR
cp $OUTPUT_DIR'/MNLI/AX.tsv' $SUBMIT_DIR

echo "Collect eval result"
TASK_HEADER=""
TASK_ACC=""
for (( i=0; i<=8; i+=1 )); do
    TASK_NAME=${TASKS[$i]}
    TASK_HEADER="${TASK_HEADER}, ${TASK_NAME}"
    if [ $TASK_NAME = 'STS-B' ]; then
        ACC=`cat $OUTPUT_DIR'/'$TASK_NAME'/eval_results.txt' | grep pearson | cut -d ' ' -f 3`
    else
        ACC=`cat $OUTPUT_DIR'/'$TASK_NAME'/eval_results.txt' | grep eval_accuracy | cut -d ' ' -f 3`
    fi
    TASK_ACC="${TASK_ACC}, ${ACC}"
done
echo $TASK_HEADER >> $OUTPUT_DIR/eval_summary.txt
echo $TASK_ACC >> $OUTPUT_DIR/eval_summary.txt