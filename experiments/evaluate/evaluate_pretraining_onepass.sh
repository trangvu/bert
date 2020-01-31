#!/bin/bash
EXP_NAMES=('bert-rand-small625' 'bert-pos-small625' 'bert-entropy-small625' 'bert-adv-small625' 'electra-rand-small625' 'electra-pos-small625' 'electra-entropy-small625' 'electra-adv-small625-2901')


ROOT_DIR=`cd ../../.. &&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`

TASK_SCRIPTS=( 'run-cola.sh' 'run-sst2.sh' 'run-mrpc.sh' 'run-stsb.sh' 'run-qqp.sh' 'run-mnli.sh' 'run-qnli.sh' 'run-rte.sh' 'run-wnli.sh')
TASK_IDX=$1
RUN_NAME=$2

TASKS=(CoLA	"SST-2" MRPC "STS-B" QQP MNLI QNLI RTE WNLI)
set -x
TASK_NAME=${TASKS[$TASK_IDX]}
TASK_SCRIPT=${TASK_SCRIPTS[$TASK_IDX]}
for (( EXP_IDX=0; EXP_IDX<=7; EXP_IDX+=1 )); do
    EXP_NAME=${EXP_NAMES[$EXP_IDX]}
    echo "Evaluate $EXP_NAME on $TASK_NAME"
    MODEL_DIR='/scratch/da33/trang/masked-lm/models/'$EXP_NAME
    bash ./evaluate_glue_multiple_runs.sh $TASK_SCRIPT $EXP_NAME $MODEL_DIR $TASK_NAME $RUN_NAME
done