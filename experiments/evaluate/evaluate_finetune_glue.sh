#!/bin/bash
EXP_NAMES=('adapt-glue-all-adv50_50' 'adapt-glue-all-entropy50_50' 'adapt-glue-all-rand50_50' 'adapt-glue-all-pos50_50' 'adapt-glue-all-adv75_25' 'adapt-glue-all-entropy75_25' 'adapt-glue-all-rand75_25' 'adapt-glue-all-pos75_25' )


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