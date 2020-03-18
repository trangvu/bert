#!/bin/bash

EXP_NAME=$1
MODEL_DIR=$2

ROOT_DIR=`cd ../../.. &&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`
OUTPUT_DIR=$ROOT_DIR/results/$EXP_NAME/"run-"$DATE
mkdir -p $OUTPUT_DIR
echo "Evaluate $1 $2 on GLUE"
cd .. && bash ./run-glue-tasks.sh $EXP_NAME $MODEL_DIR $OUTPUT_DIR