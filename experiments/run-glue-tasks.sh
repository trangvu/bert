#!/bin/bash

EXP_NAME=$1
MODEL_DIR=$2

for (( index=0; index<=8; index+=1 )); do
    ./run-single-glue.sh $index $MODEL_DIR $EXP_NAME
done