#!/bin/bash

EXP_NAME=$1
MODEL_DIR=$2
echo "Evaluate $1 $2 on GLUE"
cd ../.. && bash ./run-glue-tasks.sh $EXP_NAME $MODEL_DIR