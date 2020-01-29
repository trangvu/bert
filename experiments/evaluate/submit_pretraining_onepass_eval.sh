#!/bin/bash

RUN_NAME=$1
for (( index=0; index<=8; index+=1 )); do
    sbatch --job-name=eval-pre-epochs-$index --mem-per-cpu=20000 submit-job-m3h-P100.sh evaluate_pretraining_onepass.sh $index $RUN_NAME
done