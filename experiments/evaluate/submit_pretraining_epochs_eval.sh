#!/bin/bash

RUN_NAME=$1
for (( index=0; index<=8; index+=1 )); do
    sbatch --job-name=eval-pre-epochs-$index submit-job-m3h-P100.sh $index $RUN_NAME
done