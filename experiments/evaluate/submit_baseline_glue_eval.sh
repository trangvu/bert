#!/bin/bash

RUN_NAME=$1
for (( index=0; index<=4; index+=1 )); do
    sbatch --job-name=eval-glue-$index --mem-per-cpu=30000 submit-job-rqtp.sh evaluate_baseline.sh $index $RUN_NAME
done

for (( index=5; index<=8; index+=1 )); do
    sbatch --job-name=eval-glue-$index --mem-per-cpu=30000 submit-job-m3g-V100.sh evaluate_baseline.sh $index $RUN_NAME
done