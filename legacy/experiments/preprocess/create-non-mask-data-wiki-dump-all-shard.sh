#!/bin/bash

seq_len=$1
for (( index=0; index<=13; index+=1 )); do
    sbatch --job-name=wikishard-$index --mem-per-cpu=70000 submit-job-monarch-gpu.sh ./create-non-mask-data-wiki-dump.sh $index $seq_len
done