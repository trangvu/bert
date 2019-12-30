#!/bin/bash

seq_len=512

for (( index=1; index<=10; index+=1 )); do
    sbatch --job-name=book-shard-$index --mem-per-cpu=70000 submit-job-gpu.sh ./create-non-mask-data-bert-book.sh $index  $seq_len
done