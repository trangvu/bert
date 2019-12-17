#!/bin/bash

for (( index=0; index<=8; index+=1 )); do
    ./run-single-glue.sh $index $1
done