#!/bin/bash
LANGUAGE=$1
PREFIX=$2
echo "adapt rand conll"
cd .. && bash ./run_transfer.sh cross-lingual-rand$PREFIX $LANGUAGE --mask_strategy=random