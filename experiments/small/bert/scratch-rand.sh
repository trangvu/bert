#!/bin/bash
PREFIX=$1

echo "pretrain small BERT random masking"
cd ../.. && bash ./pretrain-bert.sh bert-rand-small$PREFIX --mask_strategy=random