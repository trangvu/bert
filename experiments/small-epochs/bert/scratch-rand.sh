#!/bin/bash
PREFIX=$1

echo "pretrain small BERT random masking"
cd ../.. && bash ./pretrain-bert.sh bert-rand-small-epoch$PREFIX --mask_strategy=random