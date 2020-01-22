#!/bin/bash
PREFIX=$1

echo "pretrain small BERT random masking"
cd ../.. && bash ./pretrain-electra.sh bert-rand-small$PREFIX --mask_strategy=random