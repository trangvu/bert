#!/bin/bash
PREFIX=$1
echo "pretrain small BERT entropy masking"
cd ../.. && bash ./pretrain-bert.sh bert-entropy-small$PREFIX --mask_strategy=entropy