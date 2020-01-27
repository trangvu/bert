#!/bin/bash
PREFIX=$1
echo "pretrain small BERT entropy masking"
cd ../.. && bash ./pretrain-bert-epochs.sh bert-entropy-small-epoch$PREFIX --mask_strategy=entropy