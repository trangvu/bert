#!/bin/bash
PREFIX=$1

echo "pretrain small BERT random masking"
cd ../.. && bash ./pretrain-electra-epochs.sh electra-rand-small-epoch$PREFIX --mask_strategy=random