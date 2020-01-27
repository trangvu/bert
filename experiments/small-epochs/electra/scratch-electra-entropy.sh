#!/bin/bash
PREFIX=$1

echo "pretrain small BERT entropy masking"
cd ../.. && bash ./pretrain-electra-epochs.sh electra-entropy-small-epoch$PREFIX --mask_strategy=entropy