#!/bin/bash
PREFIX=$1

echo "pretrain small BERT pos masking"
cd ../.. && bash ./pretrain-electra.sh bert-pos-small$PREFIX --mask_strategy=pos