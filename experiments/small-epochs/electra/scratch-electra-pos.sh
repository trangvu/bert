#!/bin/bash
PREFIX=$1

echo "pretrain small BERT pos masking"
cd ../.. && bash ./pretrain-electra.sh electra-pos-small-epoch$PREFIX --mask_strategy=pos