#!/bin/bash
PREFIX=$1

echo "pretrain small BERT pos masking"
cd ../.. && bash ./pretrain-bert.sh bert-pos-small--epoch$PREFIX --mask_strategy=pos