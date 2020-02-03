#!/bin/bash
LANGUAGE=$1
PREFIX=$2
echo "pretrain small BERT entropy masking"
cd .. && bash ./run_transfer.sh cross-lingual-entropy$PREFIX $LANGUAGE --mask_strategy=entropy