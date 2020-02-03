#!/bin/bash
LANGUAGE=$1
PREFIX=$2
echo "pretrain small BERT adversarial masking"
cd .. && bash ./run_transfer_adv.sh cross-lingual-adv$PREFIX $LANGUAGE