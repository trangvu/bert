#!/bin/bash
PREFIX=$1

echo "pretrain small BERT pos masking"
cd .. && bash ./domain_tuning_conll.sh adapt-conll-pos$PREFIX --mask_strategy=pos