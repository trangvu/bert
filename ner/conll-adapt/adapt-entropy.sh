#!/bin/bash
PREFIX=$1
echo "pretrain small BERT entropy masking"
cd .. && bash ./domain_tuning_conll.sh adapt-conll-entropy$PREFIX --mask_strategy=entropy