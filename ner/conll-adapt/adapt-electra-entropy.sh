#!/bin/bash
PREFIX=$1

echo "pretrain small BERT entropy masking"
cd .. && bash ./domain_tuning_conll_electra.sh adapt-conll-electra-entropy$PREFIX --mask_strategy=entropy