#!/bin/bash
PREFIX=$1

echo "pretrain small BERT random masking"
cd .. && bash ./domain_tuning_conll_electra.sh adapt-conll-electra-rand$PREFIX --mask_strategy=random