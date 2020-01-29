#!/bin/bash
PREFIX=$1

echo "pretrain small BERT pos masking"
cd .. && bash ./domain_tuning_conll_electra.sh adapt-conll-electra-pos$PREFIX --mask_strategy=pos