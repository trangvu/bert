#!/bin/bash
PREFIX=$1
echo "pretrain small BERT adversarial masking"
cd .. && bash ./domain_tuning_conll_adv.sh adapt-conll-adv$PREFIX