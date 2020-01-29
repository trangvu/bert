#!/bin/bash
PREFIX=$1

echo "pretrain small BERT adversarial masking"
cd .. && bash ./domain_tuning_conll_electra_adv.sh adapt-conll-electra-adv$PREFIX