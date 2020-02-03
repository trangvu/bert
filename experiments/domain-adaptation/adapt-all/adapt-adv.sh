#!/bin/bash
PREFIX=$1
echo "pretrain small BERT adversarial masking"
cd .. && bash ./domain_tuning_glue_all_adv.sh adapt-glue-all-adv$PREFIX