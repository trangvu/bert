#!/bin/bash
PREFIX=$1
echo "pretrain small BERT adversarial masking"
cd .. && bash ./domain_tuning_glue_all_adv_100_0.sh adapt-glue-all-adv$PREFIX