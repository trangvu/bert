#!/bin/bash
PREFIX=$1
echo "pretrain small BERT entropy masking"
cd .. && bash ./domain_tuning_glue_all_100_0.sh adapt-glue-all-entropy$PREFIX --mask_strategy=entropy