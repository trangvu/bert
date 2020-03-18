#!/bin/bash
PREFIX=$1
echo "pretrain small BERT entropy masking"
cd .. && bash ./domain_tuning_glue_all.sh adapt-glue-all-entropy$PREFIX --mask_strategy=entropy