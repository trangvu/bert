#!/bin/bash
PREFIX=$1
echo "pretrain small BERT entropy masking"
cd .. && bash ./domain_tuning_glue_single.sh adapt-glue-cola-entropy$PREFIX cola --mask_strategy=entropy
cd .. && bash ./domain_tuning_glue_single.sh adapt-glue-sst2-entropy$PREFIX sst2 --mask_strategy=entropy
cd .. && bash ./domain_tuning_glue_single.sh adapt-glue-mrpc-entropy$PREFIX mrpc --mask_strategy=entropy
cd .. && bash ./domain_tuning_glue_single.sh adapt-glue-stsb-entropy$PREFIX stsb --mask_strategy=entropy
cd .. && bash ./domain_tuning_glue_single.sh adapt-glue-qqp-entropy$PREFIX qqp --mask_strategy=entropy
cd .. && bash ./domain_tuning_glue_single.sh adapt-glue-mnli-entropy$PREFIX mnli --mask_strategy=entropy
cd .. && bash ./domain_tuning_glue_single.sh adapt-glue-qnli-entropy$PREFIX qnli --mask_strategy=entropy
cd .. && bash ./domain_tuning_glue_single.sh adapt-glue-rte-entropy$PREFIX rte --mask_strategy=entropy
cd .. && bash ./domain_tuning_glue_single.sh adapt-glue-wnli-entropy$PREFIX wnli --mask_strategy=entropy