#!/bin/bash
PREFIX=$1
echo "pretrain small BERT adversarial masking"
cd .. && bash ./domain_tuning_glue_single_adv.sh adapt-glue-cola-adv$PREFIX cola
cd .. && bash ./domain_tuning_glue_single_adv.sh adapt-glue-sst2-adv$PREFIX sst2
cd .. && bash ./domain_tuning_glue_single_adv.sh adapt-glue-mrpc-adv$PREFIX mrpc
cd .. && bash ./domain_tuning_glue_single_adv.sh adapt-glue-stsb-adv$PREFIX stsb
cd .. && bash ./domain_tuning_glue_single_adv.sh adapt-glue-qqp-adv$PREFIX qqp
cd .. && bash ./domain_tuning_glue_single_adv.sh adapt-glue-qnli-adv$PREFIX qnli
cd .. && bash ./domain_tuning_glue_single_adv.sh adapt-glue-mnli-adv$PREFIX mnli
cd .. && bash ./domain_tuning_glue_single_adv.sh adapt-glue-rte-adv$PREFIX rte
cd .. && bash ./domain_tuning_glue_single_adv.sh adapt-glue-wnli-adv$PREFIX wnli