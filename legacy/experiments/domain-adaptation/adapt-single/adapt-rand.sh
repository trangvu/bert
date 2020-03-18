#!/bin/bash
PREFIX=$1

echo "adapt rand conll"
cd .. && bash ./domain_tuning_glue_single.sh adapt-glue-cola-rand$PREFIX cola --mask_strategy=random
cd .. && bash ./domain_tuning_glue_single.sh adapt-glue-sst2-rand$PREFIX sst2 --mask_strategy=random
cd .. && bash ./domain_tuning_glue_single.sh adapt-glue-mrpc-rand$PREFIX mrpc --mask_strategy=random
cd .. && bash ./domain_tuning_glue_single.sh adapt-glue-stsb-rand$PREFIX stsb --mask_strategy=random
cd .. && bash ./domain_tuning_glue_single.sh adapt-glue-qqp-rand$PREFIX qqp --mask_strategy=random
cd .. && bash ./domain_tuning_glue_single.sh adapt-glue-mnli-rand$PREFIX mnli --mask_strategy=random
cd .. && bash ./domain_tuning_glue_single.sh adapt-glue-qnli-rand$PREFIX qnli --mask_strategy=random
cd .. && bash ./domain_tuning_glue_single.sh adapt-glue-rte-rand$PREFIX rte --mask_strategy=random
cd .. && bash ./domain_tuning_glue_single.sh adapt-glue-wnli-rand$PREFIX wnli --mask_strategy=random