#!/bin/bash
PREFIX=$1

echo "adapt rand conll"
cd .. && bash ./domain_tuning_glue_all.sh adapt-glue-all-pos$PREFIX --mask_strategy=pos