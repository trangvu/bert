#!/bin/bash
PREFIX=$1

echo "adapt rand conll"
cd .. && bash ./domain_tuning_glue_all.sh adapt-glue-all-rand$PREFIX --mask_strategy=random