#!/bin/bash
PREFIX=$1

echo "adapt rand conll"
cd .. && bash ./domain_tuning_conll.sh adapt-conll-rand$PREFIX --mask_strategy=random