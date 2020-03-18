#!/bin/bash
PREFIX=$1
echo "pretrain small BERT adversarial masking"
cd ../.. && bash ./pretrain-adversarial-bert.sh bert-adv-small$PREFIX