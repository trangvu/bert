#!/bin/bash
PREFIX=$1
echo "pretrain small BERT adversarial masking"
cd ../.. && bash ./pretrain-adv-bert.sh bert-adv-small$PREFIX