#!/bin/bash
PREFIX=$1
echo "pretrain small BERT adversarial masking"
cd ../.. && bash ./pretrain-adversarial-bert-epochs.sh bert-adv-small-epochs$PREFIX