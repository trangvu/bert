#!/bin/bash
PREFIX=$1

echo "pretrain small BERT adversarial masking"
cd ../.. && bash ./pretrain-adv-electra.sh electra-adv-small$PREFIX