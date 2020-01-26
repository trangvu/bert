#!/bin/bash
PREFIX=$1

echo "pretrain small BERT adversarial masking"
cd ../.. && bash ./pretrain-adversarial-electra.sh electra-adv-small-epoch$PREFIX