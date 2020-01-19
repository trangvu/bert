#!/bin/bash

echo "pretrain small BERT entropy masking"
bash ../../pretrain-bert.sh bert-entropy-small --masking_strategy=entropy