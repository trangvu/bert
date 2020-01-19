#!/bin/bash

echo "pretrain small BERT random masking"
bash ../../pretrain-bert.sh bert-rand-small --masking_strategy=random