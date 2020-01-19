#!/bin/bash

echo "pretrain small BERT pos masking"
bash ../../pretrain-bert.sh bert-pos-small --masking_strategy=pos