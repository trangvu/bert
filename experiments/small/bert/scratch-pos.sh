#!/bin/bash

echo "pretrain small BERT pos masking"
cd ../.. && bash ./pretrain-bert.sh bert-pos-small --masking_strategy=pos