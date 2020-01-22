#!/bin/bash

echo "pretrain small BERT random masking"
cd ../.. && bash ./pretrain-bert.sh bert-rand-small --mask_strategy=random