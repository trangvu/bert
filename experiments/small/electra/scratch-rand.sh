#!/bin/bash

echo "pretrain small BERT random masking"
cd ../.. && bash ./pretrain-electra.sh bert-rand-small --mask_strategy=random