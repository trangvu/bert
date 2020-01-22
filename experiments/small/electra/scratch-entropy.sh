#!/bin/bash

echo "pretrain small BERT entropy masking"
cd ../.. && bash ./pretrain-electra.sh bert-entropy-small --mask_strategy=entropy