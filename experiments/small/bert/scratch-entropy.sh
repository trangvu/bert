#!/bin/bash

echo "pretrain small BERT entropy masking"
cd ../.. && bash ./pretrain-bert.sh bert-entropy-small --masking_strategy=entropy