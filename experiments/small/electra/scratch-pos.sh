#!/bin/bash

echo "pretrain small BERT pos masking"
cd ../.. && bash ./pretrain-electra.sh bert-pos-small --mask_strategy=pos