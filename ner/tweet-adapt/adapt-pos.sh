#!/bin/bash
PREFIX=$1

echo "pretrain small BERT pos masking"
cd .. && bash ./domain_tuning_tweet.sh adapt-tweet-pos$PREFIX --mask_strategy=pos