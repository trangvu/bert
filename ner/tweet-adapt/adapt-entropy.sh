#!/bin/bash
PREFIX=$1
echo "pretrain small BERT entropy masking"
cd .. && bash ./domain_tuning_tweet.sh adapt-tweet-entropy$PREFIX --mask_strategy=entropy