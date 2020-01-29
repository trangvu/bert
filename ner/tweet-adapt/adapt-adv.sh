#!/bin/bash
PREFIX=$1
echo "pretrain small BERT adversarial masking"
cd .. && bash ./domain_tuning_tweet_adv.sh adapt-tweet-adv$PREFIX