#!/bin/bash
PREFIX=$1

echo "adapt rand conll"
cd .. && bash ./domain_tuning_tweet.sh adapt-tweet-rand$PREFIX --mask_strategy=random