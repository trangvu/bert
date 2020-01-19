#!/usr/bin/env bash

### Prepare data
    ## Wikibook
    ./create-non-mask-data-book-all-shard.sh 128
    ./create-non-mask-data-wiki-dump-all-shard.sh 12


### Train from scratch
    ## English
    submit-job-rqtp.sh --job-name=scratch-rand scratch-rand.sh
    submit-job-rqtp.sh --job-name=scratch-pos scratch-pos.sh
    submit-job-rqtp.sh --job-name=scratch-entropy scratch-entropy.sh
    submit-job-rqtp.sh --job-name=scratch-adv scratch-adv.sh
