#!/usr/bin/env bash

### Prepare data
    ## Wikibook
    ./create-non-mask-data-book-all-shard.sh 128
    ./create-non-mask-data-wiki-dump-all-shard.sh 128


### Train from scratch
    ## English
    sbatch --job-name=scratch-rand submit-job-rqtp.sh  scratch-rand.sh
    sbatch --job-name=scratch-pos submit-job-rqtp.sh  scratch-pos.sh
    sbatch --job-name=scratch-entropy submit-job-rqtp.sh  scratch-entropy.sh
    sbatch --job-name=scratch-adv submit-job-rqtp.sh  scratch-adv.sh

    sbatch --job-name=scratch-adv submit-job-m3g-V100.sh  scratch-adv.sh
