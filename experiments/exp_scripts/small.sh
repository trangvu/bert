#!/usr/bin/env bash

### Testing with 100K
    sbatch --job-name=test-rand submit-job-rqtp.sh  scratch-rand.sh 0306
    sbatch --job-name=test-pos submit-job-rqtp.sh  scratch-pos.sh 0306
    sbatch --job-name=test-entropy submit-job-m3g-V100.sh  scratch-entropy.sh 0306
    sbatch --job-name=test-adv submit-job-m3g-V100.sh  scratch-adv.sh 0309