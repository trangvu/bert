#!/usr/bin/env bash

### Testing with 100K
    sbatch --job-name=test-rand submit-job-rqtp.sh  scratch-rand.sh 0306
    sbatch --job-name=test-rand submit-job-rqtp.sh  scratch-pos.sh 0306
    sbatch --job-name=test-rand submit-job-rqtp.sh  scratch-entropy.sh 0306