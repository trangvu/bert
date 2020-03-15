#!/usr/bin/env bash
### adapt rand
    sbatch --job-name=rand-50 submit-job-m3g-V100.sh adapt-rand.sh 50_50
    sbatch --job-name=rand-75 submit-job-m3g-V100.sh adapt-rand.sh 75_25
    sbatch --job-name=rand-100 submit-job-m3g-V100.sh adapt-rand.sh 100_0

### GLUE