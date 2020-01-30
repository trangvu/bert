#!/usr/bin/env bash

### Task-tuning on target
    @baseline
    sbatch --job-name=tweet-bert-baseline submit-job-rqtp.sh task_tuning.sh tweet-bert-baseline  /scratch/da33/trang/masked-lm//models/bert_base_uncased/bert_model.ckpt

    @tweet
    sbatch --job-name=tweet-adv-tgt submit-job-m3g-V100.sh task_tuning.sh tweet-adv-tgt  /scratch/da33/trang/masked-lm/models/adapt-tweet-adv
    sbatch --job-name=tweet-entropy-tgt submit-job-m3g-V100.sh task_tuning.sh tweet-entropy-tgt  /scratch/da33/trang/masked-lm/models/adapt-tweet-entropy
    sbatch --job-name=tweet-rand-tgt submit-job-m3g-V100.sh task_tuning.sh tweet-rand-tgt  /scratch/da33/trang/masked-lm/models/adapt-tweet-rand

    @conll
    sbatch --job-name=conll-adv-tgt submit-job-rqtp.sh task_tuning.sh conll-adv-tgt  /scratch/da33/trang/masked-lm/models/adapt-conll-adv
    sbatch --job-name=conll-entropy-tgt submit-job-rqtp.sh task_tuning.sh conll-entropy-tgt  /scratch/da33/trang/masked-lm/models/adapt-conll-entropy
    sbatch --job-name=conll-rand-tgt submit-job-rqtp.sh task_tuning.sh conll-rand-tgt  /scratch/da33/trang/masked-lm/models/adapt-conll-rand