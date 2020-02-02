#!/usr/bin/env bash
### Domain-tuning
    sbatch --job-name=tweet-sim-adv-domain-10 submit-job-rqtp.sh adapt-adv.sh sim
    sbatch --job-name=tweet-sim-rand-domain-10 submit-job-rqtp.sh adapt-rand.sh sim
    sbatch --job-name=tweet-sim-entropy-domain-10 --time=1-00:00:00 submit-job-m3g-V100.sh adapt-entropy.sh sim

    sbatch --job-name=conll-adv-domain-10 submit-job-rqtp.sh adapt-adv.sh 0302
    sbatch --job-name=conll-rand-domain-10 submit-job-rqtp.sh adapt-rand.sh 0302
    sbatch --job-name=conll-entropy-domain-10 -submit-job-rqtp.sh adapt-entropy.sh 0302



### Task-tuning on target
    @baseline
    sbatch --job-name=tweet-bert-baseline-10 submit-job-rqtp.sh task_tuning.sh tweet-bert-baseline-10-new  /scratch/da33/trang/masked-lm//models/bert_base_uncased/bert_model.ckpt

    @tweet
    sbatch --job-name=tweet-adv-tgt-10 submit-job-rqtp.sh task_tuning.sh tweet-adv-tgt-10-new  /scratch/da33/trang/masked-lm/models/adapt-tweet-adv
    sbatch --job-name=tweet-entropy-tgt-10 submit-job-rqtp.sh task_tuning.sh tweet-entropy-tgt-10-new  /scratch/da33/trang/masked-lm/models/adapt-tweet-entropy
    sbatch --job-name=tweet-rand-tgt-10 submit-job-rqtp.sh task_tuning.sh tweet-rand-tgt-10-new  /scratch/da33/trang/masked-lm/models/adapt-tweet-rand

    @conll
    sbatch --job-name=conll-adv-tgt-10 submit-job-rqtp.sh task_tuning.sh conll-adv-tgt-10-new  /scratch/da33/trang/masked-lm/models/adapt-conll-adv
    sbatch --job-name=conll-entropy-tgt-10 submit-job-rqtp.sh task_tuning.sh conll-entropy-tgt-10-new  /scratch/da33/trang/masked-lm/models/adapt-conll-entropy
    sbatch --job-name=conll-rand-tgt-10 submit-job-rqtp.sh task_tuning.sh conll-rand-tgt-10-new  /scratch/da33/trang/masked-lm/models/adapt-conll-rand

   @conll source
   sbatch --job-name=conll-adv-src-10 submit-job-rqtp.sh task_tuning_source.sh conll-adv-src-10  /scratch/da33/trang/masked-lm/models/adapt-conll-adv
    sbatch --job-name=conll-entropy-src-10 submit-job-rqtp.sh task_tuning_source.sh conll-entropy-src-10  /scratch/da33/trang/masked-lm/models/adapt-conll-entropy
    sbatch --job-name=conll-rand-tgt-src submit-job-rqtp.sh task_tuning_source.sh conll-rand-src-10  /scratch/da33/trang/masked-lm/models/adapt-conll-rand