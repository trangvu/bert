#!/usr/bin/env bash
smux new-session --partition=m3g --time=2-00:00:00 --gres=gpu:1

### Prepare data
    ## Wikibook
    ./create-non-mask-data-book-all-shard.sh 128
    ./create-non-mask-data-wiki-dump-all-shard.sh 128


    scp -r xvuthith@monarch.erc.monash.edu.au:/home/xvuthith/da33/trang/masked-lm/train/bookcorpus-128 .


### Train from scratch
    ## English
    sbatch --job-name=scratch-rand submit-job-rqtp.sh  scratch-rand.sh
    sbatch --job-name=scratch-pos submit-job-rqtp.sh  scratch-pos.sh
    sbatch --job-name=scratch-entropy submit-job-rqtp.sh  scratch-entropy.sh
    sbatch --job-name=scratch-adv submit-job-rqtp.sh  scratch-adv.sh
    sbatch --job-name=scratch-adv  submit-job-dgx.sh  scratch-adv.sh

    sbatch --job-name=electra-rand --time=2-00:00:00 submit-job-m3g-V100.sh  scratch-rand.sh
    sbatch --job-name=electra-pos --time=2-00:00:00 submit-job-m3g-V100.sh  scratch-pos.sh
    sbatch --job-name=electra-entropy --time=2-00:00:00 submit-job-m3g-V100.sh  scratch-entropy.sh
    sbatch --job-name=electra-adv --time=2-00:00:00 submit-job-m3g-V100.sh  scratch-adv.sh
    sbatch --job-name=electra-adv  submit-job-dgx.sh  scratch-adv.sh



### Evaluate
    sbatch --job-name=glue-scr-rand --time=1-00:00:00 submit-job-m3g-V100.sh evaluate_glue.sh bert-rand  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-rand-small
    sbatch --job-name=glue-scr-pos --time=1-00:00:00 submit-job-m3g-V100.sh evaluate_glue.sh bert-pos  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-pos-small
    sbatch --job-name=glue-scr-entropy --time=1-00:00:00 submit-job-m3g-V100.sh evaluate_glue.sh bert-entropy  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-entropy-small
    sbatch --job-name=glue-scr-adv --time=1-00:00:00 submit-job-m3g-V100.sh evaluate_glue.sh bert-adv  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-adv-small

    sbatch --job-name=glue-scr-rand --time=1-00:00:00 submit-job-rqtp.sh evaluate_glue.sh bert-rand  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-rand-small
    sbatch --job-name=glue-scr-pos --time=1-00:00:00 submit-job-rqtp.sh evaluate_glue.sh bert-pos  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-pos-small
    sbatch --job-name=glue-scr-entropy --time=1-00:00:00 submit-job-rqtp.sh evaluate_glue.sh bert-entropy  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-entropy-small
    sbatch --job-name=glue-scr-adv --time=1-00:00:00 submit-job-rqtp.sh evaluate_glue.sh bert-adv  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-adv-small