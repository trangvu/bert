#!/usr/bin/env bash
smux new-session --partition=m3c --time=2-00:00:00 --gres=gpu:1

### Prepare data
    ## Wikibook
    ./create-non-mask-data-book-all-shard.sh 128
    ./create-non-mask-data-wiki-dump-all-shard.sh 128


    scp -r xvuthith@monarch.erc.monash.edu.au:/home/xvuthith/da33/trang/masked-lm/train/bookcorpus-128 .


### Train from scratch
    ## English
    sbatch --job-name=scratch-rand submit-job-rqtp.sh  scratch-rand.sh 2301
    sbatch --job-name=scratch-pos submit-job-rqtp.sh  scratch-pos.sh 2301
    sbatch --job-name=scratch-entropy submit-job-rqtp.sh  scratch-entropy.sh 2301
    sbatch --job-name=scratch-adv submit-job-rqtp.sh  scratch-adv.sh
    sbatch --job-name=scratch-adv-long  submit-job-dgx.sh  scratch-adv.sh

    sbatch --job-name=scratch-rand --time=2-00:00:00 submit-job-m3g-V100.sh  scratch-rand.sh 2301
    sbatch --job-name=scratch-pos --time=2-00:00:00 submit-job-m3g-V100.sh   scratch-pos.sh 2301
    sbatch --job-name=scratch-entropy --time=2-00:00:00 submit-job-m3g-V100.sh   scratch-entropy.sh 2301
    sbatch --job-name=scratch-adv --time=2-00:00:00 submit-job-m3g-V100.sh   scratch-adv.sh 2301
    sbatch --job-name=scratch-adv  submit-job-dgx.sh  scratch-adv.sh

    sbatch --job-name=electra-rand --time=2-00:00:00 submit-job-m3g-V100.sh  scratch-electra-rand.sh 2301
    sbatch --job-name=electra-pos --time=2-00:00:00 submit-job-m3g-V100.sh  scratch-electra-pos.sh 2301
    sbatch --job-name=electra-entropy --time=2-00:00:00 submit-job-m3g-V100.sh  scratch-electra-entropy.sh 2301
    sbatch --job-name=electra-adv --time=2-00:00:00 submit-job-m3g-V100.sh  scratch-electra-adv.sh 2301
    sbatch --job-name=electra-adv  submit-job-dgx.sh  scratch-electra-adv.sh

    ## Long
    sbatch --job-name=scratch-rand-long submit-job-m3g-V100.sh  scratch-rand.sh -long
    sbatch --job-name=scratch-pos-long submit-job-m3g-V100.sh   scratch-pos.sh -long
    sbatch --job-name=scratch-entropy-long submit-job-m3g-V100.sh   scratch-entropy.sh -long
    sbatch --job-name=scratch-adv-long submit-job-m3g-V100.sh scratch-adv.sh -long

    sbatch --job-name=electra-rand-long submit-job-m3g-V100.sh  scratch-electra-rand.sh -long
    sbatch --job-name=electra-pos-long submit-job-m3g-V100.sh  scratch-electra-pos.sh -long
    sbatch --job-name=electra-entropy-long submit-job-m3g-V100.sh  scratch-electra-entropy.sh -long
    sbatch --job-name=electra-adv-long submit-job-m3g-V100.sh  scratch-electra-adv.sh -long

    ## 6.25
    sbatch --job-name=scratch-rand-625 --time=2-00:00:00 submit-job-m3g-V100.sh  scratch-rand.sh 625
    sbatch --job-name=scratch-pos-625 --time=2-00:00:00 submit-job-m3g-V100.sh   scratch-pos.sh 625
    sbatch --job-name=scratch-entropy-625 --time=2-00:00:00 submit-job-m3g-V100.sh   scratch-entropy.sh 625
    sbatch --job-name=scratch-adv-625 --time=2-00:00:00 submit-job-m3g-V100.sh scratch-adv.sh 625

    sbatch --job-name=electra-rand-625 --mem-per-cpu=25000 submit-job-rqtp.sh  scratch-electra-rand.sh 625
    sbatch --job-name=electra-pos-625 --mem-per-cpu=25000 submit-job-rqtp.sh  scratch-electra-pos.sh 625
    sbatch --job-name=electra-entropy-625 --mem-per-cpu=25000 submit-job-rqtp.sh  scratch-electra-entropy.sh 625
    sbatch --job-name=electra-adv-625 --mem-per-cpu=25000 submit-job-rqtp.sh  scratch-electra-adv.sh 625

    ## Small epochs
    sbatch --job-name=ori-adv-epochs --time=2-00:00:00 submit-job-m3g-V100.sh scratch-adv.sh
    sbatch --job-name=electra-adv-epochs --time=2-00:00:00 submit-job-m3g-V100.sh scratch-electra-adv.sh
    sbatch --job-name=ori-rand-epochs submit-job-rqtp.sh  scratch-rand.sh
    sbatch --job-name=ori-pos-epochs submit-job-rqtp.sh  scratch-pos.sh
    sbatch --job-name=ori-entropy-epochs submit-job-rqtp.sh  scratch-entropy.sh

    sbatch --job-name=electra-rand-epochs submit-job-rqtp.sh  scratch-electra-rand.sh
    sbatch --job-name=electra-pos-epochs --time=2-00:00:00 submit-job-m3g-V100.sh  scratch-electra-pos.sh
    sbatch --job-name=electra-entropy-epochs --time=2-00:00:00 submit-job-m3g-V100.sh scratch-electra-entropy.sh


### Evaluate
    ## Evaluate multiple epochs experiment
    sbatch --job-name=glue-ori-rand-epochs --time=1-00:00:00 submit-job-m3h-P100.sh evaluate_glue.sh ori-rand-epochs  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-rand-small-epoch
    sbatch --job-name=glue-ori-pos-epochs --time=1-00:00:00 submit-job-m3h-P100.sh evaluate_glue.sh ori-pos-epochs  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-pos-small-epoch
    sbatch --job-name=glue-ori-entropy-epochs --time=1-00:00:00 submit-job-m3h-P100.sh evaluate_glue.sh ori-entropy-epochs  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-entropy-small-epoch
    sbatch --job-name=glue-ori-adv-epochs --time=1-00:00:00 submit-job-m3h-P100.sh evaluate_glue.sh ori-adv-epochs  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-adv-small-epoch

    sbatch --job-name=glue-electra-rand-epochs --time=1-00:00:00 submit-job-m3h-P100.sh evaluate_glue.sh electra-rand-epochs  /home/xvuthith/da33_scratch/trang/masked-lm/models/electra-rand-small-epoch
    sbatch --job-name=glue-electra-pos-epochs --time=1-00:00:00 submit-job-m3h-P100.sh evaluate_glue.sh electra-pos-epochs  /home/xvuthith/da33_scratch/trang/masked-lm/models/electra-pos-small-epoch
    sbatch --job-name=glue-electra-entropy-epochs --time=1-00:00:00 submit-job-m3h-P100.sh evaluate_glue.sh electra-entropy-epochs  /home/xvuthith/da33_scratch/trang/masked-lm/models/electra-entropy-small-epoch
    sbatch --job-name=glue-electra-adv-epochs --time=1-00:00:00 submit-job-m3h-P100.sh evaluate_glue.sh electra-adv-epochs  /home/xvuthith/da33_scratch/trang/masked-lm/models/electra-adv-small-epoch

    sbatch --job-name=glue-scr-rand --time=1-00:00:00 submit-job-m3g-V100.sh evaluate_glue.sh bert-rand  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-rand-small
    sbatch --job-name=glue-scr-pos --time=1-00:00:00 submit-job-m3g-V100.sh evaluate_glue.sh bert-pos  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-pos-small
    sbatch --job-name=glue-scr-entropy --time=1-00:00:00 submit-job-m3g-V100.sh evaluate_glue.sh bert-entropy  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-entropy-small
    sbatch --job-name=glue-scr-adv --time=1-00:00:00 submit-job-m3g-V100.sh evaluate_glue.sh bert-adv  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-adv-small

    sbatch --job-name=glue-scr-rand --time=1-00:00:00 submit-job-rqtp.sh evaluate_glue.sh bert-rand  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-rand-small
    sbatch --job-name=glue-scr-pos --time=1-00:00:00 submit-job-rqtp.sh evaluate_glue.sh bert-pos  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-pos-small
    sbatch --job-name=glue-scr-entropy --time=1-00:00:00 submit-job-rqtp.sh evaluate_glue.sh bert-entropy  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-entropy-small
    sbatch --job-name=glue-scr-adv --time=1-00:00:00 submit-job-rqtp.sh evaluate_glue.sh bert-adv  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-adv-small


    sbatch --job-name=glue-electra-entropy --time=5:00:00 submit-job-rqtp.sh evaluate_glue.sh electra-entropy-small  /home/xvuthith/da33_scratch/trang/masked-lm/models/electra-entropy-small
    sbatch --job-name=glue-bert-entropy --time=5:00:00 submit-job-rqtp.sh evaluate_glue.sh bert-entropy-small2201  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-entropy-small2201

    sbatch --job-name=glue-bert-pos --time=5:00:00 submit-job-m3g-V100.sh evaluate_glue.sh bert-pos-small0122  /scratch/da33/trang/masked-lm/models/bert-pos-small0122
    sbatch --job-name=glue-bert-rand --time=5:00:00 submit-job-m3g-V100.sh evaluate_glue.sh bert-rand-small0122  /scratch/da33/trang/masked-lm/models/bert-rand-small0122

    sbatch --job-name=glue-electra-rand --time=5:00:00 --mem-per-cpu=50000 submit-job-m3g-V100.sh evaluate_glue.sh electra-rand-small2301  /scratch/da33/trang/masked-lm/models/electra-rand-small2301
    sbatch --job-name=glue-electra-pos --time=5:00:00 submit-job-m3g-V100.sh evaluate_glue.sh electra-pos-small  /scratch/da33/trang/masked-lm/models/electra-pos-small

    sbatch --job-name=glue-bert-rand --time=5:00:00 --mem-per-cpu=50000 submit-job-m3g-V100.sh evaluate_glue.sh bert-rand-small2301  /scratch/da33/trang/masked-lm/models/bert-rand-small2301

    sbatch --job-name=glue-bert-pos --time=5:00:00 submit-job-m3g-V100.sh evaluate_glue.sh bert-rand-small2201  /scratch/da33/trang/masked-lm/models/bert-rand-small2201


sbatch --job-name=glue-bert-adv0122 --time=5:00:00 --mem-per-cpu=50000 submit-job-m3g-V100.sh  evaluate_glue.sh bert-adv-small0122  /scratch/da33/trang/masked-lm/models/bert-adv-small0122
sbatch --job-name=glue-bert-pos2301 --time=5:00:00 --mem-per-cpu=50000 submit-job-m3g-V100.sh  evaluate_glue.sh bert-pos-small2301  /scratch/da33/trang/masked-lm/models/bert-pos-small2301

sbatch --job-name=glue-electra-pos2301 --time=5:00:00 --mem-per-cpu=50000 submit-job-rqtp.sh  evaluate_glue.sh electra-entropy-small  /scratch/da33/trang/masked-lm/models/electra-entropy-small