#!/usr/bin/env bash
smux new-session --partition=m3g --time=2-00:00:00 --gres=gpu:1

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
    sbatch --job-name=scratch-rand-long submit-job-rqtp.sh  scratch-rand.sh -long
    sbatch --job-name=scratch-pos-long submit-job-rqtp.sh   scratch-pos.sh -long
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
    sbatch --job-name=electra-adv-625 --time=2-00:00:00 submit-job-m3g-V100.sh  scratch-electra-adv.sh 625

    ## Small epochs
    sbatch --job-name=ori-adv-epochs --time=2-00:00:00 submit-job-m3g-V100.sh scratch-adv.sh 2801
    sbatch --job-name=electra-adv-epochs --time=2-00:00:00 submit-job-m3g-V100.sh scratch-electra-adv.sh 2801
    sbatch --job-name=ori-rand-epochs submit-job-rqtp.sh  scratch-rand.sh 2801
    sbatch --job-name=ori-pos-epochs submit-job-rqtp.sh  scratch-pos.sh 2801
    sbatch --job-name=ori-entropy-epochs submit-job-rqtp.sh  scratch-entropy.sh 2801

    sbatch --job-name=electra-rand-epochs submit-job-rqtp.sh  scratch-electra-rand.sh 2801
    sbatch --job-name=electra-pos-epochs --time=2-00:00:00 submit-job-m3g-V100.sh  scratch-electra-pos.sh 2801
    sbatch --job-name=electra-entropy-epochs --time=2-00:00:00 submit-job-m3g-V100.sh scratch-electra-entropy.sh 2801


### Evaluate
    ## Evaluate multiple epochs experiment

    sbatch --job-name=glue-electra-pos-625 --time=1-00:00:00 submit-job-m3h-P100.sh evaluate_glue.sh electra-pos-small625  /home/xvuthith/da33_scratch/trang/masked-lm/models/electra-pos-small625
    sbatch --job-name=glue-electra-entropy-625 --time=1-00:00:00 submit-job-m3h-P100.sh evaluate_glue.sh electra-entropy-small625  /home/xvuthith/da33_scratch/trang/masked-lm/models/electra-entropy-small625
    sbatch --job-name=glue-electra-rand-625 --time=1-00:00:00 submit-job-m3h-P100.sh evaluate_glue.sh electra-rand-small625  /home/xvuthith/da33_scratch/trang/masked-lm/models/electra-rand-small625

    sbatch --job-name=glue-bert-adv-625 --time=1-00:00:00 submit-job-m3h-P100.sh evaluate_glue.sh bert-adv-small625  /home/xvuthith/da33_scratch/trang/masked-lm/models/bert-adv-small625


### NER Adaptation
    ## Domain tuning - CoNLL
    sbatch --job-name=adapt-rand-conll --time=1-00:00:00 submit-job-m3g-V100.sh adapt-rand.sh
    sbatch --job-name=adapt-pos-conll --time=1-00:00:00 submit-job-m3g-V100.sh adapt-pos.sh
    sbatch --job-name=adapt-entropy-conll submit-job-rqtp.sh adapt-entropy.sh
    sbatch --job-name=adapt-adv-conll submit-job-rqtp.sh adapt-adv.sh

    ## Domain tuning - Sentiment140
    sbatch --job-name=adapt-rand-tweet --time=1-00:00:00 submit-job-m3g-V100.sh adapt-rand.sh
    sbatch --job-name=adapt-pos-tweet --time=1-00:00:00 submit-job-m3g-V100.sh adapt-pos.sh
    sbatch --job-name=adapt-entropy-tweet --time=1-00:00:00 submit-job-m3g-V100.sh adapt-entropy.sh
    sbatch --job-name=adapt-adv-tweet --time=1-00:00:00 submit-job-m3g-V100.sh adapt-adv.sh

### Cross lingual transfer
    ## vi
    sbatch --job-name=transfer-vi-rand submit-job-rqtp.sh adapt-rand.sh vi
    sbatch --job-name=transfer-vi-entropy submit-job-rqtp.sh adapt-entropy.sh vi
    sbatch --job-name=transfer-vi-adv submit-job-rqtp.sh adapt-adv.sh vi

    ## es
    sbatch --job-name=transfer-es-rand submit-job-rqtp.sh adapt-rand.sh es
    sbatch --job-name=transfer-es-entropy submit-job-rqtp.sh adapt-entropy.sh es
    sbatch --job-name=transfer-es-adv submit-job-m3g-V100.sh adapt-adv.sh es

    ## ru
    sbatch --job-name=transfer-ru-rand submit-job-m3g-V100.sh adapt-rand.sh ru
    sbatch --job-name=transfer-ru-entropy submit-job-m3g-V100.sh adapt-entropy.sh ru
    sbatch --job-name=transfer-ru-adv submit-job-m3g-V100.sh adapt-adv.sh ru

    ## bg
    sbatch --job-name=transfer-bg-rand submit-job-m3g-V100.sh adapt-rand.sh bg
    sbatch --job-name=transfer-bg-entropy submit-job-m3g-V100.sh adapt-entropy-reza.sh bg
    sbatch --job-name=transfer-bg-adv submit-job-m3g-V100.sh adapt-adv-reza.sh bg

    ## el
    sbatch --job-name=transfer-el-rand submit-job-m3g-V100.sh adapt-rand-reza.sh el
    sbatch --job-name=transfer-el-entropy submit-job-m3g-V100.sh adapt-entropy-reza.sh el
    sbatch --job-name=transfer-el-adv submit-job-m3g-V100.sh adapt-adv-reza.sh el


python create-training-data.py --input_file=/home/gholamrh/da33/trang/masked-lm/train/glue_data/CoLA/train_wiki_75_25.txt  \
--do_lower_case --vocab_file=./config/en_uncase_vocab.txt --output_file=/home/gholamrh/da33/trang/masked-lm/train/glue_data/CoLA/train_wiki_75_25_cola.tfrecord

python3 run_adversarial_transfer.py --input_file=/home/xvuthith/da33_scratch/trang/masked-lm/train/wikidump-vi-128/train-00.tfrecord --output_dir=/home/xvuthith/da33_scratch/trang/masked-lm/models/cross-lingual-adv-vi --do_train=True --do_eval=True --do_lower_case --bert_config_file=/home/xvuthith/da33_scratch/trang/masked-lm/bert/config/base_bert_config.json --teacher_config_file=/home/xvuthith/da33_scratch/trang/masked-lm/bert/config/base_teacher_config.json --train_batch_size=32 --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=500000 --num_warmup_steps=50000 --learning_rate=5e-5 --vocab_file=/home/xvuthith/da33_scratch/trang/masked-lm/bert/config/vi_uncase_vocab.txt --teacher_update_rate=0.7 --teacher_rate_update_step=5000 --teacher_rate_decay=0.963 --teacher_learning_rate=5e-5 --init_checkpoint=/home/xvuthith/da33_scratch/trang/masked-lm/models/bert_base_uncased/bert_model.ckpt
