#!/bin/bash

#MASK_STRATEGY=( "rand" "pos" )
MASK_IDX=3
CONFIG_FILE=$SRC_PATH"/config/rnn_agent_base.json"
echo " FINETUNE BERT BASE MODELS ON WIKIBOOK DATASET WITH AGENT BASED MASKING, BERT VOCABULARY"
bash ../pretrain-bert.sh $MASK_IDX $CONFIG_IDX  --agent_config_file $CONFIG_FILE