#!/bin/bash


ROOT_DIR=`cd ../../../.. &&pwd`
SRC_PATH=$ROOT_DIR'/bert-pertub'
module load python/3.6.2
source $ROOT_DIR/env/bin/activate
cd $SRC_PATH && python3 merge_wiki_book.py