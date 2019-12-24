#!/bin/bash


ROOT_DIR=`cd ../.. &&pwd`
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR'/wikiextractor'
DATA_ROOT="/home/xvuthith/da33/trang/masked-lm"
DATA_DIR=$DATA_ROOT"/wikidump"

module load python/3.6.2
source $ROOT_DIR/env/bin/activate
language=$1
OUTPUT=$DATA_ROOT/wikidump/$language-$DATE
mkdir -p $OUTPUT
INPUT_FILE=$DATA_DIR/${language}wiki-latest-pages-articles.xml.bz2

cd $SRC_PATH && python3 WikiExtractor.py --output $OUTPUT \
        --bytes 1000M \
        --json \
        --no_templates \
        -it b,a,div,center,section,span,blockquote,font,br,small,large,u,poem,big,ins,p,q,bdi,name,ul,li,ol,includeonly,onlyinclude,link,h1,h2,h3,h4,h5,h6,tt,i,abbr,samp,strong,em,s,var\
        --discard_elements gallery,timeline,noinclude,references,nowiki,templatestyles,cite,onlyinclude,source,grammarly-btn,link,mapframe,chem,templatestyles,pre,table,caption,imagemap,img,sub,sup,menu,dir,select,option,pre,form,table,tr,td,th,dd,dl,ref,dt,ruby,rp,rt,math \
        --filter_disambig_pages \
        --processes 4 \
        --log_file $OUTPUT/log.out \
        $INPUT_FILE

BERT_SRC=$ROOT_DIR'/bert-pertub/experiments'
INPUT_DIR=$OUTPUT/AA
OUTPUT_DIR=$DATA_ROOT/wikidump/$language"-raw"
mkdir -p $OUTPUT_DIR
cd $BERT_SRC && python3 parse_wiki_dump_json.py \
        --input_dir $INPUT_DIR \
        --output_dir $OUTPUT_DIR \
        --prefix_name wiki