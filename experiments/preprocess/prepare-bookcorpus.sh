#!/bin/bash
BOOKCORPUS_DIR="/home/xvuthith/da33/trang/masked-lm/wikibook/bookcorpus"
INDIR="/home/xvuthith/da33/trang/masked-lm/wikibook/bookcorpus/out_txts"
OUT_DIR="/home/xvuthith/da33/trang/masked-lm/wikibook/bookcorpus/sharded_bookcorpus"
module load python/3.6.2
echo " * DOWNLOAD book URL list"
cd $BOOKCORPUS_DIR && python3 -u download_list.py > url_list.jsonl

echo " * Download all book files"
cd $BOOKCORPUS_DIR && python3 download_files.py --list $BOOKCORPUS_DIR/url_list.jsonl --out $BOOKCORPUS_DIR/out_txts --trash-bad-count

echo " * Concatenate text with sentence-per-line format"
cd $BOOKCORPUS_DIR && python3 make_sentlines.py --input_dir $INDIR --output_dir $OUT_DIR