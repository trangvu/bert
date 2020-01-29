### Prepare data
```bash
python covert-to-tfrecord.py --data_dir=./data --output_filr=./data/domain_tuning.tfrecord --do_lower_case --vocab_file=../config/en_uncase_vocab.txt
```

#### Cleaning Sentiment140
```bash
cat training.1600000.processed.noemoticon.csv | cut -d ',' -f6- > sentiment140.train.txt
python clean-data.py
iconv -t utf-8 sentiment140.clean.train.txt > sentiment140.clean.utf8.train.txt

cat sentiment140.clean.utf8.train.txt | $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en | $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l en > sentiment140.clean.utf8.train.tok

shuf -n 100000 sentiment140.clean.utf8.train.tok > sentiment140.100k.tok
shuf -n 500000 sentiment140.clean.utf8.train.tok > sentiment140.500k.tok
shuf -n 1000000 sentiment140.clean.utf8.train.tok > sentiment140.1M.tok

```