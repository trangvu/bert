
### Wikibook
#### Wikidump
```bash
sbatch --job-name=crawl-en --mem_per_cpu=70000 submit-job-monarch-cpu.sh crawl_wiki_dump.sh en
```
wget http://dumps.wikimedia.org/frwiki/latest/frwiki-latest-pages-articles.xml.bz2
wget http://dumps.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles.xml.bz2

- lowercase
- tokenize by moses
- joints to a single file

- Learn vocabulary using huggingface tokenize
   python learn_wordpiece.py --files 
    

#### Bookcorpus
```bash
sbatch --job-name=crawl-en --mem_per_cpu=70000 submit-job--monarch-cpu.sh prepare-bookcorpus.sh
```

#### Merge wikidump and bookcorpus


### Pubmed