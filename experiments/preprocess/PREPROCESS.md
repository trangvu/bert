
### Wikibook
#### Wikidump
```bash
bunzip2 enwiki-latest-pages-articles.xml.bz2

sbatch --job-name=crawl-en --mem_per_cpu=70000 submit-job-comp.sh crawl_wiki_dump.sh en
```

#### Bookcorpus
```bash
sbatch --job-name=crawl-en --mem_per_cpu=70000 prepare-bookcorpus.sh
```

#### Merge wikidump and bookcorpus


### Pubmed