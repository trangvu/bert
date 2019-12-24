
### Wikibook
#### Wikidump
```bash
sbatch --job-name=crawl-en --mem_per_cpu=70000 submit-job-monarch-cpu.sh crawl_wiki_dump.sh en
```

#### Bookcorpus
```bash
sbatch --job-name=crawl-en --mem_per_cpu=70000 submit-job--monarch-cpu.sh prepare-bookcorpus.sh
```

#### Merge wikidump and bookcorpus


### Pubmed