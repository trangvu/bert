### BLUE

sbatch --job-name=biosses-base submit-job-m3g-V100.sh eval_biosses.sh base /scratch/da33/trang/masked-lm/models/bert_base_uncased 2903
sbatch --job-name=biosses-rand submit-job-m3g-V100.sh eval_biosses.sh rand /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-random 2903
sbatch --job-name=biosses-pos submit-job-m3g-V100.sh eval_biosses.sh pos /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-pos 2903
sbatch --job-name=biosses-entropy submit-job-m3g-V100.sh eval_biosses.sh entropy /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-entropy 2903
sbatch --job-name=biosses-adv submit-job-m3g-V100.sh eval_biosses.sh adv /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-adv 2903
  "init_checkpoint": "/scratch/da33/trang/masked-lm/models/bert_base_uncased/bert_model.ckpt",