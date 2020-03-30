### BLUE

sbatch --job-name=pm-adv submit-job-m3g-V100.sh pretrain-adv.sh 9e6
sbatch --job-name=pm-entropy submit-job-m3g-V100.sh pretrain-entropy.sh 9e6
sbatch --job-name=pm-pos submit-job-m3g-V100.sh pretrain-pos.sh 9e6
sbatch --job-name=pm-rand submit-job-m3g-V100.sh pretrain-rand.sh 9e6

sbatch --job-name=biosses-base submit-job-m3g-V100.sh eval_biosses.sh base /scratch/da33/trang/masked-lm/models/bert_base_uncased 2903
sbatch --job-name=biosses-rand submit-job-m3g-V100.sh eval_biosses.sh rand /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-random 2903
sbatch --job-name=biosses-pos submit-job-m3g-V100.sh eval_biosses.sh pos /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-pos 2903
sbatch --job-name=biosses-entropy submit-job-m3g-V100.sh eval_biosses.sh entropy /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-entropy 2903
sbatch --job-name=biosses-adv submit-job-m3g-V100.sh eval_biosses.sh adv /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-adv 2903
  "init_checkpoint": "/scratch/da33/trang/masked-lm/models/bert_base_uncased/bert_model.ckpt",


sbatch --job-name=chemprot-base submit-job-m3g-V100.sh eval_chemprot.sh base /scratch/da33/trang/masked-lm/models/bert_base_uncased 2903
sbatch --job-name=chemprot-rand submit-job-m3g-V100.sh eval_chemprot.sh rand /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-random 2903
sbatch --job-name=chemprot-pos submit-job-m3g-V100.sh eval_chemprot.sh pos /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-pos 2903
sbatch --job-name=chemprot-entropy submit-job-m3g-V100.sh eval_chemprot.sh entropy /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-entropy 2903
sbatch --job-name=chemprot-adv submit-job-m3g-V100.sh eval_chemprot.sh adv /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-adv 2903

sbatch --job-name=ddi-base submit-job-m3g-V100.sh eval_ddi.sh base /scratch/da33/trang/masked-lm/models/bert_base_uncased 2903
sbatch --job-name=ddi-rand submit-job-m3g-V100.sh eval_ddi.sh rand /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-random 2903
sbatch --job-name=ddi-pos submit-job-m3g-V100.sh eval_ddi.sh pos /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-pos 2903
sbatch --job-name=ddi-entropy submit-job-m3g-V100.sh eval_ddi.sh entropy /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-entropy 2903
sbatch --job-name=ddi-adv submit-job-m3g-V100.sh eval_ddi.sh adv /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-adv 2903

sbatch --job-name=hoc-base submit-job-m3g-V100.sh eval_hoc.sh base /scratch/da33/trang/masked-lm/models/bert_base_uncased 2903
sbatch --job-name=hoc-rand submit-job-m3g-V100.sh eval_hoc.sh rand /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-random 2903
sbatch --job-name=hoc-pos submit-job-m3g-V100.sh eval_hoc.sh pos /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-pos 2903
sbatch --job-name=hoc-entropy submit-job-m3g-V100.sh eval_hoc.sh entropy /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-entropy 2903
sbatch --job-name=hoc-adv submit-job-m3g-V100.sh eval_hoc.sh adv /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-32-adv 2903