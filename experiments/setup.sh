#!/usr/bin/env bash
#### python
module load cuda/9.0
module load cudnn/7.3.0-cuda9
#module load cuda/10.1
#module load cudnn/7.6.5-cuda10.1
module load python/3.6.2
virtualenv --system-site-packages -p python3 .
pip install tensorflow-gpu==1.11.0

#### Conda
module load anaconda/2019.03-Python3.7-gcc5
module load cuda/9.0
module load cudnn/7.3.0-cuda9
conda install pip
conda create -n tfbert python=3.6.8
conda activate tfbert
python -m pip install tensorflow-gpu==1.13.1


#update bashrc file
export BERT_BASE_DIR=/home/vuth0001/workspace/2019-bert-selective-masking/adversarial/bert
/scratch/da33/trang/masked-lm/models/bert_base_uncased
export GLUE_DIR=/home/vuth0001/workspace/2019-bert-selective-masking/jiant/data
/project/da33/data_nlp/natural_language_understanding

python3 -m pip install --ignore-installed --upgrade tensorflow-gpu==1.13.0

cut -d'	' -f1-2

#Testing result

smux new-session --partition=m3g --time=2-00:00:00 --gres=gpu:1