'''
Created by trangvu on 29/01/20
'''

file="/home/vuth0001/workspace/2019-bert-selective-masking/data/twitter/trainingandtestdata/sentiment140.train.txt"
outfile="/home/vuth0001/workspace/2019-bert-selective-masking/data/twitter/trainingandtestdata/sentiment140.clean.train.txt"
clean_sents = []

with open(file,'r', encoding = "ISO-8859-1") as fout:
    for line in fout:
        token = line.split(' ')
        if len(token) > 5 and len(token) < 100:
            line = line[1:-2]
            clean_sents.append(line)

with open(outfile,'w') as fout:
    for sent in clean_sents:
        fout.write("{}\n".format(sent))
