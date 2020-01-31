'''
Created by trangvu on 29/01/20
'''
import pickle

file="/home/trang/workspace/maskedlm/new-bert/ner/data/twitter_train.pkl"
outfile="/home/trang/workspace/maskedlm/new-bert/ner/data/twitter_train.txt"
clean_sents = []

# with open(file,'r') as fout:
#     for line in fout:
#         token = line.split(' ')
#         if len(token) > 5 and len(token) < 100:
#             line = line[0:-1]
#             clean_sents.append(line)
data = pickle.load(open(file, 'rb'))
with open(outfile,'w') as fout:
    for sent in data:
        fout.write("{}\n".format(' '.join(sent[0])))
