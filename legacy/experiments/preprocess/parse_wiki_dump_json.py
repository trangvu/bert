'''
Created by trangvu on 1/10/19
'''
import argparse
import logging
import os
import re
import json
import ast
from nltk.tokenize import sent_tokenize

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--prefix_name",
                        type=str,
                        default="wiki",
                        help="Prefix name")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    args = parser.parse_args()

    rx = r'{}_([0-9]+)'.format(re.escape(args.prefix_name))
    partitions = [re.findall(rx, f)[0] for f in os.listdir(args.input_dir) if re.match(rx, f)]
    for idx in partitions:
        discard = 0
        select = 0
        with open("{}/{}_{}".format(args.output_dir, args.prefix_name, idx), 'w') as fout:
            with open("{}/{}_{}".format(args.input_dir, args.prefix_name, idx), 'r') as fin:
                for line in fin:
                    doc = json.loads(line.strip())
                    txt = doc['text']
                    # txt = ast.literal_eval(doc['text'])
                    #cleanup text: remove links
                    uri=r'<?\w+:\/?\/?[^\s]+>?'
                    clean_txt = re.sub(uri, ' ', txt)
                    tempstyles=r'<\/?templatestyles[^>]*>'
                    ref = r'<\/?ref[^>]*>'
                    clean_txt = re.sub(tempstyles, ' ', clean_txt)
                    clean_txt = re.sub(ref, ' ', clean_txt)
                    docs = clean_txt.split('\n')
                    for doc in docs:
                        sentences = sent_tokenize(doc)
                        if len(sentences) < 3:
                            discard += 1
                        else:
                            select += 1
                            for sent in sentences:
                                sent = sent.strip()
                                if len(sent) > 0:
                                    fout.write("{}\n".format(sent))
                            fout.write("\n")

if __name__ == "__main__":
    main()
