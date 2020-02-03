'''
Created by trangvu on 28/01/20
'''
import collections
import os
import pickle
import random

import tensorflow as tf
import tokenization
from spacy.symbols import IDS

import spacy

'''
Combine ner dataset for training:
    + For domain-tuning:  sep_twitter_train.pkl twitter_train + sep_twitter_train.pkl twitter_test + conll_train.pkl conll_train
'''

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "data dir")
flags.DEFINE_string("output_file", "output.tfrecord",
                    "output file")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")


def infer_tags(tokens, pos_tags, spacy_tokens):
    '''
    Greedy assign spacy inferred POS tags to wordpiece token
    :param tokens:
    :param pos_tags:
    :param spacy_tokens:
    :return:
    '''
    tags = []
    num_token = len(tokens)
    num_tags = len(pos_tags)
    i = 0
    j = 0
    tok_cnt = 0
    spacy_tok_cnt = 0
    while i < num_token:
        tok = tokens[i].lower()
        if '##' in tok:
            tok = tok[2:]
        spacy_tok = spacy_tokens[j].lower()
        tags.append(IDS[pos_tags[j]])
        tok_cnt += len(tok)
        new_spacy_tok_cnt = spacy_tok_cnt + len(spacy_tok)
        if tok_cnt >= new_spacy_tok_cnt:
            j += 1
            if j >= num_tags:
                j = num_tags - 1
            spacy_tok_cnt = new_spacy_tok_cnt
        i += 1

    return tags


def convert_text_to_features(nlp, text, max_seq_length, max_predictions_per_seq, tokenizer):
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    txt = [token.text for token in doc]

    tokens = tokenizer.tokenize(text)
    tags_ids = infer_tags(tokens, pos, txt)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:max_seq_length-2]
        tags_ids = tags_ids[0:max_seq_length-2]
    tokens.insert(0, "[CLS]")
    tags_ids.insert(0, -1)
    tokens.append("[SEP]")
    tags_ids.append(-1)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)
      tags_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    masked_lm_positions = []
    masked_lm_ids = []
    masked_lm_weights = []
    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)
    next_sentence_label = 0
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])
    features["tag_ids"] = create_int_feature(tags_ids)
    return features

def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature

def main(_):
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=True)
    total_written = 0
    writer = tf.python_io.TFRecordWriter(FLAGS.output_file)
    inst_index = -1

    nlp = spacy.load("en_core_web_sm")
    with open(FLAGS.input_file, 'r') as fin:
        for line in fin:
            total_written += 1
            features = convert_text_to_features(nlp, line[:-1], 128, 20, tokenizer)
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
            inst_index += 1
            if inst_index < 20:
              tf.logging.info("*** Example ***")
              tf.logging.info("tokens: %s" % line)

              for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                  values = feature.int64_list.value
                elif feature.float_list.value:
                  values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))
    writer.close()
    tf.logging.info("Wrote %d total instances", total_written)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
