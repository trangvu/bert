'''
Created by trangvu on 28/01/20
'''
import collections
import os
import pickle
import random

import tensorflow as tf
from model import tokenization

'''
Combine ner dataset for training:
    + For domain-tuning:  sep_twitter_train.pkl twitter_train + sep_twitter_train.pkl twitter_test + conll_train.pkl conll_train
'''

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None,
                    "data dir")
flags.DEFINE_string("output_file", "wnut_train.tfrecord",
                    "output file")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

mlm_cvg_hack = 10

class DataProcessor(object):
    """Processor for the MRPC data set (GLUE version)."""

    def get_conll_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_pkl(os.path.join(data_dir, "conll_train.pkl")), "conll_train")

    def get_conll_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_pkl(os.path.join(data_dir, "conll_test.pkl")), "conll_dev")

    def get_sep_twitter_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_pkl(os.path.join(data_dir, "sep_twitter_train.pkl")), "twitter_train", oversampling_ratio=10)

    def get_sep_twitter_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_pkl(os.path.join(data_dir, "sep_twitter_train.pkl.pkl")), "twitter_test")

    def get_twitter_general_examples(self, data_dir):
        """See base class."""
        return self._create_examples_without_replacement(
            self._read_pkl(os.path.join(data_dir, "twitter_general.pkl")), "twitter_general")

    def get_labels(self, data_dir):
        """See base class."""
        return ['B', 'I', 'O']

    def _create_examples(self, data, set_type, oversampling_ratio=1):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, elem) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text = ' '.join(elem[0])
            # label = elem[1] # not necessary for cloze-style LM fine-tuning
            for j in range(oversampling_ratio):
                examples.append(InputExample(guid=guid, text=text))
        return examples

    def _create_examples_without_replacement(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, elem) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text = ' '.join(elem[0])
            # label = elem[1] # not necessary for cloze-style LM fine-tuning
            examples.append(InputExample(guid=guid, text=text))
        return examples

    def _read_pkl(self, input_file):
        """Reads a tab separated value file."""
        data = pickle.load(open(input_file, 'rb'))
        return data

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text # list of tokens

def convert_example_to_features(example, max_seq_length, max_predictions_per_seq, tokenizer):
    tokens = tokenizer.tokenize(example.text)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:max_seq_length-2]
    tokens.insert(0, "[CLS]")
    tokens.append("[SEP]")
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    tags_ids = [-1] * len(input_ids)
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
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    processor = DataProcessor()
    twitter_examples = processor.get_sep_twitter_train_examples(FLAGS.data_dir)
    # conll_examples = processor.get_conll_train_examples(FLAGS.data_dir)
    examples = twitter_examples #+ conll_examples
    random.shuffle(examples)
    total_written = 0
    writer = tf.python_io.TFRecordWriter(FLAGS.output_file)
    for (inst_index, example) in enumerate(examples):
        total_written += 1
        features = convert_example_to_features(example, 128, 20, tokenizer)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        if inst_index < 20:
          tf.logging.info("*** Example ***")
          tf.logging.info("tokens: %s" % example.text)

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
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
