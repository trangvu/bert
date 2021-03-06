# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from model import modeling, optimization, tokenization
import tensorflow as tf
import numpy as np
from spacy.symbols import IDS

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "mask_strategy", "random", "Mask strategy. Should be in one of the following values: random, pos, entropy"
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")


## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "seed", 128,
    "Seed")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, mask_strategy, vocab_size, pad_id, cls_id, sep_id, mask_id, max_predictions_per_seq):
  """Returns `model_fn` closure for TPUEstimator."""

  if mask_strategy == 'pos':
    PREFER_TAGS = ['ADJ', 'VERB', 'NOUN', 'PRON', 'ADV']
    PREFER_TAG_IDS = [IDS[tag] for tag in PREFER_TAGS]

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    raw_input_ids = features["input_ids"]
    raw_input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    raw_masked_lm_positions = features["masked_lm_positions"]
    raw_masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]
    raw_tag_ids = features["tag_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)


    def apply_masking(input_ids, input_mask, masked_lm_ids, masked_lm_positions, tag_ids):

        shape = input_ids.shape
        if mask_strategy == 'random':
            probs = np.full(shape, 1)
            probs = np.where(np.logical_not(np.logical_or(np.logical_or(np.equal(input_ids, sep_id), np.equal(input_ids, cls_id)), np.equal(input_ids, pad_id))), probs, 0)
            probs = probs/ probs.sum(axis=1,keepdims=1)

        elif mask_strategy == 'pos':
            probs = np.full(shape, 1)
            probs = np.where(np.isin(tag_ids, PREFER_TAG_IDS), probs, 5)
            probs = np.where(np.logical_not(
                np.logical_or(np.logical_or(np.equal(input_ids, sep_id), np.equal(input_ids, cls_id)),
                              np.equal(input_ids, pad_id))), probs, 0)
            probs = probs / probs.sum(axis=1, keepdims=1)


        seq_len = np.shape(probs)[1]
        masked_lm_ids = []
        masked_lm_positions = []
        masked_lm_weights = []
        for input_id, p in zip(input_ids, probs):
            k = np.count_nonzero(p)
            if max_predictions_per_seq < k:
                mask_ids = np.random.choice(seq_len, max_predictions_per_seq, replace=False, p=p)
            else:
                mask_ids = np.random.choice(seq_len, k, replace=False, p=p)
            label_ids = input_id[mask_ids]
            masked_lm_weight = [1.0] * len(mask_ids)
            input_id[mask_ids] = mask_id

            # 10% is KEEP and 10% is replaced with random words
            if max_predictions_per_seq < k:
                num_keep = int(0.1 * max_predictions_per_seq)
                special_indices = np.random.choice(max_predictions_per_seq, num_keep * 2, replace=False)
            else:
                num_keep = int(0.1 * k)
                special_indices = np.random.choice(k, num_keep * 2, replace=False)

            keep_indices = special_indices[:num_keep]
            random_indices = special_indices[num_keep:]
            input_id[mask_ids[keep_indices]] = label_ids[keep_indices]
            random_ids = np.random.choice(vocab_size - 5, num_keep) + 5
            input_id[mask_ids[random_indices]] = random_ids

            if len(mask_ids) < max_predictions_per_seq:
                print("WARNING less than k")
                #padding if we have less than k
                num_pad = max_predictions_per_seq - len(mask_ids)
                mask_ids = np.pad(mask_ids, (0, num_pad), 'constant', constant_values=(0,0))
                label_ids = np.pad(label_ids, (0, num_pad), 'constant', constant_values=(0,0))
                masked_lm_weight = np.pad(masked_lm_weight, (0, num_pad), constant_values=(0.0,0.0))
            masked_lm_ids.append(label_ids)
            masked_lm_positions.append(mask_ids)
            masked_lm_weights.append(masked_lm_weight)

        masked_lm_ids = np.asarray(masked_lm_ids)
        masked_lm_positions = np.asarray(masked_lm_positions, dtype='int32')
        masked_lm_weights = np.asarray(masked_lm_weights, dtype='float32')

        return (input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights, tag_ids)


    def apply_entropy_masking(input_ids, input_mask, masked_lm_ids, masked_lm_positions, entropies):

        shape = input_ids.shape

        seq_len = shape[1]
        masked_lm_ids = []
        masked_lm_positions = []
        masked_lm_weights = []
        #TODO: exclude CLS and SEP
        for input_id, entropy in zip(input_ids, entropies):
            mask_ids = np.argpartition(entropy,-max_predictions_per_seq)[-max_predictions_per_seq:]
            label_ids = input_id[mask_ids]
            masked_lm_weight = [1.0] * len(mask_ids)
            input_id[mask_ids] = mask_id

            # 10% is KEEP and 10% is replaced with random words
            num_keep = int(0.1 * max_predictions_per_seq)
            special_indices = np.random.choice(max_predictions_per_seq, num_keep * 2, replace=False)
            keep_indices = special_indices[:num_keep]
            random_indices = special_indices[num_keep:]
            input_id[mask_ids[keep_indices]] = label_ids[keep_indices]
            random_ids = np.random.choice(vocab_size - 5, num_keep) + 5
            input_id[mask_ids[random_indices]] = random_ids
            masked_lm_ids.append(label_ids)
            masked_lm_positions.append(mask_ids)
            masked_lm_weights.append(masked_lm_weight)

        masked_lm_ids = np.asarray(masked_lm_ids)
        masked_lm_positions = np.asarray(masked_lm_positions, dtype='int32')
        masked_lm_weights = np.asarray(masked_lm_weights, dtype='float32')


        return (input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights)

    if mask_strategy == 'entropy':
        ent_model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=raw_input_ids,
            input_mask=raw_input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)
        masked_lm_log_probs = get_entropy_output(
        bert_config, ent_model.get_sequence_output(), ent_model.get_embedding_table())
        input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights = tf.py_func(
            apply_entropy_masking,
            [raw_input_ids, raw_input_mask, raw_masked_lm_ids, raw_masked_lm_positions, masked_lm_log_probs],
            (tf.int32, tf.int32, tf.int32, tf.int32, tf.float32))
        input_ids.set_shape(raw_input_ids.get_shape())
        input_mask.set_shape(raw_input_mask.get_shape())
        masked_lm_ids.set_shape(raw_masked_lm_ids.get_shape())
        masked_lm_positions.set_shape(raw_masked_lm_positions.get_shape())
        masked_lm_weights.set_shape(raw_masked_lm_positions.get_shape())
    else:
        input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights, tag_ids = tf.py_func(apply_masking,
                    [raw_input_ids, raw_input_mask, raw_masked_lm_ids, raw_masked_lm_positions, raw_tag_ids], (tf.int32,tf.int32,tf.int32,tf.int32, tf.float32,tf.int32))
        input_ids.set_shape(raw_input_ids.get_shape())
        input_mask.set_shape(raw_input_mask.get_shape())
        masked_lm_ids.set_shape(raw_masked_lm_ids.get_shape())
        masked_lm_positions.set_shape(raw_masked_lm_positions.get_shape())
        masked_lm_weights.set_shape(raw_masked_lm_positions.get_shape())

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    # (next_sentence_loss, next_sentence_example_loss,
    #  next_sentence_log_probs) = get_next_sentence_output(
    #      bert_config, model.get_pooled_output(), next_sentence_labels)

    total_loss = masked_lm_loss #+ next_sentence_loss

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=10)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          training_hooks = [logging_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)
        eval_metric_op = {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss
            }
        output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=eval_metric_op)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn

def get_masked_lm_logits(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.embedding_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)

def get_entropy_output(bert_config, input_tensor, output_weights):

    with tf.variable_scope("cls/predictions",reuse=tf.AUTO_REUSE):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform",reuse=tf.AUTO_REUSE):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.embedding_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probs = tf.nn.softmax(logits, axis=-1)
        entropy = -tf.reduce_sum(probs * tf.log(probs), [2])

    return entropy


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions", reuse=tf.AUTO_REUSE):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform", reuse=tf.AUTO_REUSE):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.embedding_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     mask_strategy, masked_lm_prob, pad_id, cls_id, sep_id,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
        "tag_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    params['mask_strategy'] = mask_strategy
    params['masked_lm_prob'] = masked_lm_prob
    params['pad_id'] = pad_id
    params['cls_id'] = cls_id
    params['sep_id'] = sep_id
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features, params),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d

  return input_fn


def _decode_record(record, name_to_features, params):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  vocab_size = bert_config.vocab_size

  tf.gfile.MakeDirs(FLAGS.output_dir)
  tf.logging.info("Num GPUs Available: {}".format(len(tf.config.experimental.list_physical_devices('GPU'))))
  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      # save_checkpoints_secs=3600,
      tf_random_seed=FLAGS.seed,
      # session_config=tf.ConfigProto(log_device_placement=True),
      log_step_count_steps = 100,
      keep_checkpoint_max=0
  )

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  special_tokens = tokenizer.convert_tokens_to_ids(["[PAD]","[SEP]", "[CLS]", "[MASK]"])
  pad_id = special_tokens[0]
  sep_id = special_tokens[1]
  cls_id = special_tokens[2]
  mask_id = special_tokens[3]

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      mask_strategy=FLAGS.mask_strategy,
      vocab_size=vocab_size,
      pad_id = pad_id,
      sep_id = sep_id,
      cls_id = cls_id,
      max_predictions_per_seq=FLAGS.max_predictions_per_seq,
      mask_id = mask_id
  )
  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={'batch_size': FLAGS.train_batch_size}
    )
    train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True,
      mask_strategy=FLAGS.mask_strategy,
      masked_lm_prob=FLAGS.masked_lm_prob,
      pad_id = pad_id,
      sep_id = sep_id,
      cls_id = cls_id)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={'batch_size': FLAGS.eval_batch_size}
    )
    eval_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False,
      mask_strategy=FLAGS.mask_strategy,
      masked_lm_prob=FLAGS.masked_lm_prob,
      pad_id = pad_id,
      sep_id = sep_id,
      cls_id = cls_id)

    result = estimator.evaluate(
        input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
