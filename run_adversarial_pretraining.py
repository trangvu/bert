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
import modeling
import optimization
import tensorflow as tf
import numpy as np
import itertools

import teacher
import tokenization
import gumbel

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "teacher_config_file", None,
    "The config json file corresponding to the teacher model. "
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

def mask_special_token(probability_matrix, labels, pad_id, cls_id, sep_id):
    shape = labels.get_shape().as_list()
    tf.logging.info("Shape {}".format(shape))
    # cls_mask = tf.constant(shape, cls_id)
    # sep_mask = tf.constant(shape, sep_id)
    # pad_mask = tf.constant(shape, pad_id)
    cls_indexes = tf.not_equal(labels, cls_id)  # [[2] [3] [5] [6]], shape=(4, 1)
    sep_indexes = tf.not_equal(labels, sep_id)
    pad_indexes = tf.not_equal(labels, pad_id)
    mask = tf.cast(tf.logical_and(tf.logical_and(cls_indexes, sep_indexes), pad_indexes), tf.float32)
    # a = sess.run(probability_matrix)

    probability_matrix = probability_matrix * mask


    # b = sess.run(probability_matrix)
        # [[[1 2 3], [2 2 3], [0 0 0], [0 0 0]]
        #  [[2 2 2], [2 2 2], [2 2 2], [0 0 0]]]))
    return probability_matrix

def model_fn_builder(bert_config, teacher_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, vocab_size, pad_id, cls_id, sep_id, mask_id, max_predictions_per_seq):
  """Returns `model_fn` closure for TPUEstimator."""

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
        print(input_mask)
        print(input_ids.shape)
        print(input_ids)

        return (input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights, tag_ids)

    ent_model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=raw_input_ids,
        input_mask=raw_input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    input_states = ent_model.get_sequence_output()

    teacher_model = teacher.TeacherModel(
        config=teacher_config,
        is_training=is_training,
        input_states=input_states,
        input_mask=raw_input_mask
    )

    def gather_z_indexes(sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
        sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=2)
        batch_size = sequence_shape[0]
        seq_length = sequence_shape[1]

        flat_offsets = tf.range(0, batch_size, dtype=tf.int32) * seq_length
        flat_positions = positions + flat_offsets
        flat_sequence_tensor = tf.reshape(sequence_tensor,
                                          [batch_size * seq_length])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        return output_tensor



    logZ, log_prob = calculate_partition_table(teacher_config, teacher_model.get_action_probs(), max_predictions_per_seq)
    # logZ = tf.Print(logZ, [logZ])
    # logZ_j = logZ[:, 0, :]  # b x k x 1
    # left = tf.constant(max_predictions_per_seq, shape=[2])
    # log_Z_total = gather_z_indexes(logZ_j, left)
    # logZ_j_next = logZ[:,  1, :]  # b x k x 1
    # log_Z_yes = gather_z_indexes(logZ_j_next, left - 1)
    # logp_j = log_prob[:,0]
    # log_q_yes = logp_j + log_Z_yes - log_Z_total
    # log_q_no = tf.log(1 - tf.exp(log_q_yes))
    # # # draw 2 Gumbel noise and compute action by argmax
    # logits = tf.stack([log_q_no, log_q_yes])
    # actions = gumbel.gumbel_softmax(logits)
    # mask = tf.cast(tf.argmax(actions,1),dtype=tf.float32)
    # actions = tf.reduce_max(actions,1)
    # # logq_j =tf.log(actions) * mask
    # # logZ_j = tf.Print(logZ_j, [logZ_j])
    samples, log_q = sampling_a_subset(logZ, log_prob, max_predictions_per_seq)


    input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights, tag_ids = tf.py_func(apply_masking,
                [samples, log_q, raw_masked_lm_ids, raw_masked_lm_positions, raw_tag_ids], (tf.int32,tf.int32,tf.int32,tf.int32,tf.float32,tf.int32))
    input_ids.set_shape(raw_input_ids.get_shape())
    input_mask.set_shape(raw_input_mask.get_shape())
    masked_lm_ids.set_shape(raw_masked_lm_ids.get_shape())
    masked_lm_positions.set_shape(raw_masked_lm_positions.get_shape())
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

    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)

    total_loss = masked_lm_loss + next_sentence_loss

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
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

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_example_loss,
                    next_sentence_log_probs, next_sentence_labels):
        """Computes the loss and accuracy of the model."""
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

        next_sentence_log_probs = tf.reshape(
            next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        next_sentence_predictions = tf.argmax(
            next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
        next_sentence_accuracy = tf.metrics.accuracy(
            labels=next_sentence_labels, predictions=next_sentence_predictions)
        next_sentence_mean_loss = tf.metrics.mean(
            values=next_sentence_example_loss)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

      eval_metrics = (metric_fn, [
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights, next_sentence_example_loss,
          next_sentence_log_probs, next_sentence_labels
      ])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def calculate_partition_table(teacher_config, output_weights, max_predictions_per_seq):
    shape = teacher.get_shape_list(output_weights, expected_rank=2)
    batch_size = shape[0]
    seq_len = shape[1]

    def accum_cond(j, logZ_j, logb, loga):
        return tf.greater(j, -1)

    def accum_body(j, logZ_j, logb, loga):
        logb_j = tf.squeeze(logb[:, j])
        next_logZ_j = tf.math.reduce_logsumexp(tf.stack([loga, logb_j]),0)
        logZ_j = logZ_j.write(j,next_logZ_j)
        return [tf.subtract(j,1), logZ_j, logb, next_logZ_j]

    def dp_loop_cond(k, logZ, lastZ, init_value):
        return tf.less(k,  max_predictions_per_seq + 1)

    def dp_body(k, logZ, lastZ, init_value):
        # logZ[j-1,k] = log_sum(logZ[j,k], logp(j-1) + logZ[j,k-1])
        # shift lastZ one step
        shifted_lastZ = tf.roll(lastZ, shift=1, axis=1)
        log_yes = logp + shifted_lastZ  # b x N
        logZ_j = tf.TensorArray(tf.float32, size=seq_len)
        # logZ_j = logZ_j.write(seq_len - 1, init_value)
        _, logZ_j, logb, loga = tf.while_loop(accum_cond, accum_body, [seq_len - k, logZ_j, log_yes, init_value])
        # logZ_j = logZ_j.write(0, tf.zeros([batch_size], dtype = tf.dtypes.float32))
        logZ_j = logZ_j.stack() # N x b
        logZ_j = tf.transpose(logZ_j, [1,0]) # b x N
        lastZ = logZ_j
        init_value = tf.zeros([batch_size], dtype = tf.dtypes.float32)
        logZ = logZ.write(k, logZ_j)
        return [tf.add(k,1), logZ, lastZ, init_value]


    with tf.variable_scope("teacher/dp"):
        # Calculate DP table: aims to calculate logZ[1,K]
        # index from 1 - we need to shift the input index by 1
        # logZ size batch_size x N+1 x K+1
        initZ = tf.TensorArray(tf.float32, size=max_predictions_per_seq+1)
        logZ_0 = tf.zeros([batch_size, seq_len], dtype = tf.dtypes.float32)
        initZ = initZ.write(tf.constant(0), logZ_0)
        logp = tf.log(output_weights)
        init_value=tf.squeeze(logp[:,-1])
        #init: logz_0 (k=0) = 0
        #      logz_1 (k=1) = log_p
        k = tf.constant(1)
        _, logZ, lastZ, last_value = tf.while_loop(dp_loop_cond, dp_body,
                        [k, initZ, logZ_0, init_value],
                        shape_invariants=[k.get_shape(), tf.TensorShape([]),
                                          tf.TensorShape([batch_size,None]), init_value.get_shape()])
        logZ = logZ.stack() # N x b x N
        logZ = tf.transpose(logZ, [1,2,0])
    return logZ, logp

def sampling_a_subset(logZ, logp, max_predictions_per_seq):
    shape = teacher.get_shape_list(logp, expected_rank=2)
    batch_size = shape[0]
    seq_len = shape[1]

    logZ = tf.Print(logZ, [logZ])

    def gather_z_indexes(sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
        flat_offsets = tf.range(0, batch_size, dtype=tf.int32) * max_predictions_per_seq
        flat_positions = positions + flat_offsets
        flat_sequence_tensor = tf.reshape(sequence_tensor,
                                          [batch_size * max_predictions_per_seq])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        return output_tensor
    def sampling_loop_cond(j, subset, count, left, log_q):
        # j == N or left == 0
        return tf.logical_or(tf.less(j,  seq_len), tf.equal(tf.reduce_sum(left),0))

    def sampling_body(j, subset, count, left, log_q):
        # calculate log_q_yes and log_q_no
        logp_j = logp[:,j]
        log_Z_total = gather_z_indexes(logZ[:, j, :], left) # b
        log_Z_yes = gather_z_indexes(logZ[:, j+1, :], left - 1) # b
        log_q_yes = logp_j + log_Z_yes - log_Z_total
        log_q_no = tf.log(1 - tf.exp(log_q_yes))
        # draw 2 Gumbel noise and compute action by argmax
        logits = tf.stack([log_q_no, log_q_yes])
        actions = gumbel.gumbel_softmax(logits)
        action_mask = tf.cast(tf.argmax(actions, 1), dtype=tf.int32)
        actions = tf.reduce_max(actions, 1)
        log_actions = tf.log(actions) * tf.cast(action_mask, dtype=tf.float32)
        # compute log_q_j and update count and subset
        count = count + action_mask
        left = left - action_mask
        log_q = log_q + log_actions
        subset = subset.write(j, action_mask)

        return [tf.add(j,1), subset, count, left, log_q]

    with tf.variable_scope("teacher/sampling"):
        # Batch sampling
        subset = tf.TensorArray(tf.int32, size=seq_len)
        count = tf.zeros([batch_size], dtype = tf.dtypes.int32)
        left = tf.constant(max_predictions_per_seq, shape=[batch_size])
        log_q = tf.zeros([batch_size], dtype=tf.dtypes.float32)

        _, subset, count, left, log_q = tf.while_loop(sampling_loop_cond, sampling_body, [tf.constant(0), subset, count, left, log_q])

        subset = subset.stack()  # K x b x N
        subset = tf.transpose(subset, [1, 2, 0])
        partition = logZ[:,0, max_predictions_per_seq]
        log_q = log_q - partition
    return subset, log_q


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
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
  teacher_config = teacher.TeacherConfig.from_json_file(FLAGS.teacher_config_file)
  vocab_size = bert_config.vocab_size

  tf.gfile.MakeDirs(FLAGS.output_dir)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  special_tokens = tokenizer.convert_tokens_to_ids(["[PAD]","[SEP]", "[CLS]", "[MASK]"])
  pad_id = special_tokens[0]
  sep_id = special_tokens[1]
  cls_id = special_tokens[2]
  mask_id = special_tokens[3]

  model_fn = model_fn_builder(
      bert_config=bert_config,
      teacher_config=teacher_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      vocab_size=vocab_size,
      pad_id = pad_id,
      sep_id = sep_id,
      cls_id = cls_id,
      max_predictions_per_seq=FLAGS.max_predictions_per_seq,
      mask_id = mask_id
  )

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
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
