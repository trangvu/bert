# coding=utf-8
# @author: trangvu

"""Pre-trains dynamic and adversarial BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json

import tensorflow.compat.v1 as tf

import configure_pretraining
from model import modeling
from model import optimization
from pretrain import pretrain_data
from pretrain import pretrain_helpers
from util import training_utils
from util import utils


class PretrainingModel(object):
  """Transformer pre-training using the replaced-token-detection task."""

  def __init__(self, config: configure_pretraining.PretrainingConfig,
               features, is_training):
    # Set up model config
    self._config = config
    self._bert_config = training_utils.get_bert_config(config)
    if config.debug:
      self._bert_config.num_hidden_layers = 3
      self._bert_config.hidden_size = 144
      self._bert_config.intermediate_size = 144 * 4
      self._bert_config.num_attention_heads = 4

    # Mask the input
    masked_inputs = pretrain_helpers.mask(
        config, pretrain_data.features_to_inputs(features), config.mask_prob)


    embedding_size = (
        self._bert_config.hidden_size if config.embedding_size is None else
        config.embedding_size)

    # BERT model
    model = self._build_transformer(
        masked_inputs, is_training,
        embedding_size = embedding_size)
    mlm_output = self._get_masked_lm_output(masked_inputs, model)
    self.total_loss = mlm_output.loss

    # Evaluation
    eval_fn_inputs = {
        "input_ids": masked_inputs.input_ids,
        "masked_lm_preds": mlm_output.preds,
        "mlm_loss": mlm_output.per_example_loss,
        "masked_lm_ids": masked_inputs.masked_lm_ids,
        "masked_lm_weights": masked_inputs.masked_lm_weights,
        "input_mask": masked_inputs.input_mask
    }
    eval_fn_keys = eval_fn_inputs.keys()
    eval_fn_values = [eval_fn_inputs[k] for k in eval_fn_keys]

    """Computes the loss and accuracy of the model."""
    d = {k: arg for k, arg in zip(eval_fn_keys, eval_fn_values)}
    metrics = dict()
    metrics["masked_lm_accuracy"] = tf.metrics.accuracy(
        labels=tf.reshape(d["masked_lm_ids"], [-1]),
        predictions=tf.reshape(d["masked_lm_preds"], [-1]),
        weights=tf.reshape(d["masked_lm_weights"], [-1]))
    metrics["masked_lm_loss"] = tf.metrics.mean(
        values=tf.reshape(d["mlm_loss"], [-1]),
        weights=tf.reshape(d["masked_lm_weights"], [-1]))
    self.eval_metrics = metrics

  def _get_masked_lm_output(self, inputs: pretrain_data.Inputs, model):
    """Masked language modeling softmax layer."""
    masked_lm_weights = inputs.masked_lm_weights
    with tf.variable_scope("generator_predictions"):
      if self._config.uniform_generator:
        logits = tf.zeros(self._bert_config.vocab_size)
        logits_tiled = tf.zeros(
            modeling.get_shape_list(inputs.masked_lm_ids) +
            [self._bert_config.vocab_size])
        logits_tiled += tf.reshape(logits, [1, 1, self._bert_config.vocab_size])
        logits = logits_tiled
      else:
        relevant_hidden = pretrain_helpers.gather_positions(
            model.get_sequence_output(), inputs.masked_lm_positions)
        hidden = tf.layers.dense(
            relevant_hidden,
            units=modeling.get_shape_list(model.get_embedding_table())[-1],
            activation=modeling.get_activation(self._bert_config.hidden_act),
            kernel_initializer=modeling.create_initializer(
                self._bert_config.initializer_range))
        hidden = modeling.layer_norm(hidden)
        output_bias = tf.get_variable(
            "output_bias",
            shape=[self._bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(hidden, model.get_embedding_table(),
                           transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

      oh_labels = tf.one_hot(
          inputs.masked_lm_ids, depth=self._bert_config.vocab_size,
          dtype=tf.float32)

      probs = tf.nn.softmax(logits)
      log_probs = tf.nn.log_softmax(logits)
      label_log_probs = -tf.reduce_sum(log_probs * oh_labels, axis=-1)

      numerator = tf.reduce_sum(inputs.masked_lm_weights * label_log_probs)
      denominator = tf.reduce_sum(masked_lm_weights) + 1e-6
      loss = numerator / denominator
      preds = tf.argmax(log_probs, axis=-1, output_type=tf.int32)

      MLMOutput = collections.namedtuple(
          "MLMOutput", ["logits", "probs", "loss", "per_example_loss", "preds"])
      return MLMOutput(
          logits=logits, probs=probs, per_example_loss=label_log_probs,
          loss=loss, preds=preds)

  def _build_transformer(self, inputs: pretrain_data.Inputs, is_training,
                         bert_config=None, name="electra", reuse=False, **kwargs):
    """Build a transformer encoder network."""
    if bert_config is None:
      bert_config = self._bert_config
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
      return modeling.BertModel(
          bert_config=bert_config,
          is_training=is_training,
          input_ids=inputs.input_ids,
          input_mask=inputs.input_mask,
          token_type_ids=inputs.segment_ids,
          use_one_hot_embeddings=self._config.use_tpu,
          scope=name)


def model_fn_builder(config: configure_pretraining.PretrainingConfig):
  """Build the model for training."""

  def model_fn(features, labels, mode, params):
    """Build the model for training."""
    model = PretrainingModel(config, features,
                             mode == tf.estimator.ModeKeys.TRAIN)
    utils.log("Model is built!")

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    if config.init_checkpoint:
      (assignment_map, initialized_variable_names
       ) = modeling.get_assignment_map_from_checkpoint(tvars, config.init_checkpoint)
      tf.train.init_from_checkpoint(config.init_checkpoint, assignment_map)

    utils.log("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      utils.log("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          model.total_loss, config.learning_rate, config.num_train_steps,
          weight_decay_rate=config.weight_decay_rate,
          use_tpu=config.use_tpu,
          warmup_steps=config.num_warmup_steps,
          lr_decay_power=config.lr_decay_power
      )
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=model.total_loss,
          train_op=train_op,
          training_hooks=[training_utils.ETAHook(dict(loss=model.total_loss),
              config.num_train_steps, config.iterations_per_loop,
              config.use_tpu)]
      )
    elif mode == tf.estimator.ModeKeys.EVAL:
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=model.total_loss,
          eval_metric_ops=model.eval_metrics,
          evaluation_hooks=[training_utils.ETAHook(dict(loss=model.total_loss),
              config.num_eval_steps, config.iterations_per_loop,
              config.use_tpu, is_training=False)])
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported")
    return output_spec

  return model_fn


def train_or_eval(config: configure_pretraining.PretrainingConfig):
  """Run pre-training or evaluate the pre-trained model."""
  if config.do_train == config.do_eval:
    raise ValueError("Exactly one of `do_train` or `do_eval` must be True.")
  if config.debug:
    utils.rmkdir(config.model_dir)
  utils.heading("Config:")
  utils.log_config(config)

  num_gpus = utils.get_available_gpus()
  utils.log("Found {} gpus".format(len(num_gpus)))

  train_distribution_strategy = tf.distribute.MirroredStrategy(devices=None)
  eval_distribution_strategy = tf.distribute.MirroredStrategy(devices=None)

  run_config = tf.estimator.RunConfig(
      model_dir=config.model_dir,
      save_checkpoints_steps=config.save_checkpoints_steps,
      train_distribute=train_distribution_strategy,
      eval_distribute=eval_distribution_strategy,
      # save_checkpoints_secs=3600,
      # tf_random_seed=FLAGS.seed,
      session_config=tf.ConfigProto(log_device_placement=True),
      # keep_checkpoint_max=0,
      log_step_count_steps=100
  )
  model_fn = model_fn_builder(config=config)
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params={'train_batch_size': config.train_batch_size,
              'eval_batch_size': config.eval_batch_size})

  if config.do_train:
    utils.heading("Running training")
    estimator.train(input_fn=pretrain_data.get_input_fn(config, True),
                    max_steps=config.num_train_steps)
  if config.do_eval:
    utils.heading("Running evaluation")
    result = estimator.evaluate(
        input_fn=pretrain_data.get_input_fn(config, False),
        steps=config.num_eval_steps)
    for key in sorted(result.keys()):
      utils.log("  {:} = {:}".format(key, str(result[key])))
    return result


def train_one_step(config: configure_pretraining.PretrainingConfig):
  """Builds an ELECTRA model an trains it for one step; useful for debugging."""
  train_input_fn = pretrain_data.get_input_fn(config, True)
  features = tf.data.make_one_shot_iterator(train_input_fn(dict(
      batch_size=config.train_batch_size))).get_next()
  model = PretrainingModel(config, features, True)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    utils.log(sess.run(model.total_loss))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data-dir", required=True,
                      help="Location of models (model weights, etc).")
  parser.add_argument("--model-name", required=True,
                      help="The name of the model being fine-tuned.")
  parser.add_argument("--hparams", default="{}",
                      help="JSON dict of model hyperparameters.")
  args = parser.parse_args()
  if args.hparams.endswith(".json"):
    hparams = utils.load_json(args.hparams)
  else:
    hparams = json.loads(args.hparams)
  tf.logging.set_verbosity(tf.logging.ERROR)
  train_or_eval(configure_pretraining.PretrainingConfig(
      args.model_name, args.data_dir, **hparams))


if __name__ == "__main__":
  main()
