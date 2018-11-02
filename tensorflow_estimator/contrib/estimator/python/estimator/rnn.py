# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Recurrent Neural Network estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow_estimator.contrib.estimator.python.estimator import extenders
from tensorflow.contrib.feature_column.python.feature_column import sequence_feature_column as seq_fc
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.summary import summary
from tensorflow.python.training import optimizer as optimizer_lib
from tensorflow.python.training import training_util


# The defaults are historical artifacts of the initial implementation, but seem
# reasonable choices.
_DEFAULT_LEARNING_RATE = 0.05
_DEFAULT_CLIP_NORM = 5.0

_SIMPLE_RNN_KEY = 'simple_rnn'
_CELL_TYPES = {_SIMPLE_RNN_KEY: keras_layers.SimpleRNNCell,
               'lstm': keras_layers.LSTMCell,
               'gru': keras_layers.GRUCell}

# Indicates no value was provided by the user to a kwarg.
USE_DEFAULT = object()


def _make_rnn_cell(num_units, cell_type=_SIMPLE_RNN_KEY):
  """Convenience function to create a RNN cell for canned RNN Estimators.

  Args:
    num_units: Iterable of integer number of hidden units per RNN layer.
    cell_type: A string specifying the cell type. Supported strings are:
      `'simple_rnn'`, `'lstm'`, and `'gru'`.

  Returns:
    A Keras RNN cell.

  Raises:
    ValueError: If cell_type is not supported.
  """
  if cell_type not in _CELL_TYPES:
    raise ValueError('Supported cell types are {}; got {}'.format(
        list(_CELL_TYPES.keys()), cell_type))
  cells = [_CELL_TYPES[cell_type](units=n) for n in num_units]
  if len(cells) == 1:
    return cells[0]
  return keras_layers.StackedRNNCells(cells, reverse_state_order=True)


def _concatenate_context_input(sequence_input, context_input):
  """Replicates `context_input` across all timesteps of `sequence_input`.

  Expands dimension 1 of `context_input` then tiles it `sequence_length` times.
  This value is appended to `sequence_input` on dimension 2 and the result is
  returned.

  Args:
    sequence_input: A `Tensor` of dtype `float32` and shape `[batch_size,
      padded_length, d0]`.
    context_input: A `Tensor` of dtype `float32` and shape `[batch_size, d1]`.

  Returns:
    A `Tensor` of dtype `float32` and shape `[batch_size, padded_length,
    d0 + d1]`.

  Raises:
    ValueError: If `sequence_input` does not have rank 3 or `context_input` does
      not have rank 2.
  """
  seq_rank_check = check_ops.assert_rank(
      sequence_input,
      3,
      message='sequence_input must have rank 3',
      data=[array_ops.shape(sequence_input)])
  seq_type_check = check_ops.assert_type(
      sequence_input,
      dtypes.float32,
      message='sequence_input must have dtype float32; got {}.'.format(
          sequence_input.dtype))
  ctx_rank_check = check_ops.assert_rank(
      context_input,
      2,
      message='context_input must have rank 2',
      data=[array_ops.shape(context_input)])
  ctx_type_check = check_ops.assert_type(
      context_input,
      dtypes.float32,
      message='context_input must have dtype float32; got {}.'.format(
          context_input.dtype))
  with ops.control_dependencies(
      [seq_rank_check, seq_type_check, ctx_rank_check, ctx_type_check]):
    padded_length = array_ops.shape(sequence_input)[1]
    tiled_context_input = array_ops.tile(
        array_ops.expand_dims(context_input, 1),
        array_ops.concat([[1], [padded_length], [1]], 0))
  return array_ops.concat([sequence_input, tiled_context_input], 2)


def _rnn_logit_fn_builder(output_units, rnn_cell, sequence_feature_columns,
                          context_feature_columns, input_layer_partitioner,
                          return_sequences=False):
  """Function builder for a rnn logit_fn.

  Args:
    output_units: An int indicating the dimension of the logit layer.
    rnn_cell: A Keras RNN cell object.
    sequence_feature_columns: An iterable containing the `FeatureColumn`s
      that represent sequential input.
    context_feature_columns: An iterable containing the `FeatureColumn`s
      that represent contextual input.
    input_layer_partitioner: Partitioner for input layer.
    return_sequences: A boolean indicating whether to return the last output
      in the output sequence, or the full sequence.

  Returns:
    A logit_fn (see below).

  Raises:
    ValueError: If output_units is not an int.
  """
  if not isinstance(output_units, int):
    raise ValueError('output_units must be an int.  Given type: {}'.format(
        type(output_units)))

  def rnn_logit_fn(features, mode):
    """Recurrent Neural Network logit_fn.

    Args:
      features: This is the first item returned from the `input_fn`
                passed to `train`, `evaluate`, and `predict`. This should be a
                single `Tensor` or `dict` of same.
      mode: Optional. Specifies if this training, evaluation or prediction. See
            `ModeKeys`.

    Returns:
      A `Tensor` representing the logits.
    """
    with variable_scope.variable_scope(
        'sequence_input_layer',
        values=tuple(six.itervalues(features)),
        partitioner=input_layer_partitioner):
      sequence_input, sequence_length = seq_fc.sequence_input_layer(
          features=features, feature_columns=sequence_feature_columns)
      summary.histogram('sequence_length', sequence_length)

      if context_feature_columns:
        context_input = feature_column_lib.input_layer(
            features=features,
            feature_columns=context_feature_columns)
        sequence_input = _concatenate_context_input(sequence_input,
                                                    context_input)

    # Ignore output state.
    sequence_length_mask = array_ops.sequence_mask(sequence_length)
    rnn_layer = keras_layers.RNN(cell=rnn_cell,
                                 return_sequences=return_sequences)
    rnn_outputs = rnn_layer(sequence_input,
                            mask=sequence_length_mask,
                            training=mode == model_fn.ModeKeys.TRAIN)

    with variable_scope.variable_scope('logits', values=(rnn_outputs,)):
      logits = core_layers.dense(
          rnn_outputs,
          units=output_units,
          activation=None,
          kernel_initializer=init_ops.glorot_uniform_initializer())
    return logits

  return rnn_logit_fn


def _rnn_model_fn(features,
                  labels,
                  mode,
                  head,
                  rnn_cell,
                  sequence_feature_columns,
                  context_feature_columns,
                  return_sequences=False,
                  optimizer='Adagrad',
                  input_layer_partitioner=None,
                  config=None):
  """Recurrent Neural Net model_fn.

  Args:
    features: dict of `Tensor` and `SparseTensor` objects returned from
      `input_fn`.
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] with labels.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    head: A `head_lib._Head` instance.
    rnn_cell: A Keras RNN cell object.
    sequence_feature_columns: Iterable containing `FeatureColumn`s that
      represent sequential model inputs.
    context_feature_columns: Iterable containing `FeatureColumn`s that
      represent model inputs not associated with a specific timestep.
    return_sequences: A boolean indicating whether to return the last output
      in the output sequence, or the full sequence.
    optimizer: String, `tf.Optimizer` object, or callable that creates the
      optimizer to use for training. If not specified, will use the Adagrad
      optimizer with a default learning rate of 0.05 and gradient clip norm of
      5.0.
    input_layer_partitioner: Partitioner for input layer. Defaults
      to `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
    config: `RunConfig` object to configure the runtime settings.

  Returns:
    An `EstimatorSpec` instance.

  Raises:
    ValueError: If mode or optimizer is invalid, or features has the wrong type.
  """
  if not isinstance(features, dict):
    raise ValueError('features should be a dictionary of `Tensor`s. '
                     'Given type: {}'.format(type(features)))

  # If user does not provide an optimizer instance, use the optimizer specified
  # by the string with default learning rate and gradient clipping.
  if not isinstance(optimizer, optimizer_lib.Optimizer):
    optimizer = optimizers.get_optimizer_instance(
        optimizer, learning_rate=_DEFAULT_LEARNING_RATE)
    optimizer = extenders.clip_gradients_by_norm(optimizer, _DEFAULT_CLIP_NORM)

  num_ps_replicas = config.num_ps_replicas if config else 0
  partitioner = partitioned_variables.min_max_variable_partitioner(
      max_partitions=num_ps_replicas)
  with variable_scope.variable_scope(
      'rnn',
      values=tuple(six.itervalues(features)),
      partitioner=partitioner):
    input_layer_partitioner = input_layer_partitioner or (
        partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas,
            min_slice_size=64 << 20))

    logit_fn = _rnn_logit_fn_builder(
        output_units=head.logits_dimension,
        rnn_cell=rnn_cell,
        sequence_feature_columns=sequence_feature_columns,
        context_feature_columns=context_feature_columns,
        input_layer_partitioner=input_layer_partitioner,
        return_sequences=return_sequences)
    logits = logit_fn(features=features, mode=mode)

    def _train_op_fn(loss):
      """Returns the op to optimize the loss."""
      return optimizer.minimize(
          loss,
          global_step=training_util.get_global_step())

    return head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        train_op_fn=_train_op_fn,
        logits=logits)


def _assert_rnn_cell(cell, num_units, cell_type):
  """Assert arguments are valid and return rnn_cell."""
  if cell and (num_units or cell_type != USE_DEFAULT):
    raise ValueError(
        'num_units and cell_type must not be specified when using rnn_cell.')
  # TODO(b/118833464): Add a base interface for Keras RNN cells.
  if cell and not rnn._is_keras_rnn_cell(cell):  # pylint: disable=protected-access
    raise ValueError('Provided cell must be a Keras cell.')
  if not cell:
    if cell_type == USE_DEFAULT:
      cell_type = _SIMPLE_RNN_KEY
    cell = _make_rnn_cell(num_units, cell_type)
  return cell


class RNNClassifier(estimator.Estimator):
  """A classifier for TensorFlow RNN models.

  Trains a recurrent neural network model to classify instances into one of
  multiple classes.

  Example:

  ```python
  token_sequence = sequence_categorical_column_with_hash_bucket(...)
  token_emb = embedding_column(categorical_column=token_sequence, ...)

  estimator = RNNClassifier(
      sequence_feature_columns=[token_emb],
      num_units=[32, 16], cell_type='lstm')

  # Input builders
  def input_fn_train: # returns x, y
    pass
  estimator.train(input_fn=input_fn_train, steps=100)

  def input_fn_eval: # returns x, y
    pass
  metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
  def input_fn_predict: # returns x, None
    pass
  predictions = estimator.predict(input_fn=input_fn_predict)
  ```

  Input of `train` and `evaluate` should have following features,
  otherwise there will be a `KeyError`:

  * if `weight_column` is not `None`, a feature with
    `key=weight_column` whose value is a `Tensor`.
  * for each `column` in `sequence_feature_columns`:
    - a feature with `key=column.name` whose `value` is a `SparseTensor`.
  * for each `column` in `context_feature_columns`:
    - if `column` is a `_CategoricalColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `_WeightedCategoricalColumn`, two features: the first
      with `key` the id column name, the second with `key` the weight column
      name. Both features' `value` must be a `SparseTensor`.
    - if `column` is a `_DenseColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.

  Loss is calculated by using softmax cross entropy.

  @compatibility(eager)
  Estimators are not compatible with eager execution.
  @end_compatibility
  """

  def __init__(self,
               sequence_feature_columns,
               context_feature_columns=None,
               num_units=None,
               cell_type=USE_DEFAULT,
               rnn_cell=None,
               model_dir=None,
               n_classes=2,
               weight_column=None,
               label_vocabulary=None,
               optimizer='Adagrad',
               loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE,
               input_layer_partitioner=None,
               config=None):
    """Initializes a `RNNClassifier` instance.

    Args:
      sequence_feature_columns: An iterable containing the `FeatureColumn`s
        that represent sequential input. All items in the set should either be
        sequence columns (e.g. `sequence_numeric_column`) or constructed from
        one (e.g. `embedding_column` with `sequence_categorical_column_*` as
        input).
      context_feature_columns: An iterable containing the `FeatureColumn`s
        for contextual input. The data represented by these columns will be
        replicated and given to the RNN at each timestep. These columns must be
        instances of classes derived from `_DenseColumn` such as
        `numeric_column`, not the sequential variants.
      num_units: Iterable of integer number of hidden units per RNN layer. If
        set, `cell_type` must also be specified and `rnn_cell` must be `None`.
      cell_type: A string specifying the cell type. Supported strings are:
        `'simple_rnn'`, `'lstm'`, and `'gru'`. If set, `num_units` must also be
        specified and `rnn_cell` must be `None`.
      rnn_cell: A Keras RNN cell that will be used to construct the RNN. If set,
        `num_units` and `cell_type` cannot be set. This is for advanced users
        who need additional customization beyond `num_units` and `cell_type`.
        Note that `tf.keras.layers.StackedRNNCells` is needed for stacked RNNs.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      n_classes: Number of label classes. Defaults to 2, namely binary
        classification. Must be > 1.
      weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example. If it is a string, it is
        used as a key to fetch weight tensor from the `features`. If it is a
        `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
        then weight_column.normalizer_fn is applied on it to get weight tensor.
      label_vocabulary: A list of strings represents possible label values. If
        given, labels must be string type and have any value in
        `label_vocabulary`. If it is not given, that means labels are
        already encoded as integer or float within [0, 1] for `n_classes=2` and
        encoded as integer values in {0, 1,..., n_classes-1} for `n_classes`>2 .
        Also there will be errors if vocabulary is not provided and labels are
        string.
      optimizer: An instance of `tf.Optimizer` or string specifying optimizer
        type. Defaults to Adagrad optimizer.
      loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how
        to reduce training loss over batch. Defaults to `SUM_OVER_BATCH_SIZE`.
      input_layer_partitioner: Optional. Partitioner for input layer. Defaults
        to `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
      config: `RunConfig` object to configure the runtime settings.

    Raises:
      ValueError: If `num_units`, `cell_type`, and `rnn_cell` are not
        compatible.
    """
    rnn_cell = _assert_rnn_cell(rnn_cell, num_units, cell_type)

    if n_classes == 2:
      head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(  # pylint: disable=protected-access
          weight_column=weight_column,
          label_vocabulary=label_vocabulary,
          loss_reduction=loss_reduction)
    else:
      head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(  # pylint: disable=protected-access
          n_classes,
          weight_column=weight_column,
          label_vocabulary=label_vocabulary,
          loss_reduction=loss_reduction)

    def _model_fn(features, labels, mode, config):
      return _rnn_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          rnn_cell=rnn_cell,
          sequence_feature_columns=tuple(sequence_feature_columns or []),
          context_feature_columns=tuple(context_feature_columns or []),
          return_sequences=False,
          optimizer=optimizer,
          input_layer_partitioner=input_layer_partitioner,
          config=config)
    super(RNNClassifier, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)


class RNNEstimator(estimator.Estimator):
  """An Estimator for TensorFlow RNN models with user-specified head.

  Example:

  ```python
  token_sequence = sequence_categorical_column_with_hash_bucket(...)
  token_emb = embedding_column(categorical_column=token_sequence, ...)

  estimator = RNNEstimator(
      head=tf.contrib.estimator.regression_head(),
      sequence_feature_columns=[token_emb],
      num_units=[32, 16], cell_type='lstm')

  # Or with custom RNN cell:
  cells = [ tf.keras.layers.LSTMCell(size, dropout=0.5) for size in [32, 16] ]
  rnn_cell = tf.keras.layers.StackedRNNCells(cells, reverse_state_order=True)

  estimator = RNNEstimator(
      head=tf.contrib.estimator.regression_head(),
      sequence_feature_columns=[token_emb],
      rnn_cell=rnn_cell)

  Note: If you would like to use multiple cells, you need to use
  `StackedRNNCells` with `reverse_state_order` set to True as in the example
  above.

  # Input builders
  def input_fn_train: # returns x, y
    pass
  estimator.train(input_fn=input_fn_train, steps=100)

  def input_fn_eval: # returns x, y
    pass
  metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
  def input_fn_predict: # returns x, None
    pass
  predictions = estimator.predict(input_fn=input_fn_predict)
  ```

  Input of `train` and `evaluate` should have following features,
  otherwise there will be a `KeyError`:

  * if the head's `weight_column` is not `None`, a feature with
    `key=weight_column` whose value is a `Tensor`.
  * for each `column` in `sequence_feature_columns`:
    - a feature with `key=column.name` whose `value` is a `SparseTensor`.
  * for each `column` in `context_feature_columns`:
    - if `column` is a `_CategoricalColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `_WeightedCategoricalColumn`, two features: the first
      with `key` the id column name, the second with `key` the weight column
      name. Both features' `value` must be a `SparseTensor`.
    - if `column` is a `_DenseColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.

  Loss and predicted output are determined by the specified head.

  @compatibility(eager)
  Estimators are not compatible with eager execution.
  @end_compatibility
  """

  def __init__(self,
               head,
               sequence_feature_columns,
               context_feature_columns=None,
               num_units=None,
               cell_type=USE_DEFAULT,
               rnn_cell=None,
               return_sequences=False,
               model_dir=None,
               optimizer='Adagrad',
               input_layer_partitioner=None,
               config=None):
    """Initializes a `RNNEstimator` instance.

    Args:
      head: A `_Head` instance constructed with a method such as
        `tf.contrib.estimator.multi_label_head`. This specifies the model's
        output and loss function to be optimized.
      sequence_feature_columns: An iterable containing the `FeatureColumn`s
        that represent sequential input. All items in the set should either be
        sequence columns (e.g. `sequence_numeric_column`) or constructed from
        one (e.g. `embedding_column` with `sequence_categorical_column_*` as
        input).
      context_feature_columns: An iterable containing the `FeatureColumn`s
        for contextual input. The data represented by these columns will be
        replicated and given to the RNN at each timestep. These columns must be
        instances of classes derived from `_DenseColumn` such as
        `numeric_column`, not the sequential variants.
      num_units: Iterable of integer number of hidden units per RNN layer. If
        set, `cell_type` must also be specified and `rnn_cell` must be
        `None`.
      cell_type: A string specifying the cell type. Supported strings are:
        `'simple_rnn'`, `'lstm'`, and `'gru'`. If set, `num_units` must also be
        specified and `rnn_cell` must be `None`.
      rnn_cell: A Keras RNN cell that will be used to construct the RNN. If set,
        `num_units` and `cell_type` cannot be set. This is for advanced users
        who need additional customization beyond `num_units` and `cell_type`.
        Note that `tf.keras.layers.StackedRNNCells` is needed for stacked RNNs.
      return_sequences: A boolean indicating whether to return the last output
        in the output sequence, or the full sequence.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      optimizer: An instance of `tf.Optimizer` or string specifying optimizer
        type. Defaults to Adagrad optimizer.
      input_layer_partitioner: Optional. Partitioner for input layer. Defaults
        to `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
      config: `RunConfig` object to configure the runtime settings.

    Raises:
      ValueError: If `num_units`, `cell_type`, and `rnn_cell` are not
        compatible.
    """
    rnn_cell = _assert_rnn_cell(rnn_cell, num_units, cell_type)

    def _model_fn(features, labels, mode, config):
      return _rnn_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          rnn_cell=rnn_cell,
          sequence_feature_columns=tuple(sequence_feature_columns or []),
          context_feature_columns=tuple(context_feature_columns or []),
          return_sequences=return_sequences,
          optimizer=optimizer,
          input_layer_partitioner=input_layer_partitioner,
          config=config)
    super(RNNEstimator, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)
