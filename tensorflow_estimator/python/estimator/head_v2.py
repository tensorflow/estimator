# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Abstractions for the head(s) of a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras import metrics
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.summary import summary
from tensorflow.python.training import training_util
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import head as head_v1
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.export import export_output


class Head(object):
  """Interface for the head/top of a model.

  Head sits on top of the model network and handles computing the outputs of
  the network. Given logits (or output of a hidden layer), a Head knows how to
  compute predictions, loss, train_op, metrics and export outputs. It is meant
  to:

  1. Simplify writing model_fn and to make model_fn more configurable for
     Estimator.
  2. Simpilfy creating loss and metrics for the train and test loop in Eager
     execution.
  3. Support wide range of machine learning models. Since most heads can work
     with logits, they can support DNN, RNN, Wide, Wide&Deep,
     Global objectives, Gradient boosted trees and many other types
     of machine learning models.

  Common usage:
  Here is simplified model_fn to build a DNN regression model.
    ```python
    def _my_dnn_model_fn(features, labels, mode, params, config=None):
      # Optionally your callers can pass head to model_fn as a param.

      # TODO(yhliang): Modify it from contrib to core once migration is done.
      head = tf.contrib.estimator.regression_head(...)
      # TODO(yhliang): Update it to OO FeatureLayer once Rohan's CL is in.
      inputs = tf.feature_column.input_layer(features, ...)

      # Compute logits with tf.keras.layers API
      hidden_layer0 = tf.keras.layers.Dense(
          units=1000, activation="relu")(inputs)
      hidden_layer1 = tf.keras.layers.Dense(
          units=500, activation="relu")(hidden_layer0)
      logits = tf.keras.layers.Dense(
          units=head.logits_dimension, activation=None)(hidden_layer1)

      # Or use Keras model for logits computation
      model = tf.keras.Sequential()
      model.add(tf.keras.layers.Dense(units=1000, activation="relu"))
      model.add(tf.keras.layers.Dense(units=500, activation="relu"))
      model.add(tf.keras.layers.Dense(
         units=head.logits_dimension, activation=None))
      logits = model(inputs)

      return head.create_estimator_spec(
          features=features,
          labels=labels,
          mode=mode,
          logits=logits,
          optimizer=optimizer)
    ```

  There are cases where computing and applying gradients can not be meaningfully
  captured with `Optimizer`s or `train_op_fn`s that Tensorflow supports (for
  example, with sync optimizer). In such case, you can take the responsibility
  to define your own way to manipulate gradients and update the `train_op` in
  `model_fn`. Here is a common use case:
    ```python
    def _my_model_fn(features, labels, mode, params, config=None):
      # TODO(yhliang): Modify it from contrib to core once migration is done.
      head = tf.contrib.estimator.multi_class_head(n_classes=3)
      estimator_spec = head.create_estimator_spec(
          features=features,
          labels=labels,
          mode=mode,
          logits=logits,
          train_op_fn=lambda _: tf.no_op())
      if mode == model_fn.ModeKeys.TRAIN:
        optimizer = ...
        sync_opt = tf.train.SyncReplicasOptimizer(opt=optimizer, ...)
        update_op = sync_opt.minimize(
            estimator_spec.loss, global_step=tf.get_global_step())
        hooks = [sync.make_session_run_hook(is_chief)]
        # update train_op and hooks in EstimatorSpec
        estimator_spec.train_op = update_op
        estimator_spec.training_hooks = hooks
        return estimator_spec
    ```
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def name(self):
    """The name of this head.

    Returns:
      A string.
    """
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractproperty
  def logits_dimension(self):
    """Size of the last dimension of the logits `Tensor`.

    Often is the number of classes, labels, or real values to be predicted.
    Typically, logits is of shape `[batch_size, logits_dimension]`.

    Returns:
      The expected size of the `logits` tensor.
    """
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractproperty
  def loss_reduction(self):
    """One of `tf.losses.Reduction`.

    Describes how to reduce training loss over batch, such as mean or sum.

    Returns:
      The type of loss reduction used in the head.
    """
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def loss(self, logits, labels, features=None, mode=None,
           regularization_losses=None):
    """Returns a loss `Tensor` from provided logits.

    Note that, the args of `features` and `mode` are most likely not used, but
    some Head implementations may require them.

    Args:
      logits: Logits `Tensor` to be used for loss construction.
      labels: Labels `Tensor`, or `dict` mapping string label names to `Tensor`
        objects of the label values.
      features: Input `dict` mapping string feature names to `Tensor` or
        `SparseTensor` objects containing the values for that feature in a
        minibatch. Often to be used to fetch example-weight tensor.
      mode: Estimator's `ModeKeys`. To be used in case loss calculation is
        different in Train and Eval mode.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses.

    Returns:
      A scalar `Tensor` representing regularized training loss used in train and
      eval.
    """
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def predictions(self, logits):
    """Returns a `dict` of predictions from provided logits.

    Args:
      logits: Logits `Tensor` to be used for prediction construction.

    Returns:
      A `dict` of predicted `Tensor` keyed by prediction name.
    """
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def metrics(self, regularization_losses=None):
    """Returns a `dict` of metric objects.

    Args:
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses.

    Returns:
       A `dict` of metrics keyed by string name. The value is an instance of
       `Metric` class.
    """
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def update_metrics(self, eval_metrics, features, logits, labels,
                     mode=None, regularization_losses=None):
    """Updates metric objects and returns a `dict` of the updated metrics.

    Args:
      eval_metrics: A `dict` of metrics to be updated.
      features: Input `dict` mapping string feature names to `Tensor` or
        `SparseTensor` objects containing the values for that feature in a
        minibatch. Often to be used to fetch example-weight tensor.
      logits: logits `Tensor` to be used for metrics update.
      labels: Labels `Tensor`, or `dict` mapping string label names to `Tensor`
        objects of the label values.
      mode: Estimator's `ModeKeys`.
      regularization_losses: A list of additional scalar losses to be added to
        the training and evaluation loss, such as regularization losses.

    Returns:
       A `dict` of updated metrics keyed by name. The value is an instance of
       `Metric` class.
    """
    raise NotImplementedError('Calling an abstract method.')

  def _summary_key(self, key):
    return '{}/{}'.format(key, self.name) if self.name else key

  def create_estimator_spec(
      self, features, mode, logits, labels=None, optimizer=None,
      train_op_fn=None, regularization_losses=None):
    """Returns `EstimatorSpec` that a model_fn can return.

    It is recommended to pass all args via name.

    Args:
      features: Input `dict` mapping string feature names to `Tensor` or
        `SparseTensor` objects containing the values for that feature in a
        minibatch. Often to be used to fetch example-weight tensor.
      mode: Estimator's `ModeKeys`.
      logits: Logits `Tensor` to be used by the head.
      labels: Labels `Tensor`, or `dict` mapping string label names to `Tensor`
        objects of the label values.
      optimizer: `Optimizer` instance to optimize the loss in TRAIN mode.
        Namely, sets `train_op = optimizer.minimize(loss, global_step)`, which
        updates variables and increments `global_step`.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns an op
        to optimize the model with the loss in TRAIN mode. Used if `optimizer`
        is `None`. Exactly one of `train_op_fn` and `optimizer` must be set in
        TRAIN mode. By default, it is `None` in other modes. If you want to
        optimize loss yourself, you can pass `lambda _: tf.no_op()` and then use
        EstimatorSpec.loss to compute and apply gradients.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses.

    Returns:
      `EstimatorSpec`.
    """
    # Not all subclasses of Head will have implemented
    # _create_tpu_estimator_spec. If it is implemented, we can convert it to
    # the normal `EstimatorSpec` by calling the method of
    # `_TPUEstimatorSpec.as_estimator_spec()`.
    try:
      tpu_estimator_spec = (
          self._create_tpu_estimator_spec(
              features, mode, logits, labels, optimizer, train_op_fn,
              regularization_losses))
      return tpu_estimator_spec.as_estimator_spec()
    except NotImplementedError:
      raise NotImplementedError(
          'Subclasses of Head must implement `create_estimator_spec()` or '
          '_create_tpu_estimator_spec().')

  def _create_tpu_estimator_spec(
      self, features, mode, logits, labels=None, optimizer=None,
      train_op_fn=None, regularization_losses=None):
    """Returns `model_fn._TPUEstimatorSpec` that a model_fn can return.

    Args:
      features: Input `dict` mapping string feature names to `Tensor` or
        `SparseTensor` objects containing the values for that feature in a
        minibatch. Often to be used to fetch example-weight tensor.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` to be used by the head.
      labels: Labels `Tensor`, or `dict` mapping string label names to `Tensor`
        objects of the label values.
      optimizer: `Optimizer` instance to optimize the loss in TRAIN mode.
        Namely, sets `train_op = optimizer.minimize(loss, global_step)`, which
        updates variables and increments `global_step`.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns an op
        to optimize the model with the loss in TRAIN mode. Used if `optimizer`
        is `None`. Exactly one of `train_op_fn` and `optimizer` must be set in
        TRAIN mode. None is allowed in other modes. If you want to optimize loss
        yourself you can pass `lambda _: tf.no_op()` and then use
        EstimatorSpec.loss to compute and apply gradients.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses.

    Returns:
      A `model_fn._TPUEstimatorSpec' instance.
    """
    raise NotImplementedError(
        'TPUEstimatorSpec not available for this model head.')


# Note that, tensor shape checking is slow in Eager mode. To amend it, the
# tensor static shape is used for checking. The duplication of shape checking
# for eager mode in the following helper functions can be safely removed
# if there's some way to get around it in the future.

# Label shape error messages.
_LABEL_NONE_ERR_MSG = (
    'You must provide a labels Tensor. Given: None. '
    'Suggested troubleshooting steps: Check that your data contains your label '
    'feature. Check that your input_fn properly parses and returns labels.')

_SPARSE_LABEL_ERR_MSG = (
    'SparseTensor labels are not supported. Labels must be a Tensor of shape '
    '[D0, D1, ..., DN, {}], e.g. [batch_size, {}].Suggested Fix (1): Check the'
    ' label feature in your data. Each example must contain {} value(s). If '
    'not, your choice of label was probably incorrect. Suggested Fix (2): In '
    'your input_fn, use tf.sparse_tensor_to_dense() to turn labels into a '
    'Tensor.')

_MISMATCHED_LABEL_DIM_ERR_MSG = (
    'Mismatched label shape. Expected labels dimension={}.  Received {}. '
    'Suggested Fix: If your classifier expects one-hot encoding label, check '
    'your n_classes argument to the estimator and/or the shape of your label. '
    'Otherwise, check the shape of your label.')

_LABEL_SHAPE_ERR_MSG = (
    'labels shape must be [D0, D1, ... DN, {}]. Suggested Fix: check your '
    'n_classes argument to the head and/or the shape of your label.')


def _check_dense_labels_match_logits_and_reshape(
    labels, logits, expected_labels_dimension):
  """Checks labels shape matches logits, and reshapes if needed.

  Consider logits of shape [D0, D1, ... DN, logits_dimension]. Then labels
  shape must be [D0, D1, ... DN, expected_labels_dimension].
  If expected_labels_dimension=1, labels could be [D0, D1, ... DN] and this
  method reshapes them to [D0, D1, ... DN, 1].

  Args:
    labels: labels Tensor.
    logits: logits Tensor.
    expected_labels_dimension: Integer.

  Returns:
    Validated and reshaped labels Tensor.

  Raises:
    ValueError: If labels is a SparseTensor.
    ValueError: If labels shape is statically defined and fails validation.
    OpError: If labels shape is not statically defined and fails validation.
  """
  if labels is None:
    raise ValueError(_LABEL_NONE_ERR_MSG)
  with ops.name_scope(None, 'labels', (labels, logits)) as scope:
    labels = sparse_tensor.convert_to_tensor_or_sparse_tensor(labels)
    if isinstance(labels, sparse_tensor.SparseTensor):
      raise ValueError(_SPARSE_LABEL_ERR_MSG.format(
          expected_labels_dimension, expected_labels_dimension,
          expected_labels_dimension))
    # Eager mode.
    if context.executing_eagerly():
      labels_rank = labels._rank()  # pylint: disable=protected-access
      logits_rank = logits._rank()  # pylint: disable=protected-access
      if (labels_rank is not None and logits_rank is not None
          and labels_rank == logits_rank - 1):
        labels = array_ops.expand_dims(labels, -1)
        labels_rank += 1
      labels_shape = labels._shape_tuple()  # pylint: disable=protected-access
      if labels_rank < 2:
        raise ValueError('labels must have rank at least 2.  Received rank {}, '
                         'shape {}'.format(labels_rank, labels_shape))
      if labels_shape[-1] != expected_labels_dimension:
        raise ValueError(_MISMATCHED_LABEL_DIM_ERR_MSG.format(
            expected_labels_dimension, labels_shape[-1]))
      logits_shape = logits._shape_tuple()  # pylint: disable=protected-access
      expected_labels_shape = logits_shape[:-1] + (expected_labels_dimension,)
      if expected_labels_shape != labels_shape:
        raise ValueError(
            '{}, expected_labels_shape: {}. labels_shape: {}.'
            .format(_LABEL_SHAPE_ERR_MSG.format(expected_labels_dimension),
                    expected_labels_shape, labels_shape))
      return labels

    # Graph mode.
    if (labels.shape.ndims is not None and logits.shape.ndims is not None and
        labels.shape.ndims == logits.shape.ndims - 1):
      labels = array_ops.expand_dims(labels, -1)
    assert_rank = check_ops.assert_rank_at_least(
        labels, 2,
        message=_LABEL_SHAPE_ERR_MSG.format(expected_labels_dimension))
    with ops.control_dependencies([assert_rank]):
      static_shape = labels.shape
      if static_shape.ndims is not None:
        final_dim = static_shape[-1]
        if (final_dim is not None) and (final_dim != expected_labels_dimension):
          raise ValueError(_MISMATCHED_LABEL_DIM_ERR_MSG.format(
              expected_labels_dimension, final_dim))
      logits_shape = array_ops.shape(logits)
      expected_labels_shape = array_ops.concat(
          [logits_shape[:-1], [expected_labels_dimension]], axis=0)
      labels_shape = array_ops.shape(labels)
      assert_dimension = check_ops.assert_equal(
          expected_labels_shape, labels_shape,
          message=_LABEL_SHAPE_ERR_MSG.format(expected_labels_dimension),
          data=['expected_labels_shape: ', expected_labels_shape,
                'labels_shape: ', labels_shape])
      with ops.control_dependencies([assert_dimension]):
        return array_ops.identity(labels, name=scope)


def _get_weights_and_check_match_logits(
    features, weight_column, logits, allow_per_logit_weights=False):
  """Fetches weights from features and checks that the shape matches logits.

  Consider logits of shape [D0, D1, ... DN, logits_dimension]. Weights shape
  can be either:
  * [D0, D1, ... DN, logits_dimension] if `allow_per_logit_weights=True`.
  * [D0, D1, ... DN, 1]
  * [D0, D1, ... DN]: In this case, weights is reshaped into
    [D0, D1, ... DN, 1] to work with weight broadcasting rules.

  Args:
    features: The features dict that contains weights.
    weight_column: The weight column. If not given, this method returns 1.
    logits: logits Tensor.
    allow_per_logit_weights: Boolean. Whether we allow weights along the logits
      dimension, namely shape `[D0, D1, ... DN, logits_dimension]`.

  Returns:
    Validated and reshaped weights Tensor.

  Raises:
    ValueError: If the weights `Tensor` cannot be cast into float.
  """
  if allow_per_logit_weights:
    err_msg = (
        'weights shape must be [D0, D1, ... DN], [D0, D1, ... DN, 1] or '
        '[D0, D1, ... DN, logits_dimension]')
  else:
    err_msg = (
        'weights shape must be [D0, D1, ... DN] or [D0, D1, ... DN, 1]')
  with ops.name_scope(
      None, 'weights',
      values=tuple(six.itervalues(features)) + (logits,)) as scope:
    # Fetch the weights.
    if weight_column is None:
      return 1.
    # TODO(b/117839674): update feature_column
    if isinstance(weight_column, six.string_types):
      weight_column = feature_column_lib.numeric_column(
          key=weight_column, shape=(1,))
    if not isinstance(weight_column, feature_column_lib._NumericColumn):  # pylint: disable=protected-access
      raise TypeError('Weight column must be either a string or _NumericColumn.'
                      ' Given type: {}.'.format(type(weight_column)))
    weights = weight_column._get_dense_tensor(  # pylint: disable=protected-access
        feature_column_lib._LazyBuilder(features))  # pylint: disable=protected-access
    if not (weights.dtype.is_floating or weights.dtype.is_integer):
      raise ValueError('Weight column should be castable to float. '
                       'Given dtype: {}'.format(weights.dtype))
    weights = math_ops.to_float(weights, name='weights')
    # Validate the weights shape.
    # Eager mode.
    if context.executing_eagerly():
      weights_shape = weights._shape_tuple()  # pylint: disable=protected-access
      logits_shape = logits._shape_tuple()  # pylint: disable=protected-access
      weights_rank = weights._rank()  # pylint: disable=protected-access
      logits_rank = logits._rank()  # pylint: disable=protected-access
      if (weights_rank is not None and logits_rank is not None
          and weights_rank == logits_rank - 1):
        if logits_shape[:-1] != weights_shape:
          raise ValueError(
              '{}, logits_shape: {}. weights_shape: {}.'
              .format(err_msg, logits_shape, weights_shape))
        return array_ops.expand_dims(weights, -1, name=scope)
      supported_weights_shape = logits_shape[:-1] + (1,)
      if allow_per_logit_weights:
        if (logits_shape != weights_shape
            and supported_weights_shape != weights_shape):
          raise ValueError(
              '{}, logits_shape: {}. weights_shape: {}.'
              .format(err_msg, logits_shape, weights_shape))
      else:
        if supported_weights_shape != weights_shape:
          raise ValueError(
              '{}, logits_shape: {}. weights_shape: {}.'
              .format(err_msg, logits_shape, weights_shape))
      return weights

    # Graph mode.
    weights_shape = array_ops.shape(weights, name='weights_shape')
    logits_shape = array_ops.shape(logits, name='logits_shape')
    if (weights.shape.ndims is not None and logits.shape.ndims is not None and
        weights.shape.ndims == logits.shape.ndims - 1):
      assert_dimension = check_ops.assert_equal(
          logits_shape[:-1], weights_shape, message=err_msg,
          data=['logits_shape: ', logits_shape,
                'weights_shape: ', weights_shape])
      with ops.control_dependencies([assert_dimension]):
        return array_ops.expand_dims(weights, -1, name=scope)
    supported_weights_shape = array_ops.concat([logits_shape[:-1], [1]], axis=0)
    if allow_per_logit_weights:
      condition = math_ops.reduce_any(
          [math_ops.reduce_all(math_ops.equal(logits_shape, weights_shape)),
           math_ops.reduce_all(math_ops.equal(
               supported_weights_shape, weights_shape))])
      assert_dimension = control_flow_ops.Assert(
          condition=condition,
          data=[err_msg, 'logits_shape: ', logits_shape,
                'weights_shape: ', weights_shape])
    else:
      assert_dimension = check_ops.assert_equal(
          supported_weights_shape, weights_shape, message=err_msg,
          data=['logits_shape: ', logits_shape,
                'weights_shape: ', weights_shape])
    with ops.control_dependencies([assert_dimension]):
      return array_ops.identity(weights, name=scope)


def _check_logits_final_dim(logits, expected_logits_dimension):
  """Checks that logits shape is [D0, D1, ... DN, logits_dimension]."""
  with ops.name_scope(None, 'logits', (logits,)) as scope:
    logits = math_ops.to_float(logits)
    # Eager mode
    if context.executing_eagerly():
      logits_shape = logits._shape_tuple()  # pylint: disable=protected-access
      logits_rank = logits._rank()  # pylint: disable=protected-access
      if logits_rank < 2:
        raise ValueError('logits must have rank at least 2.  Received rank {}, '
                         'shape {}'.format(logits_rank, logits_shape))
      if logits_shape[-1] != expected_logits_dimension:
        raise ValueError(
            'logits shape must be [D0, D1, ... DN, logits_dimension], '
            'got {}.'.format(logits_shape))
      return logits
    # Graph mode
    logits_shape = logits.shape
    if logits_shape.ndims is None:
      logits_shape = array_ops.shape(logits)
    assert_rank = check_ops.assert_rank_at_least(
        logits, 2, data=[logits_shape],
        message='logits shape must be [D0, D1, ... DN, logits_dimension]')
    with ops.control_dependencies([assert_rank]):
      static_shape = logits.shape
      if static_shape.ndims is not None and static_shape[-1] is not None:
        if static_shape[-1] != expected_logits_dimension:
          raise ValueError(
              'logits shape must be [D0, D1, ... DN, logits_dimension], '
              'got {}.'.format(static_shape))
        return logits
      assert_dimension = check_ops.assert_equal(
          expected_logits_dimension, logits_shape[-1], data=[logits_shape],
          message='logits shape must be [D0, D1, ... DN, logits_dimension]')
      with ops.control_dependencies([assert_dimension]):
        return array_ops.identity(logits, name=scope)


def _validate_loss_fn_args(loss_fn):
  """Validates loss_fn arguments.

  Required arguments: labels, logits.
  Optional arguments: features, loss_reduction.

  Args:
    loss_fn: The loss function.

  Raises:
    ValueError: If the signature is unexpected.
  """
  loss_fn_args = function_utils.fn_args(loss_fn)
  for required_arg in ['labels', 'logits']:
    if required_arg not in loss_fn_args:
      raise ValueError(
          'loss_fn must contain argument: {}. '
          'Given arguments: {}'.format(required_arg, loss_fn_args))
  invalid_args = list(set(loss_fn_args) - set(
      ['labels', 'logits', 'features', 'loss_reduction']))
  if invalid_args:
    raise ValueError('loss_fn has unexpected args: {}'.format(invalid_args))


def _call_loss_fn(loss_fn, labels, logits, features, expected_loss_dim=1):
  """Calls loss_fn and checks the returned shape.

  For shape checking, eager uses the static dimension to improve performance.

  Args:
    loss_fn: The loss function.
    labels: Processed labels Tensor.
    logits: Logits Tensor of shape [D0, D1, ... DN, logits_dimension].
    features: Features dict.
    expected_loss_dim: The expected last dimension of loss Tensor.
  Returns:
    Loss Tensor with shape [D0, D1, ... DN, expected_loss_dim].
  Raises:
    ValueError: If the loss tensor shape is unexpected.
  """
  loss_fn_args = function_utils.fn_args(loss_fn)
  kwargs = {}
  if 'features' in loss_fn_args:
    kwargs['features'] = features
  with ops.name_scope(
      None, 'call_loss_fn',
      values=[labels, logits] + list(six.itervalues(features))):
    unweighted_loss = loss_fn(labels=labels, logits=logits, **kwargs)
    # Eager mode.
    if context.executing_eagerly():
      loss_shape = unweighted_loss._shape_tuple()  # pylint: disable=protected-access
      logits_shape = logits._shape_tuple()  # pylint: disable=protected-access
      expected_loss_shape = logits_shape[:-1] + (expected_loss_dim,)
      if loss_shape != expected_loss_shape:
        raise ValueError('loss_fn must return Tensor of shape '
                         '[D0, D1, ... DN, {}]. '.format(expected_loss_dim),
                         'logits_shape: ', logits_shape,
                         'loss_shape: ', loss_shape)
      return unweighted_loss
    # Graph mode.
    logits_shape = logits.shape
    if logits_shape.ndims is None:
      logits_shape = array_ops.shape(logits, name='logits_shape')
    expected_loss_shape = array_ops.concat(
        [logits_shape[:-1], [expected_loss_dim]], axis=0,
        name='expected_loss_shape')
    loss_shape = unweighted_loss.shape
    if loss_shape.ndims is None:
      loss_shape = array_ops.shape(unweighted_loss, name='loss_shape')
    check_loss_shape_op = control_flow_ops.Assert(
        math_ops.reduce_all(math_ops.equal(loss_shape, expected_loss_shape)),
        data=[
            'loss_fn must return Tensor of shape '
            '[D0, D1, ... DN, {}]. '.format(expected_loss_dim),
            'logits_shape: ', logits_shape, 'loss_shape: ', loss_shape],
        name='check_loss_shape')
    with ops.control_dependencies([check_loss_shape_op]):
      return array_ops.identity(unweighted_loss)


def _check_prediction_keys(pred_keys, valid_keys):
  pred_keys = list(pred_keys or [])
  for key in pred_keys:
    if key not in valid_keys:
      raise ValueError(
          'Prediction key must be in PredictionKeys, given: {}.'
          'Valid prediction keys include {}.'.format(key, valid_keys))
  return pred_keys


def _check_label_range(labels, n_classes, message=None):
  """Check if labels are in the range of [0, n_classes)."""
  with ops.name_scope(None, 'check_label_range', (labels,)):
    # Eager mode
    if context.executing_eagerly():
      assert_less = math_ops.reduce_all(
          math_ops.less_equal(labels, n_classes - 1))
      if not assert_less:
        raise ValueError('Labels must be <= {} - 1'.format(n_classes))
      assert_greater = math_ops.reduce_all(math_ops.greater_equal(labels, 0))
      if not assert_greater:
        raise ValueError('Labels must be >= 0')
      return labels
    # Graph mode
    assert_less = check_ops.assert_less_equal(
        labels,
        ops.convert_to_tensor(n_classes - 1, dtype=labels.dtype),
        message=message or 'Labels must be <= n_classes - 1')
    assert_greater = check_ops.assert_non_negative(
        labels, message=message or 'Labels must be >= 0')
    with ops.control_dependencies((assert_less, assert_greater)):
      return array_ops.identity(labels)


def _create_estimator_spec_train_op(head_name, optimizer, train_op_fn,
                                    regularized_training_loss):
  """Create train_op for estimator_spec."""
  with ops.name_scope(head_name, 'head'):
    if optimizer is not None:
      if train_op_fn is not None:
        raise ValueError('train_op_fn and optimizer cannot both be set.')
      train_op = optimizer.minimize(
          regularized_training_loss,
          global_step=training_util.get_global_step())
    elif train_op_fn is not None:
      train_op = train_op_fn(regularized_training_loss)
    else:
      raise ValueError('train_op_fn and optimizer cannot both be None.')
    train_op = head_v1._append_update_ops(train_op)  # pylint: disable=protected-access
    return train_op


def _create_estimator_spec_summary(summary_key_fn, regularized_training_loss,
                                   regularization_losses):
  """Create summary for estimator_spec."""
  with ops.name_scope(''):
    keys = metric_keys.MetricKeys
    summary.scalar(summary_key_fn(keys.LOSS), regularized_training_loss)
    if regularization_losses is not None:
      regularization_loss = math_ops.add_n(regularization_losses)
      summary.scalar(
          summary_key_fn(keys.LOSS_REGULARIZATION), regularization_loss)


def _multi_class_head(
    n_classes,
    weight_column=None,
    label_vocabulary=None,
    loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE,
    loss_fn=None,
    name=None):
  """Creates a 'Head' for multi class classification.

  The head expects `logits` with shape `[D0, D1, ... DN, n_classes]`.
  In many applications, the shape is `[batch_size, n_classes]`.

  `labels` must be a dense `Tensor` with shape matching `logits`, namely
  `[D0, D1, ... DN, 1]`. If `label_vocabulary` given, `labels` must be a string
  `Tensor` with values from the vocabulary. If `label_vocabulary` is not given,
  `labels` must be an integer `Tensor` with values specifying the class index.

  If `weight_column` is specified, weights must be of shape
  `[D0, D1, ... DN]`, or `[D0, D1, ... DN, 1]`.

  The loss is the weighted sum over the input dimensions. Namely, if the input
  labels have shape `[batch_size, 1]`, the loss is the weighted sum over
  `batch_size`.

  Also supports custom `loss_fn`. `loss_fn` takes `(labels, logits)` or
  `(labels, logits, features, loss_reduction)` as arguments and returns
  unreduced loss with shape `[D0, D1, ... DN, 1]`. `loss_fn` must support
  integer `labels` with shape `[D0, D1, ... DN, 1]`. Namely, the head applies
  `label_vocabulary` to the input labels before passing them to `loss_fn`.

  Args:
    n_classes: Number of classes, must be greater than 2 (for 2 classes, use
      `_BinaryLogisticHead`).
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    label_vocabulary: A list or tuple of strings representing possible label
      values. If it is not given, that means labels are already encoded as an
      integer within [0, n_classes). If given, labels must be of string type and
      have any value in `label_vocabulary`. Note that errors will be raised if
      `label_vocabulary` is not provided but labels are strings.
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch. Defaults to `SUM_OVER_BATCH_SIZE`.
    loss_fn: Optional loss function.
    name: name of the head. If provided, summary and metrics keys will be
      suffixed by `"/" + name`. Also used as `name_scope` when creating ops.

  Returns:
    An instance of `Head` for multi class classification.

  Raises:
    ValueError: If `n_classes`, `label_vocabulary` or `loss_reduction` is
      invalid.
  """
  return _MultiClassHead(
      n_classes=n_classes,
      weight_column=weight_column,
      label_vocabulary=label_vocabulary,
      loss_reduction=loss_reduction,
      loss_fn=loss_fn,
      name=name)


class _MultiClassHead(Head):
  """`Head` for multiclass classification."""

  def __init__(self,
               n_classes,
               weight_column=None,
               label_vocabulary=None,
               loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE,
               loss_fn=None,
               name=None):
    """See `_multi_class_head` for details."""
    if (n_classes is None) or (n_classes <= 2):
      raise ValueError('n_classes must be > 2: {}.'.format(n_classes))
    if label_vocabulary is not None and not isinstance(label_vocabulary,
                                                       (list, tuple)):
      raise ValueError(
          'label_vocabulary should be a list or a tuple. Given type: {}'.format(
              type(label_vocabulary)))
    if (loss_reduction not in losses.Reduction.all() or
        loss_reduction == losses.Reduction.NONE):
      raise ValueError('Invalid loss_reduction: {}'.format(loss_reduction))
    if loss_fn:
      _validate_loss_fn_args(loss_fn)
    self._n_classes = n_classes
    self._weight_column = weight_column
    self._label_vocabulary = label_vocabulary
    if label_vocabulary:
      self._class_string_table = lookup_ops.index_to_string_table_from_tensor(
          vocabulary_list=self._label_vocabulary, name='class_string_lookup')
      self._class_id_table = lookup_ops.index_table_from_tensor(
          vocabulary_list=tuple(self._label_vocabulary), name='class_id_lookup')
    self._loss_reduction = loss_reduction
    self._loss_fn = loss_fn
    self._name = name
    # Metric keys.
    keys = metric_keys.MetricKeys
    self._loss_mean_key = self._summary_key(keys.LOSS_MEAN)
    self._accuracy_key = self._summary_key(keys.ACCURACY)
    self._loss_regularization_key = self._summary_key(keys.LOSS_REGULARIZATION)

  @property
  def name(self):
    """See `Head` for details."""
    return self._name

  @property
  def logits_dimension(self):
    """See `Head` for details."""
    return self._n_classes

  @property
  def loss_reduction(self):
    """See `Head` for details."""
    return self._loss_reduction

  def _processed_labels(self, logits, labels):
    """Converts labels to integer id space."""
    labels = _check_dense_labels_match_logits_and_reshape(
        labels=labels, logits=logits, expected_labels_dimension=1)
    if self._label_vocabulary is None:
      if not labels.dtype.is_integer:
        raise ValueError('Labels dtype should be integer. Instead got {}.'.
                         format(labels.dtype))
      label_ids = labels
    else:
      if labels.dtype != dtypes.string:
        raise ValueError('Labels dtype should be string if there is a '
                         'vocabulary. Instead got {}'.format(labels.dtype))
      label_ids = self._class_id_table.lookup(labels)
    return _check_label_range(label_ids, self._n_classes)

  def _unweighted_loss_and_weights(self, logits, label_ids, features):
    """Computes loss spec."""
    if self._loss_fn:
      unweighted_loss = _call_loss_fn(
          loss_fn=self._loss_fn, labels=label_ids, logits=logits,
          features=features, expected_loss_dim=1)
    else:
      unweighted_loss = losses.sparse_softmax_cross_entropy(
          labels=label_ids, logits=logits, reduction=losses.Reduction.NONE)
      # Restore the squeezed dim, so unweighted_loss matches the weights shape.
      unweighted_loss = array_ops.expand_dims(unweighted_loss, axis=-1)
    weights = _get_weights_and_check_match_logits(
        features=features, weight_column=self._weight_column, logits=logits)
    return unweighted_loss, weights

  def loss(self, logits, labels, features=None, mode=None,
           regularization_losses=None):
    """Returns loss. See `Head` for details."""
    del mode  # Unused for this head.
    with ops.name_scope(None, 'losses', (logits, labels, regularization_losses,
                                         features)):
      logits = _check_logits_final_dim(logits, self.logits_dimension)
      label_ids = self._processed_labels(logits, labels)
      unweighted_loss, weights = self._unweighted_loss_and_weights(
          logits, label_ids, features)
      training_loss = losses.compute_weighted_loss(
          unweighted_loss, weights=weights, reduction=self._loss_reduction)
      regularization_loss = math_ops.add_n(
          regularization_losses) if regularization_losses is not None else None
      regularized_training_loss = (
          training_loss + regularization_loss if regularization_loss is not None
          else training_loss)
    return regularized_training_loss

  def predictions(self, logits, keys):
    """Return the predictions based on keys.

    Args:
      logits: logits `Tensor` with shape `[D0, D1, ... DN, logits_dimension]`.
        For many applications, the shape is `[batch_size, logits_dimension]`.
      keys: a list of prediction keys. Key can be either the class variable
        of prediction_keys.PredictionKeys or its string value, such as:
        prediction_keys.PredictionKeys.CLASSES or 'classes'.

    Returns:
      A dict of predictions.
    """
    pred_keys = prediction_keys.PredictionKeys
    valid_keys = [pred_keys.LOGITS, pred_keys.PROBABILITIES,
                  pred_keys.CLASS_IDS, pred_keys.CLASSES]
    keys = _check_prediction_keys(keys, valid_keys)
    logits = _check_logits_final_dim(logits, self.logits_dimension)
    predictions = {}
    with ops.name_scope(None, 'predictions', (logits,)):
      if pred_keys.LOGITS in keys:
        predictions[pred_keys.LOGITS] = logits
      if pred_keys.PROBABILITIES in keys:
        probabilities = nn.softmax(logits, name=pred_keys.PROBABILITIES)
        predictions[pred_keys.PROBABILITIES] = probabilities
      if pred_keys.CLASS_IDS in keys or pred_keys.CLASSES in keys:
        # class_ids's shape is [D0, D1, ... DN].
        class_ids = math_ops.argmax(logits, axis=-1, name=pred_keys.CLASS_IDS)
        # Expand to [batch_size, 1].
        class_ids = array_ops.expand_dims(class_ids, axis=-1)
        if pred_keys.CLASS_IDS in keys:
          predictions[pred_keys.CLASS_IDS] = class_ids
        if pred_keys.CLASSES in keys:
          if self._label_vocabulary:
            classes = self._class_string_table.lookup(class_ids)
          else:
            classes = string_ops.as_string(class_ids, name='str_classes')
          predictions[pred_keys.CLASSES] = classes
      return predictions

  def metrics(self, regularization_losses=None):
    """Creates metrics. See `Head` for details."""
    keys = metric_keys.MetricKeys
    with ops.name_scope(None, 'metrics', (regularization_losses,)):
      # Mean metric.
      eval_metrics = {}
      eval_metrics[self._loss_mean_key] = metrics.Mean(name=keys.LOSS_MEAN)
      if regularization_losses is not None:
        eval_metrics[self._loss_regularization_key] = metrics.Mean(
            name=keys.LOSS_REGULARIZATION)
      # Accuracy metric.
      eval_metrics[self._accuracy_key] = (
          metrics.SparseCategoricalAccuracy(name=keys.ACCURACY))
    return eval_metrics

  def update_metrics(self, eval_metrics, features, logits, labels,
                     regularization_losses=None):
    """Updates and returns the Eval metric ops. See `Head` for more details."""
    # SparseCategoricalAccuracy metric uses logits as input for `y_pred` arg.
    logits = _check_logits_final_dim(logits, self.logits_dimension)
    label_ids = self._processed_labels(logits, labels)
    unweighted_loss, weights = self._unweighted_loss_and_weights(
        logits, label_ids, features)

    # Update metrics.
    eval_metrics[self._loss_mean_key].update_state(
        values=unweighted_loss, sample_weight=weights)
    eval_metrics[self._accuracy_key].update_state(
        y_true=label_ids, y_pred=logits, sample_weight=weights)

    if regularization_losses is not None:
      regularization_loss = math_ops.add_n(regularization_losses)
      eval_metrics[self._loss_regularization_key].update_state(
          values=regularization_loss)
    return eval_metrics

  def _create_tpu_estimator_spec(
      self, features, mode, logits, labels=None, optimizer=None,
      train_op_fn=None, regularization_losses=None):
    """Returns a `model_fn._TPUEstimatorSpec`.

    Args:
      features: Input `dict` of `Tensor` or `SparseTensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` with shape `[D0, D1, ... DN, logits_dimension]`.
        For many applications, the shape is `[batch_size, logits_dimension]`.
      labels: Labels integer or string `Tensor` with shape matching `logits`,
        namely `[D0, D1, ... DN, 1]` or `[D0, D1, ... DN]`. `labels` is
        required argument when `mode` equals `TRAIN` or `EVAL`.
      optimizer: `Optimizer` instance to optimize the loss in TRAIN mode.
        Namely, sets `train_op = optimizer.minimize(loss, global_step)`, which
        updates variables and increments `global_step`.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns
        `train_op`. Used if `optimizer` is `None`.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses. These losses are
        usually expressed as a batch average, so for best results users need to
        set `loss_reduction=SUM_OVER_BATCH_SIZE` or
        `loss_reduction=SUM_OVER_NONZERO_WEIGHTS` when creating the head to
        avoid scaling errors.

    Returns:
      A `model_fn._TPUEstimatorSpec` instance.

    Raises:
      ValueError: If both `train_op_fn` and `optimizer` are `None` in TRAIN
        mode, or if both are set.
    """
    # pylint: disable=protected-access
    with ops.name_scope(self._name, 'head'):
      # Predict.
      pred_keys = prediction_keys.PredictionKeys
      keys = [pred_keys.LOGITS, pred_keys.PROBABILITIES, pred_keys.CLASS_IDS,
              pred_keys.CLASSES]
      predictions = self.predictions(logits, keys)
      if mode == model_fn.ModeKeys.PREDICT:
        probabilities = predictions[pred_keys.PROBABILITIES]
        classifier_output = head_v1._classification_output(
            scores=probabilities, n_classes=self._n_classes,
            label_vocabulary=self._label_vocabulary)
        return model_fn._TPUEstimatorSpec(
            mode=model_fn.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                head_v1._DEFAULT_SERVING_KEY: classifier_output,
                head_v1._CLASSIFY_SERVING_KEY: classifier_output,
                head_v1._PREDICT_SERVING_KEY: export_output.PredictOutput(
                    predictions)
            })
      regularized_training_loss = self.loss(
          logits=logits, labels=labels, features=features, mode=mode,
          regularization_losses=regularization_losses)
      # Eval.
      if mode == model_fn.ModeKeys.EVAL:
        eval_metrics = self.metrics(regularization_losses=regularization_losses)
        return model_fn._TPUEstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions=predictions,
            loss=regularized_training_loss,
            eval_metrics=head_v1._create_eval_metrics_tuple(
                self.update_metrics, {
                    'eval_metrics': eval_metrics,
                    'features': features,
                    'logits': logits,
                    'labels': labels,
                    'regularization_losses': regularization_losses
                }))
      # Train.
      train_op = _create_estimator_spec_train_op(
          self._name, optimizer, train_op_fn, regularized_training_loss)
    # Create summary.
    _create_estimator_spec_summary(self._summary_key, regularized_training_loss,
                                   regularization_losses)
    return model_fn._TPUEstimatorSpec(
        mode=model_fn.ModeKeys.TRAIN,
        predictions=predictions,
        loss=regularized_training_loss,
        train_op=train_op)
# pylint: enable=protected-access


def _regression_head(
    weight_column=None,
    label_dimension=1,
    loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE,
    loss_fn=None,
    inverse_link_fn=None,
    name=None):
  """Creates a `Head` for regression using the `mean_squared_error` loss.

  The loss is the weighted sum over all input dimensions. Namely, if the input
  labels have shape `[batch_size, label_dimension]`, the loss is the weighted
  sum over both `batch_size` and `label_dimension`.

  The head expects `logits` with shape `[D0, D1, ... DN, label_dimension]`.
  In many applications, the shape is `[batch_size, label_dimension]`.

  The `labels` shape must match `logits`, namely
  `[D0, D1, ... DN, label_dimension]`. If `label_dimension=1`, shape
  `[D0, D1, ... DN]` is also supported.

  If `weight_column` is specified, weights must be of shape
  `[D0, D1, ... DN]`, `[D0, D1, ... DN, 1]` or
  `[D0, D1, ... DN, label_dimension]`.

  Supports custom `loss_fn`. `loss_fn` takes `(labels, logits)` or
  `(labels, logits, features)` as arguments and returns unreduced loss with
  shape `[D0, D1, ... DN, label_dimension]`.

  Also supports custom `inverse_link_fn`, also known as 'mean function'.
  `inverse_link_fn` takes `logits` as argument and returns predicted values.
  This function is the inverse of the link function defined in
  https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function
  Namely, for poisson regression, set `inverse_link_fn=tf.exp`.

  Args:
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    label_dimension: Number of regression labels per example. This is the size
      of the last dimension of the labels `Tensor` (typically, this has shape
      `[batch_size, label_dimension]`).
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch. Defaults to `SUM_OVER_BATCH_SIZE`.
    loss_fn: Optional loss function. Defaults to `mean_squared_error`.
    inverse_link_fn: Optional inverse link function, also known as 'mean
      function'. Defaults to identity.
    name: name of the head. If provided, summary and metrics keys will be
      suffixed by `"/" + name`. Also used as `name_scope` when creating ops.

  Returns:
    An instance of `Head` for linear regression.

  Raises:
    ValueError: If `label_dimension` or `loss_reduction` is invalid.
  """
  return _RegressionHead(
      weight_column=weight_column,
      label_dimension=label_dimension,
      loss_reduction=loss_reduction,
      loss_fn=loss_fn,
      inverse_link_fn=inverse_link_fn,
      name=name)


class _RegressionHead(Head):
  """`Head` for regression."""

  def __init__(
      self,
      label_dimension,
      weight_column=None,
      loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE,
      loss_fn=None,
      inverse_link_fn=None,
      name=None):
    """See `_regression_head` for details."""
    if label_dimension < 1:
      raise ValueError('Invalid label_dimension {}.'.format(label_dimension))
    if (loss_reduction not in losses.Reduction.all() or
        loss_reduction == losses.Reduction.NONE):
      raise ValueError('Invalid loss_reduction: {}'.format(loss_reduction))
    if loss_fn:
      _validate_loss_fn_args(loss_fn)
    self._logits_dimension = label_dimension
    self._weight_column = weight_column
    self._loss_reduction = loss_reduction
    self._loss_fn = loss_fn
    self._inverse_link_fn = inverse_link_fn
    self._name = name
    # Metric keys.
    keys = metric_keys.MetricKeys
    self._loss_mean_key = self._summary_key(keys.LOSS_MEAN)
    self._prediction_mean_key = self._summary_key(keys.PREDICTION_MEAN)
    self._label_mean_key = self._summary_key(keys.LABEL_MEAN)
    self._loss_regularization_key = self._summary_key(keys.LOSS_REGULARIZATION)

  @property
  def name(self):
    """See `Head` for details."""
    return self._name

  @property
  def logits_dimension(self):
    """See `Head` for details."""
    return self._logits_dimension

  @property
  def loss_reduction(self):
    """See `Head` for details."""
    return self._loss_reduction

  def _processed_labels(self, logits, labels):
    labels = _check_dense_labels_match_logits_and_reshape(
        labels=labels, logits=logits,
        expected_labels_dimension=self._logits_dimension)
    labels = math_ops.to_float(labels)
    return labels

  def _unweighted_loss_and_weights(self, logits, labels, features):
    """Computes loss spec."""
    if self._loss_fn:
      unweighted_loss = _call_loss_fn(
          loss_fn=self._loss_fn, labels=labels, logits=logits,
          features=features, expected_loss_dim=self._logits_dimension)
    else:
      unweighted_loss = losses.mean_squared_error(
          labels=labels, predictions=logits, reduction=losses.Reduction.NONE)
    weights = _get_weights_and_check_match_logits(
        features=features, weight_column=self._weight_column, logits=logits,
        allow_per_logit_weights=True)
    return unweighted_loss, weights

  def loss(self, logits, labels, features=None, mode=None,
           regularization_losses=None):
    """Returns loss. See `Head` for details."""
    del mode  # Unused for this head.
    with ops.name_scope(None, 'losses', (logits, labels, regularization_losses,
                                         features)):
      logits = _check_logits_final_dim(logits, self._logits_dimension)
      labels = self._processed_labels(logits, labels)
      unweighted_loss, weights = self._unweighted_loss_and_weights(
          logits, labels, features)
      training_loss = losses.compute_weighted_loss(
          unweighted_loss, weights=weights, reduction=self._loss_reduction)
      regularization_loss = math_ops.add_n(
          regularization_losses) if regularization_losses is not None else None
      regularized_training_loss = (
          training_loss + regularization_loss if regularization_loss is not None
          else training_loss)
    return regularized_training_loss

  def predictions(self, logits):
    """Create predictions. See `Head` for details."""
    # We need to check the logits dim for both eager and graph mode.
    logits = _check_logits_final_dim(logits, self._logits_dimension)
    pred_keys = prediction_keys.PredictionKeys
    with ops.name_scope(None, 'predictions', (logits,)):
      if self._inverse_link_fn:
        predicted_value = self._inverse_link_fn(logits)
        predictions = {
            pred_keys.PREDICTIONS: predicted_value,
            pred_keys.LOGITS: logits,
        }
      else:
        predicted_value = logits
        predictions = {
            pred_keys.PREDICTIONS: predicted_value}
    return predictions

  def metrics(self, regularization_losses=None):
    """Creates metrics. See `Head` for details."""
    with ops.name_scope(None, 'metrics', (regularization_losses,)):
      keys = metric_keys.MetricKeys
      eval_metrics = {}
      eval_metrics[self._loss_mean_key] = metrics.Mean(name=keys.LOSS_MEAN)
      eval_metrics[self._prediction_mean_key] = metrics.Mean(
          name=keys.PREDICTION_MEAN)
      eval_metrics[self._label_mean_key] = metrics.Mean(name=keys.LABEL_MEAN)

      if regularization_losses is not None:
        eval_metrics[self._loss_regularization_key] = metrics.Mean(
            name=keys.LOSS_REGULARIZATION)
    return eval_metrics

  def update_metrics(self, eval_metrics, features, logits, labels,
                     regularization_losses=None):
    """Updates and returns the Eval metric ops. See `Head` for more details."""
    # Compute predictions.
    predictions = self.predictions(logits)
    predicted_value = predictions[prediction_keys.PredictionKeys.PREDICTIONS]
    logits = _check_logits_final_dim(logits, self.logits_dimension)
    label_ids = self._processed_labels(logits, labels)
    unweighted_loss, weights = self._unweighted_loss_and_weights(
        logits, label_ids, features)

    # Update metrics.
    eval_metrics[self._loss_mean_key].update_state(
        values=unweighted_loss, sample_weight=weights)
    eval_metrics[self._label_mean_key].update_state(
        values=labels, sample_weight=weights)

    predicted_value = math_ops.to_float(predicted_value, name='predictions')
    if weights is not None:
      weights = weights_broadcast_ops.broadcast_weights(
          weights, predicted_value)
    eval_metrics[self._prediction_mean_key].update_state(
        values=predicted_value, sample_weight=weights)

    if regularization_losses is not None:
      regularization_loss = math_ops.add_n(regularization_losses)
      eval_metrics[self._loss_regularization_key].update_state(
          values=regularization_loss)
    return eval_metrics

  def _create_tpu_estimator_spec(
      self, features, mode, logits, labels=None, optimizer=None,
      train_op_fn=None, regularization_losses=None):
    """Returns an `EstimatorSpec`.

    Args:
      features: Input `dict` mapping string feature names to `Tensor` or
        `SparseTensor` objects containing the values for that feature in a
        minibatch. Often to be used to fetch example-weight tensor.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` with shape `[D0, D1, ... DN, logits_dimension]`.
        For many applications, the shape is `[batch_size, logits_dimension]`.
      labels: Labels `Tensor` with shape matching `logits`, namely
        `[D0, D1, ... DN, logits_dimension]`. When `logits_dimension=1`, shape
        `[D0, D1, ... DN]` is also supported. `labels` is a required argument
        when `mode` equals `TRAIN` or `EVAL`.
      optimizer: `Optimizer` instance to optimize the loss in TRAIN mode.
        Namely, sets `train_op = optimizer.minimize(loss, global_step)`, which
        updates variables and increments `global_step`.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns
        `train_op`. Used if `optimizer` is `None`.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses. These losses are
        usually expressed as a batch average, so for best results users need to
        set `loss_reduction=SUM_OVER_BATCH_SIZE` or
        `loss_reduction=SUM_OVER_NONZERO_WEIGHTS` when creating the head to
        avoid scaling errors.

    Returns:
      A `model_fn._TPUEstimatorSpec` instance.

    Raises:
      ValueError: If both `train_op_fn` and `optimizer` are `None` in TRAIN
        mode, or if both are set.
    """
    # pylint: disable=protected-access
    with ops.name_scope(self._name, 'head'):
      # Predict.
      predictions = self.predictions(logits)
      if mode == model_fn.ModeKeys.PREDICT:
        keys = prediction_keys.PredictionKeys
        regression_output = export_output.RegressionOutput(
            value=predictions[keys.PREDICTIONS])
        return model_fn._TPUEstimatorSpec(
            mode=model_fn.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                head_v1._DEFAULT_SERVING_KEY: regression_output,
                head_v1._REGRESS_SERVING_KEY: regression_output,
                head_v1._PREDICT_SERVING_KEY: export_output.PredictOutput(
                    predictions)
            })
      regularized_training_loss = self.loss(
          logits=logits, labels=labels, features=features, mode=mode,
          regularization_losses=regularization_losses)
      # Eval.
      if mode == model_fn.ModeKeys.EVAL:
        eval_metrics = self.metrics(regularization_losses=regularization_losses)
        return model_fn._TPUEstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions=predictions,
            loss=regularized_training_loss,
            eval_metrics=head_v1._create_eval_metrics_tuple(
                self.update_metrics, {
                    'eval_metrics': eval_metrics,
                    'features': features,
                    'logits': logits,
                    'labels': labels,
                    'regularization_losses': regularization_losses
                }))
      # Train.
      train_op = _create_estimator_spec_train_op(
          self._name, optimizer, train_op_fn, regularized_training_loss)
    # Create summary.
    _create_estimator_spec_summary(self._summary_key, regularized_training_loss,
                                   regularization_losses)
    return model_fn._TPUEstimatorSpec(
        mode=model_fn.ModeKeys.TRAIN,
        predictions=predictions,
        loss=regularized_training_loss,
        train_op=train_op)
    # pylint: enable=protected-access
