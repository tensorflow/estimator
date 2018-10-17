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
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras import metrics
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
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
    raise ValueError(
        'You must provide a labels Tensor. Given: None. '
        'Suggested troubleshooting steps: Check that your data contains '
        'your label feature. Check that your input_fn properly parses and '
        'returns labels.')
  with ops.name_scope(None, 'labels', (labels, logits)) as scope:
    labels = sparse_tensor.convert_to_tensor_or_sparse_tensor(labels)
    if isinstance(labels, sparse_tensor.SparseTensor):
      raise ValueError(
          'SparseTensor labels are not supported. '
          'labels must be a Tensor of shape [D0, D1, ..., DN, %s], '
          'e.g. [batch_size, %s]. '
          'Suggested Fix (1): Check the label feature in your data. '
          'Each example must contain %s value(s). If not, your choice of label '
          'was probably incorrect. '
          'Suggested Fix (2): In your input_fn, use '
          'tf.sparse_tensor_to_dense() to turn labels into a Tensor.'
          '' % (expected_labels_dimension, expected_labels_dimension,
                expected_labels_dimension))
    # Eager mode.
    if context.executing_eagerly():
      labels_rank = labels._rank()  # pylint: disable=protected-access
      logits_rank = logits._rank()  # pylint: disable=protected-access
      if (labels_rank is not None and logits_rank is not None
          and labels_rank == logits_rank - 1):
        labels = array_ops.expand_dims(labels, -1)
      return array_ops.identity(labels, name=scope)

    # Graph mode.
    if (labels.shape.ndims is not None and logits.shape.ndims is not None and
        labels.shape.ndims == logits.shape.ndims - 1):
      labels = array_ops.expand_dims(labels, -1)
    err_msg = (
        'labels shape must be [D0, D1, ... DN, {}]. '
        'Suggested Fix: check your n_classes argument to the estimator '
        'and/or the shape of your label.'.format(expected_labels_dimension))
    assert_rank = check_ops.assert_rank_at_least(labels, 2, message=err_msg)
    with ops.control_dependencies([assert_rank]):
      static_shape = labels.shape
      if static_shape.ndims is not None:
        final_dim = static_shape[-1]
        if (final_dim is not None) and (final_dim != expected_labels_dimension):
          raise ValueError(
              'Mismatched label shape. '
              'Expected labels dimension=%s.  Received %s. '
              'Suggested Fix:'
              'If your classifier expects one-hot encoding label,'
              'check your n_classes argument to the estimator '
              'and/or the shape of your label. '
              'Otherwise, check the shape of your label.' %
              (expected_labels_dimension, final_dim))
      logits_shape = array_ops.shape(logits)
      expected_labels_shape = array_ops.concat(
          [logits_shape[:-1], [expected_labels_dimension]], axis=0)
      labels_shape = array_ops.shape(labels)
      assert_dimension = check_ops.assert_equal(
          expected_labels_shape, labels_shape, message=err_msg,
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
      return array_ops.identity(weights, name=scope)

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
      return array_ops.identity(logits, name=scope)
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
      return array_ops.identity(unweighted_loss)
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


def _update_mean_metric_by_key(head_name, eval_metrics, key, values,
                               weights=None):
  mean_key = head_v1._summary_key(head_name, key)  # pylint: disable=protected-access
  # For prediction mean
  if key is metric_keys.MetricKeys.PREDICTION_MEAN:
    values = math_ops.to_float(values, name='predictions')
    if weights is not None:
      weights = weights_broadcast_ops.broadcast_weights(weights, values)
  eval_metrics[mean_key].update_state(
      values=values, sample_weight=weights)


def _regression_head(
    weight_column=None,
    label_dimension=1,
    loss_reduction=losses.Reduction.SUM,
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
      reduce training loss over batch. Defaults to `SUM`.
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
      loss_reduction=losses.Reduction.SUM,
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

  def _loss_spec(self, logits, labels, features):
    """Computes loss spec."""
    logits = _check_logits_final_dim(logits, self._logits_dimension)
    labels = _check_dense_labels_match_logits_and_reshape(
        labels=labels, logits=logits,
        expected_labels_dimension=self._logits_dimension)
    labels = math_ops.to_float(labels)
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
    training_loss = losses.compute_weighted_loss(
        unweighted_loss, weights=weights, reduction=self._loss_reduction)
    return head_v1.LossSpec(
        training_loss=training_loss,
        unreduced_loss=unweighted_loss,
        weights=weights,
        processed_labels=labels)

  def loss(self, logits, labels, features=None, mode=None,
           regularization_losses=None):
    """Returns loss. See `Head` for details."""
    del mode  # Unused for this head.
    with ops.name_scope(None, 'losses', (logits, labels, regularization_losses,
                                         features)):
      training_loss = self._loss_spec(logits, labels, features)[0]
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
    keys = metric_keys.MetricKeys
    with ops.name_scope(None, 'metrics', (regularization_losses,)):
      metric_key_list = [keys.LOSS_MEAN, keys.PREDICTION_MEAN, keys.LABEL_MEAN]
      if regularization_losses is not None:
        metric_key_list.append(keys.LOSS_REGULARIZATION)
      eval_metrics = {
          head_v1._summary_key(self._name, k):  # pylint: disable=protected-access
              metrics.Mean(name=k) for k in metric_key_list}
    return eval_metrics

  def update_metrics(self, eval_metrics, features, logits, labels,
                     regularization_losses=None):
    """Updates and returns the Eval metric ops. See `Head` for more details."""
    # Compute predictions
    predictions = self.predictions(logits)
    predicted_value = predictions[prediction_keys.PredictionKeys.PREDICTIONS]

    # Get loss spec for metrics
    _, unweighted_loss, weights, labels = self._loss_spec(
        logits, labels, features)

    # Update metrics
    keys = metric_keys.MetricKeys
    _update_mean_metric_by_key(self._name, eval_metrics, keys.LOSS_MEAN,
                               values=unweighted_loss, weights=weights)
    _update_mean_metric_by_key(self._name, eval_metrics, keys.LABEL_MEAN,
                               values=labels, weights=weights)
    _update_mean_metric_by_key(self._name, eval_metrics, keys.PREDICTION_MEAN,
                               values=predicted_value, weights=weights)
    if regularization_losses is not None:
      regularization_loss = math_ops.add_n(regularization_losses)
      _update_mean_metric_by_key(self._name, eval_metrics,
                                 keys.LOSS_REGULARIZATION,
                                 values=regularization_loss)
    return eval_metrics

  # TODO(yhliang): refactor it once other heads are added.
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
    with ops.name_scope(self._name, 'head'):
      # Predict
      predictions = self.predictions(logits)
      if mode == model_fn.ModeKeys.PREDICT:
        keys = prediction_keys.PredictionKeys
        regression_output = export_output.RegressionOutput(
            value=predictions[keys.PREDICTIONS])
        return model_fn._TPUEstimatorSpec(  # pylint: disable=protected-access
            mode=model_fn.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                head_v1._DEFAULT_SERVING_KEY: regression_output,  # pylint: disable=protected-access
                head_v1._REGRESS_SERVING_KEY: regression_output,  # pylint: disable=protected-access
                head_v1._PREDICT_SERVING_KEY: export_output.PredictOutput(  # pylint: disable=protected-access
                    predictions)
            })
      regularized_training_loss = self.loss(
          logits=logits, labels=labels, features=features, mode=mode,
          regularization_losses=regularization_losses)
      # Eval.
      if mode == model_fn.ModeKeys.EVAL:
        eval_metrics = self.metrics(regularization_losses=regularization_losses)
        return model_fn._TPUEstimatorSpec(  # pylint: disable=protected-access
            mode=model_fn.ModeKeys.EVAL,
            predictions=predictions,
            loss=regularized_training_loss,
            eval_metrics=head_v1._create_eval_metrics_tuple(  # pylint: disable=protected-access
                self.update_metrics, {
                    'eval_metrics': eval_metrics,
                    'features': features,
                    'logits': logits,
                    'labels': labels,
                    'regularization_losses': regularization_losses
                }))
      # Train.
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

      # Only summarize mean_loss for SUM reduction to preserve backwards
      # compatibility. Otherwise skip it to avoid unnecessary computation.
      if self._loss_reduction == losses.Reduction.SUM:
        training_loss, unreduced_loss, weights, _ = self._loss_spec(
            logits, labels, features)
        example_weight_sum = math_ops.reduce_sum(
            weights * array_ops.ones_like(unreduced_loss))
        mean_loss = training_loss / example_weight_sum
      else:
        mean_loss = None

    with ops.name_scope(''):
      keys = metric_keys.MetricKeys
      summary.scalar(
          head_v1._summary_key(self._name, keys.LOSS),  # pylint: disable=protected-access
          regularized_training_loss)
      if mean_loss is not None:
        summary.scalar(
            head_v1._summary_key(self._name, keys.LOSS_MEAN), mean_loss)  # pylint: disable=protected-access
      if regularization_losses is not None:
        regularization_loss = math_ops.add_n(regularization_losses)
        summary.scalar(
            head_v1._summary_key(self._name, keys.LOSS_REGULARIZATION),  # pylint: disable=protected-access
            regularization_loss)
    return model_fn._TPUEstimatorSpec(  # pylint: disable=protected-access
        mode=model_fn.ModeKeys.TRAIN,
        predictions=predictions,
        loss=regularized_training_loss,
        train_op=train_op)
