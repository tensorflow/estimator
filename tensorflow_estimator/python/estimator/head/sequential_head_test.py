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
"""Tests for sequential_head.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow_estimator.python.estimator.util import tf_keras
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.head import binary_class_head as binary_head_lib
from tensorflow_estimator.python.estimator.head import head_utils as test_lib
from tensorflow_estimator.python.estimator.head import multi_class_head as multi_head_lib
from tensorflow_estimator.python.estimator.head import multi_head
from tensorflow_estimator.python.estimator.head import sequential_head as seq_head_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys


def _convert_to_tensor(features):
  """Converts an arrays or dict of arrays to tensors or dict of tensors."""
  if isinstance(features, dict):
    if set(features.keys()) == set(['indices', 'values', 'dense_shape']):
      return tf.sparse.SparseTensor(**features)
    for col in features:
      features[col] = _convert_to_tensor(features[col])
    return features
  return ops.convert_to_tensor(features)


@test_util.run_all_in_graph_and_eager_modes
class TestFlatten(tf.test.TestCase, parameterized.TestCase):
  """Tests flatten functions."""

  @parameterized.named_parameters(
      {
          'testcase_name': 'one_dim_sparse_tensor',
          'tensor': {
              'indices': ((0, 0), (0, 1), (1, 0)),
              'values': (1, 2, 3),
              'dense_shape': (2, 2)
          },
          'expected': [[1], [2], [3]]
      }, {
          'testcase_name': 'multi_dim_sparse_tensor',
          'tensor': {
              'indices': ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0),
                          (1, 0, 1)),
              'values': (1, 2, 3, 4, 5, 6),
              'dense_shape': (2, 2, 2)
          },
          'expected': [[1, 2], [3, 4], [5, 6]]
      }, {
          'testcase_name': 'one_dim_dense_tensor',
          'tensor': [[1, 2], [3, 4]],
          'expected': [[1], [2], [3]]
      }, {
          'testcase_name': 'multi_dim_dense_tensor',
          'tensor': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
          'expected': [[1, 2], [3, 4], [5, 6]]
      }, {
          'testcase_name': 'unsorted_sparse_indices',
          'tensor': {
              'indices': ((0, 0), (1, 0), (0, 1)),
              'values': (1, 3, 2),
              'dense_shape': (2, 2)
          },
          'expected': [[1], [2], [3]]
      })
  def test_flatten_tensor(self, tensor, expected):
    """Tests the output of the `_flatten_tensor` function.

    Args:
      tensor: Dense or sparse array.
      expected: Array with expected output of `_flatten_tensor`.
    """
    sequence_mask = np.array([[1, 1], [1, 0]])
    tensor = _convert_to_tensor(tensor)
    flat_tensor = seq_head_lib._flatten_tensor(
        tensor, sequence_mask, expected_length=sequence_mask.sum())
    if tf.executing_eagerly():
      self.assertAllEqual(flat_tensor, expected)
      return
    with self.cached_session() as sess:
      self.assertAllEqual(sess.run(flat_tensor), expected)

  def _test_flatten_method(self, features, feature_columns):
    """Runs seq head's `_flatten` method and returns output for testing."""
    head = seq_head_lib.SequentialHeadWrapper(
        static_head=None,
        sequence_length_mask='sequence_mask',
        feature_columns=feature_columns)
    labels = {
        'indices': ((0, 0), (0, 1), (1, 0)),
        'values': (1, 2, 3),
        'dense_shape': (2, 2)
    }
    logits = np.array([[[10], [11]], [[12], [13]]])

    features = _convert_to_tensor(features)
    labels = tf.sparse.SparseTensor(**labels)
    logits = ops.convert_to_tensor(logits)
    output = head._flatten(labels, logits, features)
    if tf.executing_eagerly():
      return output
    with self.cached_session() as sess:
      return sess.run(output)

  def test_flatten_method(self):
    """Tests output of `_flatten` method."""
    features = {'sequence_mask': np.array([[1, 1], [1, 0]])}
    expected_output = ([[1], [2], [3]], [[10], [11], [12]], {})
    output = self._test_flatten_method(features, feature_columns=[])
    self.assertAllClose(expected_output, output)

  def test_flatten_with_one_feature_columns(self):
    """Tests output of `_flatten` method with one feature column provided."""
    features = {
        'sequence_mask': np.array([[1, 1], [1, 0]]),
        'weights': np.array([[0.5, 0.5], [1., 0]])
    }
    expected_output = ([[1], [2], [3]], [[10], [11], [12]], {
        'weights': np.array([[0.5], [0.5], [1.]])
    })
    output = self._test_flatten_method(features, feature_columns='weights')
    self.assertAllClose(expected_output, output)

  def test_flatten_with_multiple_feature_columns(self):
    """Tests `_flatten` method with multiple feature columns provided."""
    features = {
        'sequence_mask': np.array([[1, 1], [1, 0]]),
        'a': np.array([[0.5, 0.5], [1., 0]]),
        'b': np.array([[1.5, 1.5], [2., 0]])
    }
    expected_output = ([[1], [2], [3]], [[10], [11], [12]], {
        'a': np.array([[0.5], [0.5], [1.]]),
        'b': np.array([[1.5], [1.5], [2.]])
    })
    output = self._test_flatten_method(features, feature_columns=['a', 'b'])
    self.assertAllClose(expected_output, output)

  def test_flatten_no_mask(self):
    """Tests error in `_flatten` method when sequence mask is not provided."""
    features = {}
    with self.assertRaisesRegexp(
        ValueError, (r'The provided sequence_length_mask key `sequence_mask` '
                     r'should be included in.* Found keys: \[\].')):
      _ = self._test_flatten_method(features, feature_columns=[])

  def test_flatten_missing_feature(self):
    """Tests error in `_flatten` method when feature is not provided."""
    features = {'sequence_mask': np.array([[1, 1], [1, 0]])}
    with self.assertRaisesRegexp(
        ValueError, '`weights` column expected in features dictionary.'):
      _ = self._test_flatten_method(features, feature_columns=['weights'])

  def test_flatten_tensor_wrong_feature_dim(self):
    """Tests `_flatten` method when feature has wrong dimension."""
    features = {
        'sequence_mask': np.array([[1, 1], [1, 0]]),
        'weights': np.array([0.5, 0.5, 1., 0])
    }
    with self.assertRaisesRegexp(
        ValueError, 'Input tensor expected to have at least 2 dimensions.'):
      _ = self._test_flatten_method(features, feature_columns=['weights'])

  def test_flatten_tensor_wrong_feature_mask(self):
    """Tests `_flatten` with feature mask different from provided mask."""
    features = {'sequence_mask': np.array([[1, 1], [1, 1]])}
    error = (
        ValueError
        if tf.executing_eagerly() else tf.errors.InvalidArgumentError)
    with self.assertRaisesRegexp(
        error, 'Tensor shape is incompatible with provided mask.'):
      _ = self._test_flatten_method(features, feature_columns=[])

  def test_flatten_tensor_wrong_mask_dim(self):
    """Tests `_flatten` with mask that has wrong dimensions."""
    features = {'sequence_mask': np.array([1, 1])}
    with self.assertRaisesRegexp(
        ValueError, 'Mask is expected to have two dimensions, got .* instead.'):
      _ = self._test_flatten_method(features, feature_columns=[])


class _MockHead(object):
  """A static head to be wrapped in a sequential head, for testing."""

  def metrics(self, regularization_losses=None):
    return regularization_losses

  def loss(self, **kwargs):
    return kwargs

  def create_estimator_spec(self, **kwargs):
    Spec = collections.namedtuple('Spec', ['predictions', 'kwargs'])  # pylint: disable=invalid-name
    return Spec(predictions={}, kwargs=kwargs)


@test_util.run_all_in_graph_and_eager_modes
class TestSequentialHead(tf.test.TestCase):
  """Tests sequential head methods."""

  def _assert_equal(self, d, dref, session=None):
    """Recursively checks that all items of a dictionary are close.

    Dictionary can contain numerical values, `Tensor` objects or dictionaries of
    the former.

    If an item is a `Tensor`, its value is evaluated then compared to the
    reference.

    Args:
      d: Dictionary to check.
      dref: Dictionary to use as a reference for checks.
      session: A `tf.Session` object.
    """
    for key, ref_item in dref.items():
      if isinstance(ref_item, dict):
        self._assert_equal(d[key], dref=ref_item, session=session)
      elif isinstance(d[key], tf.Tensor):
        self.assertAllClose(
            session.run(d[key]) if session else d[key], ref_item)
      else:
        self.assertEqual(d[key], ref_item)

  def test_predictions(self):
    """Tests predictions output.

    Use `predictions` method in eager execution, else `create_estimator_spec` in
    PREDICT mode.

    logits = [[0.3, -0.4], [0.2, 0.2]]
    logistics = 1 / (1 + exp(-logits))
              = [[0.57, 0.40], [0.55, 0.55]]
    """
    head = seq_head_lib.SequentialHeadWrapper(binary_head_lib.BinaryClassHead(),
                                              'sequence_mask')

    logits = [[[0.3], [-0.4]], [[0.2], [0.2]]]
    expected_logistics = [[[0.574443], [0.401312]], [[0.549834], [0.549834]]]

    features = {
        'sequence_mask': ops.convert_to_tensor(np.array([[1, 1], [1, 0]]))
    }

    keys = prediction_keys.PredictionKeys
    if tf.executing_eagerly():
      predictions = head.predictions(
          logits=logits, keys=[keys.LOGITS, keys.LOGISTIC])
      self.assertItemsEqual(predictions.keys(), [keys.LOGITS, keys.LOGISTIC])
      self.assertAllClose(logits, predictions[keys.LOGITS])
      self.assertAllClose(expected_logistics, predictions[keys.LOGISTIC])
      return

    spec = head.create_estimator_spec(
        features=features,
        mode=ModeKeys.PREDICT,
        logits=logits,
        trainable_variables=[tf.Variable([1.0, 2.0], dtype=tf.dtypes.float32)])
    self.assertIn('sequence_mask', spec.predictions)
    with self.cached_session() as sess:
      self.assertAllEqual(
          sess.run(spec.predictions['sequence_mask']),
          features['sequence_mask'])
      self.assertAllClose(logits, sess.run(spec.predictions[keys.LOGITS]))
      self.assertAllClose(expected_logistics,
                          sess.run(spec.predictions[keys.LOGISTIC]))

  def test_metrics(self):
    """Tests the `metrics` method.

    Tests that:
    - Returned metrics match the returned metrics of the static head.
    - `regularization_losses` argument is properly passed to the static head's
      method.
    """
    head = seq_head_lib.SequentialHeadWrapper(binary_head_lib.BinaryClassHead(),
                                              'mask')
    metrics = head.metrics(regularization_losses=2.5)
    keys = metric_keys.MetricKeys
    self.assertIn(keys.ACCURACY, metrics)
    self.assertIn(keys.LOSS_REGULARIZATION, metrics)

  def test_loss_args(self):
    """Tests that variables are flattened and passed to static head's method."""
    logits = [[1, 2], [3, 4]]
    labels = [[0, 1], [0, 2]]
    features = {'weights': [[0.3, 0.2], [0.5, 100]], 'mask': [[1, 1], [1, 0]]}
    head = seq_head_lib.SequentialHeadWrapper(_MockHead(), 'mask', 'weights')
    expected_output = {
        'logits': [[1], [2], [3]],
        'labels': [[0], [1], [0]],
        'features': {
            'weights': [[0.3], [0.2], [0.5]]
        },
        'mode': 'my-mode',
        'regularization_losses': 123
    }
    output = head.loss(
        logits=_convert_to_tensor(logits),
        labels=_convert_to_tensor(labels),
        features=_convert_to_tensor(features),
        mode='my-mode',
        regularization_losses=123)
    with self.cached_session() as sess:
      self._assert_equal(output, dref=expected_output, session=sess)

  def test_create_estimator_spec_args(self):
    """Tests that variables are flattened and passed to static head's method."""
    logits = [[1, 2], [3, 4]]
    labels = [[0, 1], [0, 2]]
    features = {'weights': [[0.3, 0.2], [0.5, 100]], 'mask': [[1, 1], [1, 0]]}
    head = seq_head_lib.SequentialHeadWrapper(_MockHead(), 'mask', 'weights')
    w = tf.Variable(1)
    update_op = w.assign_add(1)
    trainable_variables = [tf.Variable([1.0, 2.0], dtype=tf.dtypes.float32)]
    expected_output = {
        'logits': [[1], [2], [3]],
        'labels': [[0], [1], [0]],
        'features': {
            'weights': [[0.3], [0.2], [0.5]]
        },
        'mode': ModeKeys.TRAIN,
        'regularization_losses': 123,
        'optimizer': 'my-opt',
        'train_op_fn': 'my-train-op',
        'trainable_variables': trainable_variables,
        'update_ops': [update_op]
    }
    spec = head.create_estimator_spec(
        logits=_convert_to_tensor(logits),
        labels=_convert_to_tensor(labels),
        features=_convert_to_tensor(features),
        mode=ModeKeys.TRAIN,
        optimizer='my-opt',
        train_op_fn='my-train-op',
        regularization_losses=123,
        update_ops=[update_op],
        trainable_variables=trainable_variables)
    with self.cached_session() as sess:
      self.assertItemsEqual(spec.kwargs.keys(), expected_output.keys())
      self._assert_equal(spec.kwargs, dref=expected_output, session=sess)

  def test_head_properties(self):
    """Tests that the head's properties are correcly implemented."""
    static_head = binary_head_lib.BinaryClassHead(
        loss_reduction=tf.losses.Reduction.SUM, name='a_static_head')
    head = seq_head_lib.SequentialHeadWrapper(static_head,
                                              'a_sequence_mask_col')
    self.assertEqual(head.name, 'a_static_head_sequential')
    self.assertEqual(head.logits_dimension, 1)
    self.assertEqual(head.loss_reduction, tf.losses.Reduction.SUM)
    self.assertEqual(head.input_sequence_mask_key, 'a_sequence_mask_col')
    self.assertEqual(head.static_head.name, 'a_static_head')

  def test_loss_reduction(self):
    """Tests loss reduction.

    Use `loss` method in eager execution, else `create_estimator_spec` in TRAIN
    mode.

    logits = [[[2., 3., 4.], [5., -0.5, 0.]],
              [[-1.0, 2.0, 0.5], [_]]],
    labels = [[0, 1],
              [2, _]]
    weights = [[0.5, 0.2],
               [0.3, _]]
    loss = [0.5*2.40 + 0.2*5.41 + 0.3*1.74] / 3 = 0.94
    """
    static_head = multi_head_lib.MultiClassHead(
        n_classes=3, weight_column='weights')
    head = seq_head_lib.SequentialHeadWrapper(static_head, 'sequence_mask',
                                              'weights')
    expected_loss = 0.942783
    features = {
        'weights':
            tf.sparse.SparseTensor(
                indices=((0, 0), (0, 1), (1, 0)),
                values=(0.5, 0.2, 0.3),
                dense_shape=(2, 2)),
        'sequence_mask':
            ops.convert_to_tensor([[1, 1], [1, 0]])
    }
    logits = ops.convert_to_tensor([[[2., 3., 4.], [5., -0.5, 0.]],
                                    [[-1.0, 2.0, 0.5], [1.0, 0.5, 2.0]]])
    labels = tf.sparse.SparseTensor(
        indices=((0, 0), (0, 1), (1, 0)), values=(0, 1, 2), dense_shape=(2, 2))

    class _Optimizer(tf_keras.optimizers.Optimizer):

      def get_updates(self, loss, params):
        del params, loss
        return [tf.constant('op')]

      def get_config(self):
        config = super(_Optimizer, self).get_config()
        return config

    if tf.executing_eagerly():
      loss = head.loss(logits=logits, labels=labels, features=features)
    else:
      spec = head.create_estimator_spec(
          features,
          ModeKeys.TRAIN,
          logits,
          labels=labels,
          optimizer=_Optimizer('my_optimizer'),
          trainable_variables=[
              tf.Variable([1.0, 2.0], dtype=tf.dtypes.float32)
          ])
      with self.cached_session() as sess:
        loss = sess.run(spec.loss)
    self.assertAllClose(loss, expected_loss, atol=1e-4)

  def test_metrics_computation(self):
    """Runs metrics computation tests.

    Use `update_metrics` method in eager execution, else `create_estimator_spec`
    in EVAL mode.

    logits = [[-101, 102, -103], [104, _, _]]
    predicted_labels = [[0, 1, 0], [1, _, _]]
    labels = [[1, 1, 1], [1, _, _]]
    weights = [[2, 5, 1], [2, _, _]]

    loss = (101*2 + 103*1) / 10 = 30.5
    accuracy = (0 + 5 + 0 + 2) / (2 + 5 + 1 + 2) = 0.7
    prediction_mean = (0 + 5 + 0 + 2) / (2 + 5 + 1 + 2) = 0.7
    precision = (5 + 2) / (5 + 2) = 1.0
    recall = (5 + 2) / (2 + 5 + 1 + 2) = 0.7
    """
    static_head = binary_head_lib.BinaryClassHead(weight_column='weights')
    head = seq_head_lib.SequentialHeadWrapper(static_head, 'sequence_mask',
                                              'weights')

    features = {
        'sequence_mask': np.array([[1, 1, 1], [1, 0, 0]]),
        'weights': np.array([[2, 5, 1], [2, 100, 100]])
    }
    regularization_losses = [100.]
    logits = _convert_to_tensor([[-101, 102, -103], [104, 100, 100]])
    labels = tf.sparse.SparseTensor(
        values=[1, 1, 1, 1],
        indices=((0, 0), (0, 1), (0, 2), (1, 0)),
        dense_shape=(2, 3))
    features = _convert_to_tensor(features)
    expected_loss = 30.5
    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: expected_loss,
        keys.ACCURACY: 0.7,
        keys.PREDICTION_MEAN: 0.7,
        keys.LABEL_MEAN: 1.0,
        keys.LOSS_REGULARIZATION: 100,
        keys.PRECISION: 1.0,
        keys.RECALL: 0.7,
        keys.ACCURACY_BASELINE: 1.0,
        keys.AUC: 0.,
        keys.AUC_PR: 1.0
    }

    if tf.executing_eagerly():
      eval_metrics = head.metrics(regularization_losses=regularization_losses)
      updated_metrics = head.update_metrics(eval_metrics, features, logits,
                                            labels, regularization_losses)
      self.assertItemsEqual(expected_metrics.keys(), updated_metrics.keys())
      self.assertAllClose(
          expected_metrics,
          {k: updated_metrics[k].result() for k in updated_metrics})
      return

    spec = head.create_estimator_spec(
        features=features,
        mode=ModeKeys.EVAL,
        logits=logits,
        labels=labels,
        regularization_losses=regularization_losses,
        trainable_variables=[tf.Variable([1.0, 2.0], dtype=tf.dtypes.float32)])

    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      _ = sess.run(update_ops)
      self.assertAllClose(expected_metrics,
                          {k: value_ops[k].eval() for k in value_ops})

  def test_wrong_mask_type(self):
    """Tests error raised when the mask doesn't have proper type."""
    with self.assertRaisesRegexp(TypeError,
                                 '`sequence_mask` column must be a string.'):
      _ = seq_head_lib.SequentialHeadWrapper(None, sequence_length_mask=1)

  def test_wrong_feature_column_type(self):
    """Tests error raised when the feature column doesn't have proper type."""
    with self.assertRaisesRegexp(
        TypeError, '`feature_columns` must be either a string or an iterable'):
      _ = seq_head_lib.SequentialHeadWrapper(None, 'mask', feature_columns=1)

  def test_wrong_feature_column_type_in_iterable(self):
    """Tests error raised when the feature column doesn't have proper type."""
    with self.assertRaisesRegexp(TypeError,
                                 'Column must a string. Given type: .*.'):
      _ = seq_head_lib.SequentialHeadWrapper(None, 'mask', feature_columns=[1])

  def test_multi_head_provided(self):
    """Tests error raised when a multi-head is provided."""
    with self.assertRaisesRegexp(
        ValueError,
        '`MultiHead` is not supported with `SequentialHeadWrapper`.'):
      _ = seq_head_lib.SequentialHeadWrapper(
          multi_head.MultiHead(
              [binary_head_lib.BinaryClassHead(name='test-head')]))


if __name__ == '__main__':
  tf.test.main()
