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
"""Tests for multi_label_head.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column_v2 as feature_column
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import monitored_session
from tensorflow_estimator.python.estimator.canned import dnn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.head import head_utils as test_lib
from tensorflow_estimator.python.estimator.head import multi_label_head as head_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys


def _sigmoid_cross_entropy(labels, logits):
  """Returns sigmoid cross entropy averaged over classes."""
  sigmoid_logits = 1 / (1 + np.exp(-logits))
  unreduced_result = (
      -labels * np.log(sigmoid_logits)
      -(1 - labels) * np.log(1 - sigmoid_logits))
  # Mean over classes
  return np.mean(unreduced_result, axis=-1, keepdims=True)


class MultiLabelHead(test.TestCase):

  def test_n_classes_is_none(self):
    with self.assertRaisesRegexp(
        ValueError,
        r'n_classes must be > 1 for multi-label classification\. Given: None'):
      head_lib.MultiLabelHead(n_classes=None)

  def test_n_classes_is_1(self):
    with self.assertRaisesRegexp(
        ValueError,
        r'n_classes must be > 1 for multi-label classification\. Given: 1'):
      head_lib.MultiLabelHead(n_classes=1)

  def test_threshold_too_small(self):
    with self.assertRaisesRegexp(
        ValueError,
        r'thresholds must be in \(0, 1\) range\. Given: 0\.0'):
      head_lib.MultiLabelHead(n_classes=2, thresholds=[0., 0.5])

  def test_threshold_too_large(self):
    with self.assertRaisesRegexp(
        ValueError,
        r'thresholds must be in \(0, 1\) range\. Given: 1\.0'):
      head_lib.MultiLabelHead(n_classes=2, thresholds=[0.5, 1.0])

  def test_label_vocabulary_dict(self):
    with self.assertRaisesRegexp(
        ValueError,
        r'label_vocabulary must be a list or tuple\. '
        r'Given type: <(type|class) \'dict\'>'):
      head_lib.MultiLabelHead(n_classes=2, label_vocabulary={'foo': 'bar'})

  def test_label_vocabulary_wrong_size(self):
    with self.assertRaisesRegexp(
        ValueError,
        r'Length of label_vocabulary must be n_classes \(3\). Given: 2'):
      head_lib.MultiLabelHead(n_classes=3, label_vocabulary=['foo', 'bar'])

  def test_invalid_loss_reduction(self):
    with self.assertRaisesRegexp(
        ValueError, r'Invalid loss_reduction: invalid_loss_reduction'):
      head_lib.MultiLabelHead(
          n_classes=3, loss_reduction='invalid_loss_reduction')
    with self.assertRaisesRegexp(
        ValueError, r'Invalid loss_reduction: none'):
      head_lib.MultiLabelHead(
          n_classes=3, loss_reduction=losses_utils.ReductionV2.NONE)

  def test_loss_fn_arg_labels_missing(self):
    def _loss_fn(logits):
      del logits  # Unused
    with self.assertRaisesRegexp(
        ValueError,
        r'loss_fn must contain argument: labels\. '
        r'Given arguments: \(\'logits\',\)'):
      head_lib.MultiLabelHead(n_classes=3, loss_fn=_loss_fn)

  def test_loss_fn_arg_logits_missing(self):
    def _loss_fn(labels):
      del labels  # unused
    with self.assertRaisesRegexp(
        ValueError,
        r'loss_fn must contain argument: logits\. '
        r'Given arguments: \(\'labels\',\)'):
      head_lib.MultiLabelHead(n_classes=3, loss_fn=_loss_fn)

  def test_loss_fn_arg_features_ok(self):
    def _loss_fn(labels, logits, features):
      del labels, logits, features  # Unused
    head_lib.MultiLabelHead(n_classes=3, loss_fn=_loss_fn)

  def test_loss_fn_arg_invalid(self):
    def _loss_fn(labels, logits, name=None):
      del labels, logits, name  # Unused
    with self.assertRaisesRegexp(
        ValueError,
        r'loss_fn has unexpected args: \[\'name\'\]'):
      head_lib.MultiLabelHead(n_classes=3, loss_fn=_loss_fn)

  def test_classes_for_class_based_metrics_invalid(self):
    with self.assertRaisesRegexp(
        ValueError,
        r'All classes_for_class_based_metrics must be in range \[0, 2\]\. '
        r'Given: -1'):
      head_lib.MultiLabelHead(
          n_classes=3, classes_for_class_based_metrics=[2, -1])

  def test_classes_for_class_based_metrics_string_invalid(self):
    with self.assertRaisesRegexp(
        ValueError, r'\'z\' is not in list'):
      head_lib.MultiLabelHead(
          n_classes=3, label_vocabulary=['a', 'b', 'c'],
          classes_for_class_based_metrics=['c', 'z'])

  def test_predict(self):
    n_classes = 4
    head = head_lib.MultiLabelHead(n_classes)
    self.assertEqual(n_classes, head.logits_dimension)

    logits = np.array(
        [[0., 1., 2., -1.], [-1., -2., -3., 1.]], dtype=np.float32)
    expected_probabilities = nn.sigmoid(logits)
    expected_export_classes = [[b'0', b'1', b'2', b'3']] * 2

    keys = prediction_keys.PredictionKeys
    preds = head.predictions(logits, [keys.LOGITS, keys.PROBABILITIES])
    self.assertAllClose(logits, self.evaluate(preds[keys.LOGITS]))
    self.assertAllClose(expected_probabilities,
                        self.evaluate(preds[keys.PROBABILITIES]))
    if context.executing_eagerly():
      return
    spec = head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=ModeKeys.PREDICT,
        logits=logits,
        trainable_variables=[
            variables.Variable([1.0, 2.0], dtype=dtypes.float32)])
    self.assertItemsEqual(
        (test_lib._DEFAULT_SERVING_KEY, 'predict', 'classification'),
        spec.export_outputs.keys())
    # Assert predictions and export_outputs.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      predictions = sess.run(spec.predictions)
      self.assertAllClose(logits,
                          predictions[prediction_keys.PredictionKeys.LOGITS])
      self.assertAllClose(
          expected_probabilities,
          predictions[prediction_keys.PredictionKeys.PROBABILITIES])
      self.assertAllClose(
          expected_probabilities,
          sess.run(spec.export_outputs[test_lib._DEFAULT_SERVING_KEY].scores))
      self.assertAllEqual(
          expected_export_classes,
          sess.run(spec.export_outputs[test_lib._DEFAULT_SERVING_KEY].classes))

  def test_weight_should_not_impact_prediction(self):
    n_classes = 4
    head = head_lib.MultiLabelHead(n_classes, weight_column='example_weights')
    self.assertEqual(n_classes, head.logits_dimension)

    logits = np.array(
        [[0., 1., 2., -1.], [-1., -2., -3., 1.]], dtype=np.float32)
    expected_probabilities = nn.sigmoid(logits)
    weights_2x1 = [[1.], [2.]]
    features = {
        'x': np.array(((42,),), dtype=np.int32),
        'example_weights': weights_2x1
    }

    keys = prediction_keys.PredictionKeys
    preds = head.predictions(logits, [keys.LOGITS, keys.PROBABILITIES])
    self.assertAllClose(logits, self.evaluate(preds[keys.LOGITS]))
    self.assertAllClose(expected_probabilities,
                        self.evaluate(preds[keys.PROBABILITIES]))
    if context.executing_eagerly():
      return

    spec = head.create_estimator_spec(
        features=features,
        mode=ModeKeys.PREDICT,
        logits=logits,
        trainable_variables=[
            variables.Variable([1.0, 2.0], dtype=dtypes.float32)])
    # Assert predictions and export_outputs.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      predictions = sess.run(spec.predictions)
      self.assertAllClose(logits,
                          predictions[prediction_keys.PredictionKeys.LOGITS])
      self.assertAllClose(
          expected_probabilities,
          predictions[prediction_keys.PredictionKeys.PROBABILITIES])

  def test_eval_create_loss(self):
    """Tests head.loss for eval mode."""
    n_classes = 2
    head = head_lib.MultiLabelHead(n_classes)

    logits = np.array([[-1., 1.], [-1.5, 1.]], dtype=np.float32)
    labels = np.array([[1, 0], [1, 1]], dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # loss = (labels * -log(sigmoid(logits)) +
    #         (1 - labels) * -log(1 - sigmoid(logits))) / 2
    expected_training_loss = 0.5 * np.sum(
        _sigmoid_cross_entropy(labels=labels, logits=logits))
    actual_training_loss = head.loss(
        logits=logits,
        labels=labels,
        features=features,
        mode=ModeKeys.EVAL)
    self.assertAllClose(expected_training_loss,
                        self.evaluate(actual_training_loss))

  def test_eval_create_loss_large_logits(self):
    """Tests head.loss for eval mode and large logits."""
    n_classes = 2
    head = head_lib.MultiLabelHead(n_classes)

    logits = np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)
    labels = np.array([[1, 0], [1, 1]], dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # loss = labels * -log(sigmoid(logits)) +
    #        (1 - labels) * -log(1 - sigmoid(logits))
    # For large logits, this is approximated as:
    # loss = labels * (logits < 0) * (-logits) +
    #        (1 - labels) * (logits > 0) * logits
    expected_training_loss = 0.5 * np.sum(
        np.array([[(10. + 10.) / 2.], [(15. + 0.) / 2.]], dtype=np.float32))
    actual_training_loss = head.loss(
        logits=logits,
        labels=labels,
        features=features,
        mode=ModeKeys.EVAL)
    self.assertAllClose(expected_training_loss,
                        self.evaluate(actual_training_loss), atol=1e-4)

  def test_eval_create_loss_labels_wrong_shape(self):
    """Tests head.loss for eval mode when labels has the wrong shape."""
    n_classes = 2
    head = head_lib.MultiLabelHead(n_classes)

    logits = np.array([[-1., 1.], [-1.5, 1.]], dtype=np.float32)
    labels_2x1 = np.array([[1], [1]], dtype=np.int64)
    labels_2 = np.array([1, 1], dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}

    if context.executing_eagerly():
      with self.assertRaisesRegexp(ValueError, 'Expected labels dimension=2'):
        head.loss(logits=logits, labels=labels_2x1, features=features,
                  mode=ModeKeys.EVAL)
      with self.assertRaisesRegexp(ValueError, 'Expected labels dimension=2'):
        head.loss(logits=logits, labels=labels_2, features=features,
                  mode=ModeKeys.EVAL)
    else:
      labels_placeholder = array_ops.placeholder(dtype=dtypes.int64)
      actual_training_loss = head.loss(
          logits=logits, labels=labels_placeholder, features=features,
          mode=ModeKeys.EVAL)
      with self.cached_session():
        test_lib._initialize_variables(self, monitored_session.Scaffold())
        with self.assertRaisesRegexp(
            errors.InvalidArgumentError,
            r'\[expected_labels_shape: \] \[2 2\] \[labels_shape: \] \[2 1\]'):
          actual_training_loss.eval({
              labels_placeholder: labels_2x1
          })
        with self.assertRaisesRegexp(
            errors.InvalidArgumentError,
            r'labels shape must be \[D0, D1, ... DN, 2\]\..*'
            r'\[Received shape: \] \[2\]'):
          actual_training_loss.eval({
              labels_placeholder: labels_2
          })

  def test_eval_create_loss_loss_fn(self):
    """Tests head.loss for eval mode and custom loss_fn."""
    loss = np.array([[1.], [2.]], dtype=np.float32)
    logits_input = np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)
    labels_input = np.array([[1, 0], [1, 1]], dtype=np.int64)
    def _loss_fn(labels, logits):
      check_labels = control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(labels, labels_input)),
          data=[labels])
      check_logits = control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(logits, logits_input)),
          data=[logits])
      with ops.control_dependencies([check_labels, check_logits]):
        return constant_op.constant(loss)
    head = head_lib.MultiLabelHead(n_classes=2, loss_fn=_loss_fn)

    actual_training_loss = head.loss(
        logits=logits_input,
        labels=labels_input,
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=ModeKeys.EVAL)
    self.assertAllClose(np.sum(loss) / 2., self.evaluate(actual_training_loss))

  def test_eval_create_loss_loss_fn_wrong_shape(self):
    """Tests custom loss_fn that returns Tensor of unexpected shape."""
    loss = np.array([1., 2.], dtype=np.float32)
    def _loss_fn(labels, logits):
      del labels, logits  # Unused
      return constant_op.constant(loss)
    head = head_lib.MultiLabelHead(n_classes=2, loss_fn=_loss_fn)

    logits = np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)
    labels = np.array([[1, 0], [1, 1]], dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}

    if context.executing_eagerly():
      with self.assertRaisesRegexp(
          ValueError,
          'loss_shape'):
        head.loss(logits=logits, labels=labels, features=features,
                  mode=ModeKeys.EVAL)
    else:
      actual_training_loss = head.loss(
          logits=logits, labels=labels, features=features,
          mode=ModeKeys.EVAL)
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[loss_fn must return Tensor of shape \[D0, D1, ... DN, 1\]\. \] '
          r'\[logits_shape: \] \[2 2\] \[loss_shape: \] \[2\]'):
        self.evaluate(actual_training_loss)

  def test_eval_labels_none(self):
    """Tests that error is raised when labels is None."""
    head = head_lib.MultiLabelHead(n_classes=2)
    with self.assertRaisesRegexp(
        ValueError, r'You must provide a labels Tensor\. Given: None\.'):
      head.loss(
          logits=np.array([[-10., 10.], [-15., 10.]], dtype=np.float32),
          labels=None,
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=ModeKeys.EVAL)

  def _test_eval(
      self, head, logits, labels, expected_loss, expected_metrics,
      features=None, regularization_losses=None):
    tol = 1e-3
    if context.executing_eagerly():
      loss = head.loss(
          labels, logits, features=features or {}, mode=ModeKeys.EVAL,
          regularization_losses=regularization_losses)
      self.assertIsNotNone(loss)
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)

      eval_metrics = head.metrics(regularization_losses=regularization_losses)
      updated_metrics = head.update_metrics(
          eval_metrics, features or {}, logits, labels,
          regularization_losses=regularization_losses)
      self.assertItemsEqual(expected_metrics.keys(), updated_metrics.keys())
      self.assertAllClose(
          expected_metrics,
          {k: updated_metrics[k].result() for k in updated_metrics},
          rtol=tol,
          atol=tol)
      return

    spec = head.create_estimator_spec(
        features=features or {},
        mode=ModeKeys.EVAL,
        logits=logits,
        labels=labels,
        regularization_losses=regularization_losses,
        trainable_variables=[
            variables.Variable([1.0, 2.0], dtype=dtypes.float32)])
    # Assert spec contains expected tensors.
    self.assertIsNotNone(spec.loss)
    self.assertItemsEqual(expected_metrics.keys(), spec.eval_metric_ops.keys())
    self.assertIsNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    test_lib._assert_no_hooks(self, spec)
    # Assert predictions, loss, and metrics.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, _ = sess.run((spec.loss, update_ops))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      # Check results of value ops (in `metrics`).
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval() for k in value_ops},
          rtol=tol,
          atol=tol)

  def test_eval(self):
    n_classes = 2
    head = head_lib.MultiLabelHead(n_classes)
    logits = np.array([[-1., 1.], [-1.5, 1.5]], dtype=np.float32)
    labels = np.array([[1, 0], [1, 1]], dtype=np.int64)
    # loss = labels * -log(sigmoid(logits)) +
    #        (1 - labels) * -log(1 - sigmoid(logits))
    # Sum over examples, divide by batch_size.
    expected_loss = 0.5 * np.sum(
        _sigmoid_cross_entropy(labels=labels, logits=logits))
    keys = metric_keys.MetricKeys
    expected_metrics = {
        # Average loss over examples.
        keys.LOSS_MEAN: expected_loss,
        # auc and auc_pr cannot be reliably calculated for only 4 samples, but
        # this assert tests that the algorithm remains consistent.
        keys.AUC: 0.3333,
        keys.AUC_PR: 0.7689,
    }
    self._test_eval(
        head=head,
        logits=logits,
        labels=labels,
        expected_loss=expected_loss,
        expected_metrics=expected_metrics)

  def test_eval_sparse_labels(self):
    n_classes = 2
    head = head_lib.MultiLabelHead(n_classes)
    logits = np.array([[-1., 1.], [-1.5, 1.5]], dtype=np.float32)
    # Equivalent to multi_hot = [[1, 0], [1, 1]]
    labels = sparse_tensor.SparseTensor(
        values=[0, 0, 1],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    labels_multi_hot = np.array([[1, 0], [1, 1]], dtype=np.int64)
    # loss = labels * -log(sigmoid(logits)) +
    #        (1 - labels) * -log(1 - sigmoid(logits))
    # Sum over examples, divide by batch_size.
    expected_loss = 0.5 * np.sum(
        _sigmoid_cross_entropy(labels=labels_multi_hot, logits=logits))
    keys = metric_keys.MetricKeys
    expected_metrics = {
        # Average loss over examples.
        keys.LOSS_MEAN: expected_loss,
        # auc and auc_pr cannot be reliably calculated for only 4 samples, but
        # this assert tests that the algorithm remains consistent.
        keys.AUC: 0.3333,
        keys.AUC_PR: 0.7689,
    }
    self._test_eval(
        head=head,
        logits=logits,
        labels=labels,
        expected_loss=expected_loss,
        expected_metrics=expected_metrics)

  def test_eval_with_regularization_losses(self):
    n_classes = 2
    head = head_lib.MultiLabelHead(n_classes)
    logits = np.array([[-1., 1.], [-1.5, 1.5]], dtype=np.float32)
    labels = np.array([[1, 0], [1, 1]], dtype=np.int64)
    regularization_losses = [1.5, 0.5]
    expected_regularization_loss = 2.
    # unregularized_loss = sum(
    #     labels * -log(sigmoid(logits)) +
    #     (1 - labels) * -log(1 - sigmoid(logits))) / batch_size
    expected_unregularized_loss = np.sum(
        _sigmoid_cross_entropy(labels=labels, logits=logits)) / 2.
    expected_regularized_loss = (
        expected_unregularized_loss + expected_regularization_loss)
    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: expected_unregularized_loss,
        keys.LOSS_REGULARIZATION: expected_regularization_loss,
        # auc and auc_pr cannot be reliably calculated for only 4 samples, but
        # this assert tests that the algorithm remains consistent.
        keys.AUC: 0.3333,
        keys.AUC_PR: 0.7689,
    }
    self._test_eval(
        head=head,
        logits=logits,
        labels=labels,
        expected_loss=expected_regularized_loss,
        expected_metrics=expected_metrics,
        regularization_losses=regularization_losses)

  def test_eval_with_label_vocabulary(self):
    n_classes = 2
    head = head_lib.MultiLabelHead(
        n_classes, label_vocabulary=['class0', 'class1'])
    logits = np.array([[-1., 1.], [-1.5, 1.5]], dtype=np.float32)
    # Equivalent to multi_hot = [[1, 0], [1, 1]]
    labels = sparse_tensor.SparseTensor(
        values=['class0', 'class0', 'class1'],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    labels_multi_hot = np.array([[1, 0], [1, 1]], dtype=np.int64)
    # loss = labels * -log(sigmoid(logits)) +
    #        (1 - labels) * -log(1 - sigmoid(logits))
    # Sum over examples, divide by batch_size.
    expected_loss = 0.5 * np.sum(
        _sigmoid_cross_entropy(labels=labels_multi_hot, logits=logits))
    keys = metric_keys.MetricKeys
    expected_metrics = {
        # Average loss over examples.
        keys.LOSS_MEAN: expected_loss,
        # auc and auc_pr cannot be reliably calculated for only 4 samples, but
        # this assert tests that the algorithm remains consistent.
        keys.AUC: 0.3333,
        keys.AUC_PR: 0.7689,
    }
    self._test_eval(
        head=head,
        logits=logits,
        labels=labels,
        expected_loss=expected_loss,
        expected_metrics=expected_metrics)

  def test_eval_with_label_vocabulary_with_multi_hot_input(self):
    n_classes = 2
    head = head_lib.MultiLabelHead(
        n_classes, label_vocabulary=['class0', 'class1'])
    logits = np.array([[-1., 1.], [-1.5, 1.5]], dtype=np.float32)
    labels_multi_hot = np.array([[1, 0], [1, 1]], dtype=np.int64)
    # loss = labels * -log(sigmoid(logits)) +
    #        (1 - labels) * -log(1 - sigmoid(logits))
    # Sum over examples, divide by batch_size.
    expected_loss = 0.5 * np.sum(
        _sigmoid_cross_entropy(labels=labels_multi_hot, logits=logits))
    keys = metric_keys.MetricKeys
    expected_metrics = {
        # Average loss over examples.
        keys.LOSS_MEAN: expected_loss,
        # auc and auc_pr cannot be reliably calculated for only 4 samples, but
        # this assert tests that the algorithm remains consistent.
        keys.AUC: 0.3333,
        keys.AUC_PR: 0.7689,
    }
    self._test_eval(
        head=head,
        logits=logits,
        labels=labels_multi_hot,
        expected_loss=expected_loss,
        expected_metrics=expected_metrics)

  def test_eval_with_thresholds(self):
    n_classes = 2
    thresholds = [0.25, 0.5, 0.75]
    head = head_lib.MultiLabelHead(n_classes, thresholds=thresholds)

    logits = np.array([[-1., 1.], [-1.5, 1.5]], dtype=np.float32)
    labels = np.array([[1, 0], [1, 1]], dtype=np.int64)
    # loss = labels * -log(sigmoid(logits)) +
    #        (1 - labels) * -log(1 - sigmoid(logits))
    # Sum over examples, divide by batch_size.
    expected_loss = 0.5 * np.sum(
        _sigmoid_cross_entropy(labels=labels, logits=logits))

    keys = metric_keys.MetricKeys
    expected_metrics = {
        # Average loss over examples.
        keys.LOSS_MEAN: expected_loss,
        # auc and auc_pr cannot be reliably calculated for only 4 samples, but
        # this assert tests that the algorithm remains consistent.
        keys.AUC: 0.3333,
        keys.AUC_PR: 0.7689,
        keys.ACCURACY_AT_THRESHOLD % thresholds[0]: 2. / 4.,
        keys.PRECISION_AT_THRESHOLD % thresholds[0]: 2. / 3.,
        keys.RECALL_AT_THRESHOLD % thresholds[0]: 2. / 3.,
        keys.ACCURACY_AT_THRESHOLD % thresholds[1]: 1. / 4.,
        keys.PRECISION_AT_THRESHOLD % thresholds[1]: 1. / 2.,
        keys.RECALL_AT_THRESHOLD % thresholds[1]: 1. / 3.,
        keys.ACCURACY_AT_THRESHOLD % thresholds[2]: 2. / 4.,
        keys.PRECISION_AT_THRESHOLD % thresholds[2]: 1. / 1.,
        keys.RECALL_AT_THRESHOLD % thresholds[2]: 1. / 3.,
    }

    self._test_eval(
        head=head,
        logits=logits,
        labels=labels,
        expected_loss=expected_loss,
        expected_metrics=expected_metrics)

  def test_eval_with_classes_for_class_based_metrics(self):
    head = head_lib.MultiLabelHead(
        n_classes=2, classes_for_class_based_metrics=[0, 1])

    logits = np.array([[-1., 1.], [-1.5, 1.5]], dtype=np.float32)
    labels = np.array([[1, 0], [1, 1]], dtype=np.int64)
    # loss = labels * -log(sigmoid(logits)) +
    #        (1 - labels) * -log(1 - sigmoid(logits))
    # Sum over examples, divide by batch_size.
    expected_loss = 0.5 * np.sum(
        _sigmoid_cross_entropy(labels=labels, logits=logits))

    keys = metric_keys.MetricKeys
    expected_metrics = {
        # Average loss over examples.
        keys.LOSS_MEAN: expected_loss,
        # auc and auc_pr cannot be reliably calculated for only 4 samples, but
        # this assert tests that the algorithm remains consistent.
        keys.AUC: 0.3333,
        keys.AUC_PR: 0.7689,
        keys.PROBABILITY_MEAN_AT_CLASS % 0:
            math_ops.reduce_sum(nn.sigmoid(logits[:, 0])) / 2.,
        keys.AUC_AT_CLASS % 0: 0.,
        keys.AUC_PR_AT_CLASS % 0: 1.,
        keys.PROBABILITY_MEAN_AT_CLASS % 1:
            math_ops.reduce_sum(nn.sigmoid(logits[:, 1])) / 2.,
        keys.AUC_AT_CLASS % 1: 1.,
        keys.AUC_PR_AT_CLASS % 1: 1.,
    }

    self._test_eval(
        head=head,
        logits=logits,
        labels=labels,
        expected_loss=expected_loss,
        expected_metrics=expected_metrics)

  def test_eval_with_classes_for_class_based_metrics_string(self):
    head = head_lib.MultiLabelHead(
        n_classes=2, label_vocabulary=['a', 'b'],
        classes_for_class_based_metrics=['a', 'b'])

    logits = np.array([[-1., 1.], [-1.5, 1.5]], dtype=np.float32)
    labels = sparse_tensor.SparseTensor(
        values=['a', 'a', 'b'],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    labels_onehot = np.array([[1, 0], [1, 1]], dtype=np.int64)
    # loss = labels * -log(sigmoid(logits)) +
    #        (1 - labels) * -log(1 - sigmoid(logits))
    # Sum over examples, divide by batch_size.
    expected_loss = 0.5 * np.sum(
        _sigmoid_cross_entropy(labels=labels_onehot, logits=logits))

    keys = metric_keys.MetricKeys
    expected_metrics = {
        # Average loss over examples.
        keys.LOSS_MEAN: expected_loss,
        # auc and auc_pr cannot be reliably calculated for only 4 samples, but
        # this assert tests that the algorithm remains consistent.
        keys.AUC: 0.3333,
        keys.AUC_PR: 0.7689,
        keys.PROBABILITY_MEAN_AT_NAME % 'a':
            math_ops.reduce_sum(nn.sigmoid(logits[:, 0])) / 2.,
        keys.AUC_AT_NAME % 'a': 0.,
        keys.AUC_PR_AT_NAME % 'a': 1.,
        keys.PROBABILITY_MEAN_AT_NAME % 'b':
            math_ops.reduce_sum(nn.sigmoid(logits[:, 1])) / 2.,
        keys.AUC_AT_NAME % 'b': 1.,
        keys.AUC_PR_AT_NAME % 'b': 1.,
    }

    self._test_eval(
        head=head,
        logits=logits,
        labels=labels,
        expected_loss=expected_loss,
        expected_metrics=expected_metrics)

  def test_eval_with_weights(self):
    n_classes = 2
    head = head_lib.MultiLabelHead(n_classes, weight_column='example_weights')

    logits = np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)
    labels = np.array([[1, 0], [1, 1]], dtype=np.int64)
    features = {
        'x': np.array([[41], [42]], dtype=np.int32),
        'example_weights': np.array([[1.], [2.]], dtype=np.float32),
    }
    # For large logits, sigmoid cross entropy loss is approximated as:
    # loss = labels * (logits < 0) * (-logits) +
    #        (1 - labels) * (logits > 0) * logits =>
    # expected_unweighted_loss = [[10., 10.], [15., 0.]]
    # Average over classes, weighted sum over examples, divide by batch_size.
    # loss = (1 * (10 + 10) / 2 + 2 * (15 + 0) / 2) / 2
    expected_loss = 12.5
    keys = metric_keys.MetricKeys
    expected_metrics = {
        # Average loss over weighted examples (denominator is sum(weights)).
        keys.LOSS_MEAN: expected_loss * (2. / 3.),
        # auc and auc_pr cannot be reliably calculated for only 4 samples, but
        # this assert tests that the algorithm remains consistent.
        keys.AUC: 0.2000,
        keys.AUC_PR: 0.7280,
    }
    self._test_eval(
        head=head,
        logits=logits,
        labels=labels,
        expected_loss=expected_loss,
        expected_metrics=expected_metrics,
        features=features)

  def test_train_create_loss_large_logits(self):
    """Tests head.create_loss for train mode and large logits."""
    n_classes = 2
    head = head_lib.MultiLabelHead(n_classes, weight_column='example_weights')

    logits = np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)
    labels = np.array([[1, 0], [1, 1]], dtype=np.int64)
    weights = np.array([[1.], [2.]], dtype=np.float32)
    features = {
        'x': np.array(((42,),), dtype=np.int32),
        'example_weights': weights
    }
    # loss = labels * -log(sigmoid(logits)) +
    #        (1 - labels) * -log(1 - sigmoid(logits))
    # For large logits, this is approximated as:
    # loss = labels * (logits < 0) * (-logits) +
    #        (1 - labels) * (logits > 0) * logits
    # expected_unreduced_loss = [[(10. + 10.) / 2.], [(15. + 0.) / 2.]]
    # expected_weights = [[1.], [2.]]
    expected_training_loss = (1. * (10. + 10.) / 2. + 2. * (15. + 0.) / 2.) / 2.
    training_loss = head.loss(
        logits=logits,
        labels=labels,
        features=features,
        mode=ModeKeys.TRAIN)
    self.assertAllClose(
        expected_training_loss, self.evaluate(training_loss), atol=1e-4)

  def test_train_create_loss_loss_reduction(self):
    """Tests head.create_loss with loss_reduction."""
    n_classes = 2
    head = head_lib.MultiLabelHead(
        n_classes,
        weight_column='example_weights',
        loss_reduction=losses_utils.ReductionV2.SUM)

    logits = np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)
    labels = np.array([[1, 0], [1, 1]], dtype=np.int64)
    weights = np.array([[1.], [2.]], dtype=np.float32)
    # loss = labels * -log(sigmoid(logits)) +
    #        (1 - labels) * -log(1 - sigmoid(logits))
    # For large logits, this is approximated as:
    # loss = labels * (logits < 0) * (-logits) +
    #        (1 - labels) * (logits > 0) * logits
    # expected_unreduced_loss = [[(10. + 10.) / 2.], [(15. + 0.) / 2.]]
    # expected_weights = [[1.], [2.]]
    expected_training_loss = (1. * (10. + 10.) + 2. * (15. + 0.)) / 2.
    training_loss = head.loss(
        logits=logits,
        labels=labels,
        features={
            'x': np.array(((42,),), dtype=np.int32),
            'example_weights': weights
        },
        mode=ModeKeys.TRAIN)
    self.assertAllClose(
        expected_training_loss, self.evaluate(training_loss), atol=1e-4)

  def test_train_labels_none(self):
    """Tests that error is raised when labels is None."""
    head = head_lib.MultiLabelHead(n_classes=2)

    with self.assertRaisesRegexp(
        ValueError, r'You must provide a labels Tensor\. Given: None\.'):
      head.loss(
          logits=np.array([[-10., 10.], [-15., 10.]], dtype=np.float32),
          labels=None,
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=ModeKeys.TRAIN)

  def test_train_invalid_indicator_labels(self):
    head = head_lib.MultiLabelHead(n_classes=2)
    logits = np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)
    # The value 2 is outside the allowed range.
    labels = np.array([[2, 0], [1, 1]], dtype=np.int64)
    if context.executing_eagerly():
      with self.assertRaisesRegexp(
          ValueError,
          r'labels must be an integer indicator Tensor with values in '
          r'\[0, 1\]'):
        head.loss(
            logits=logits,
            labels=labels,
            features={},
            mode=ModeKeys.TRAIN)
      return

    def _train_op_fn(loss):
      del loss
      return control_flow_ops.no_op()
    spec = head.create_estimator_spec(
        features={},
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn,
        trainable_variables=[
            variables.Variable([1.0, 2.0], dtype=dtypes.float32)])
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'labels must be an integer indicator Tensor with values in '
          r'\[0, 1\]'):
        sess.run(spec.loss)

  def test_train_invalid_sparse_labels(self):
    head = head_lib.MultiLabelHead(n_classes=2)
    logits = np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)
    # The value 2 is outside the allowed range.
    labels = sparse_tensor.SparseTensor(
        values=[2, 0, 1],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    if context.executing_eagerly():
      with self.assertRaisesRegexp(
          ValueError,
          r'labels must be an integer SparseTensor with values in \[0, 2\)'):
        head.loss(
            logits=logits,
            labels=labels,
            features={},
            mode=ModeKeys.TRAIN)
      return

    def _train_op_fn(loss):
      del loss
      return control_flow_ops.no_op()
    spec = head.create_estimator_spec(
        features={},
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn,
        trainable_variables=[
            variables.Variable([1.0, 2.0], dtype=dtypes.float32)])
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'labels must be an integer SparseTensor with values in \[0, 2\)'):
        sess.run(spec.loss)

  def _test_train(self, head, logits, labels, expected_loss):
    tol = 1e-3
    features = {'x': np.array(((42,),), dtype=np.int32)}
    if context.executing_eagerly():
      loss = head.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=ModeKeys.TRAIN)
      self.assertIsNotNone(loss)
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      return

    expected_train_result = 'my_train_op'
    def _train_op_fn(loss):
      return string_ops.string_join(
          [constant_op.constant(expected_train_result),
           string_ops.as_string(loss, precision=3)])
    spec = head.create_estimator_spec(
        features=features,
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn,
        trainable_variables=[
            variables.Variable([1.0, 2.0], dtype=dtypes.float32)])
    self.assertIsNotNone(spec.loss)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    test_lib._assert_no_hooks(self, spec)
    # Assert predictions, loss, train_op, and summaries.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str = sess.run((spec.loss, spec.train_op,
                                                  spec.scaffold.summary_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.3f}'.format(expected_train_result, expected_loss)),
          train_result)
      test_lib._assert_simple_summaries(
          self, {metric_keys.MetricKeys.LOSS: expected_loss}, summary_str, tol)

  def test_train(self):
    head = head_lib.MultiLabelHead(n_classes=2)
    logits = np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)
    labels = np.array([[1, 0], [1, 1]], dtype=np.int64)
    # For large logits, sigmoid cross entropy loss is approximated as:
    # loss = labels * (logits < 0) * (-logits) +
    #        (1 - labels) * (logits > 0) * logits =>
    # expected_unweighted_loss = [[10., 10.], [15., 0.]]
    # Average over classes, sum over examples, divide by batch_size.
    # loss = ((10 + 10) / 2 + (15 + 0) / 2 ) / 2
    expected_loss = 8.75
    self._test_train(
        head=head, logits=logits, labels=labels, expected_loss=expected_loss)

  def test_train_sparse_labels(self):
    head = head_lib.MultiLabelHead(n_classes=2)
    logits = np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)
    # Equivalent to multi_hot = [[1, 0], [1, 1]]
    labels = sparse_tensor.SparseTensor(
        values=[0, 0, 1],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    # For large logits, sigmoid cross entropy loss is approximated as:
    # loss = labels * (logits < 0) * (-logits) +
    #        (1 - labels) * (logits > 0) * logits =>
    # expected_unweighted_loss = [[10., 10.], [15., 0.]]
    # Average over classes, sum over examples, divide by batch_size.
    # loss = ((10 + 10) / 2 + (15 + 0) / 2 ) / 2
    expected_loss = 8.75
    self._test_train(
        head=head, logits=logits, labels=labels, expected_loss=expected_loss)

  def test_train_with_label_vocabulary(self):
    head = head_lib.MultiLabelHead(
        n_classes=2, label_vocabulary=['class0', 'class1'])
    logits = np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)
    # Equivalent to multi_hot = [[1, 0], [1, 1]]
    labels = sparse_tensor.SparseTensor(
        values=['class0', 'class0', 'class1'],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    # For large logits, sigmoid cross entropy loss is approximated as:
    # loss = labels * (logits < 0) * (-logits) +
    #        (1 - labels) * (logits > 0) * logits =>
    # expected_unweighted_loss = [[10., 10.], [15., 0.]]
    # Average over classes, sum over examples, divide by batch_size.
    # loss = ((10 + 10) / 2 + (15 + 0) / 2 ) / 2
    expected_loss = 8.75
    self._test_train(
        head=head, logits=logits, labels=labels, expected_loss=expected_loss)

  def test_train_with_regularization_losses(self):
    head = head_lib.MultiLabelHead(n_classes=2)
    logits = np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)
    labels = np.array([[1, 0], [1, 1]], dtype=np.int64)
    regularization_losses = [1.5, 0.5]
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # For large logits, sigmoid cross entropy loss is approximated as:
    # loss = labels * (logits < 0) * (-logits) +
    #        (1 - labels) * (logits > 0) * logits =>
    # expected_unweighted_loss = [[10., 10.], [15., 0.]]
    # Average over classes and over batch and add regularization loss.
    expected_loss = 35. / 4. + 2.
    expected_summaries = {
        metric_keys.MetricKeys.LOSS: expected_loss,
        metric_keys.MetricKeys.LOSS_REGULARIZATION: 2.,
    }
    tol = 1e-3
    loss = head.loss(
        logits=logits,
        labels=labels,
        features=features,
        mode=ModeKeys.TRAIN,
        regularization_losses=regularization_losses)
    self.assertIsNotNone(loss)
    self.assertAllClose(expected_loss, self.evaluate(loss), rtol=tol, atol=tol)
    if context.executing_eagerly():
      return

    expected_train_result = 'my_train_op'
    def _train_op_fn(loss):
      return string_ops.string_join(
          [constant_op.constant(expected_train_result),
           string_ops.as_string(loss, precision=3)])
    spec = head.create_estimator_spec(
        features=features,
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn,
        regularization_losses=regularization_losses,
        trainable_variables=[
            variables.Variable([1.0, 2.0], dtype=dtypes.float32)])
    # Assert predictions, loss, train_op, and summaries.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str = sess.run((spec.loss, spec.train_op,
                                                  spec.scaffold.summary_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.3f}'.format(expected_train_result, expected_loss)),
          train_result)
      test_lib._assert_simple_summaries(
          self, expected_summaries, summary_str, tol)

  def test_train_with_weights(self):
    n_classes = 2
    head = head_lib.MultiLabelHead(n_classes, weight_column='example_weights')

    logits = np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)
    labels = np.array([[1, 0], [1, 1]], dtype=np.int64)
    features = {
        'x': np.array([[41], [42]], dtype=np.int32),
        'example_weights': np.array([[1.], [2.]], dtype=np.float32),
    }
    # For large logits, sigmoid cross entropy loss is approximated as:
    # loss = labels * (logits < 0) * (-logits) +
    #        (1 - labels) * (logits > 0) * logits =>
    # expected_unweighted_loss = [[10., 10.], [15., 0.]]
    # Average over classes, weighted sum over examples, divide by batch_size.
    # loss = (1 * (10 + 10) / 2 + 2 * (15 + 0) / 2) / 2
    expected_loss = 12.5
    tol = 1e-3

    loss = head.loss(
        logits=logits,
        labels=labels,
        features=features,
        mode=ModeKeys.TRAIN)
    self.assertIsNotNone(loss)
    self.assertAllClose(expected_loss, self.evaluate(loss), rtol=tol, atol=tol)
    if context.executing_eagerly():
      return

    expected_train_result = 'my_train_op'
    def _train_op_fn(loss):
      return string_ops.string_join(
          [constant_op.constant(expected_train_result),
           string_ops.as_string(loss, precision=3)])

    spec = head.create_estimator_spec(
        features=features,
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn,
        trainable_variables=[
            variables.Variable([1.0, 2.0], dtype=dtypes.float32)])
    self.assertIsNotNone(spec.loss)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    test_lib._assert_no_hooks(self, spec)
    # Assert predictions, loss, train_op, and summaries.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str = sess.run((spec.loss, spec.train_op,
                                                  spec.scaffold.summary_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.3f}'.format(expected_train_result, expected_loss)),
          train_result)
      test_lib._assert_simple_summaries(
          self, {metric_keys.MetricKeys.LOSS: expected_loss,}, summary_str, tol)

  def test_multi_dim_weighted_train_create_loss(self):
    """Logits and labels of shape [2, 2, 3], weights [2, 2]."""
    head = head_lib.MultiLabelHead(n_classes=3, weight_column='weights')

    logits = np.array([[[-10., 10., -10.], [10., -10., 10.]],
                       [[-12., 12., -12.], [12., -12., 12.]]], dtype=np.float32)
    labels = np.array([[[1, 0, 0], [1, 0, 0]],
                       [[0, 1, 1], [0, 1, 1]]], dtype=np.int64)
    weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)
    # unreduced_loss =
    #     [[10 + 10 + 0, 0 + 0 + 10], [0 + 0 + 12, 12 + 12 + 0]] / 3
    #   = [[20/3, 10/3], [4, 8]]
    # expected_unreduced_loss = [[[20./3.], [10./3.]], [[4.], [8.]]]
    # weights are reshaped to [2, 2, 1] to match logits.
    # expected_weights = [[[1.], [1.5]], [[2.], [2.5]]]
    # loss = (1*20/3 + 1.5*10/3 + 2*4 + 2.5*8) / 4 = 9.9167
    expected_training_loss = 9.9167
    training_loss = head.loss(
        logits=logits,
        labels=labels,
        features={'weights': weights},
        mode=ModeKeys.TRAIN)
    atol = 1.e-3
    self.assertAllClose(
        expected_training_loss, self.evaluate(training_loss), atol=atol)

  def test_multi_dim_weighted_train(self):
    """Logits and labels of shape [2, 2, 3], weights [2, 2]."""
    head = head_lib.MultiLabelHead(n_classes=3, weight_column='weights')

    logits = np.array([[[-10., 10., -10.], [10., -10., 10.]],
                       [[-12., 12., -12.], [12., -12., 12.]]], dtype=np.float32)
    labels = np.array([[[1, 0, 0], [1, 0, 0]],
                       [[0, 1, 1], [0, 1, 1]]], dtype=np.int64)
    weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)
    # loss = [[10 + 10 + 0, 0 + 0 + 10], [0 + 0 + 12, 12 + 12 + 0]] / 3
    #      = [[20/3, 10/3], [4, 8]]
    # loss = (1*20/3 + 1.5*10/3 + 2*4 + 2.5*8) / 4 = 9.9167
    expected_loss = 9.9167
    atol = 1.e-3

    loss = head.loss(
        logits=logits,
        labels=labels,
        features={'weights': weights},
        mode=ModeKeys.TRAIN)
    self.assertIsNotNone(loss)
    self.assertAllClose(expected_loss, self.evaluate(loss), atol=atol)
    if context.executing_eagerly():
      return

    expected_train_result = 'my_train_op'
    def _train_op_fn(loss):
      return string_ops.string_join(
          [constant_op.constant(expected_train_result),
           string_ops.as_string(loss, precision=3)])

    spec = head.create_estimator_spec(
        features={'weights': weights},
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn,
        trainable_variables=[
            variables.Variable([1.0, 2.0], dtype=dtypes.float32)])
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, monitored_session.Scaffold())
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAllClose(expected_loss, loss, atol=atol)
      self.assertEqual(
          six.b('{0:s}{1:.3f}'.format(expected_train_result, expected_loss)),
          train_result)

  def test_multi_dim_weights_wrong_inner_dim(self):
    """Logits and labels of shape [2, 2, 3], weights [2, 1]."""
    head = head_lib.MultiLabelHead(n_classes=3, weight_column='weights')

    logits = np.array([[[-10., 10., -10.], [10., -10., 10.]],
                       [[-12., 12., -12.], [12., -12., 12.]]], dtype=np.float32)
    labels = np.array([[[1, 0, 0], [1, 0, 0]],
                       [[0, 1, 1], [0, 1, 1]]], dtype=np.int64)
    weights = np.array([[1.], [2.]], dtype=np.float32)

    if context.executing_eagerly():
      with self.assertRaisesRegexp(ValueError, 'weights shape'):
        head.loss(
            logits=logits,
            labels=labels,
            features={'weights': weights},
            mode=ModeKeys.TRAIN)
      return

    def _train_op_fn(loss):
      del loss
      return control_flow_ops.no_op()

    spec = head.create_estimator_spec(
        features={'weights': weights},
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn,
        trainable_variables=[
            variables.Variable([1.0, 2.0], dtype=dtypes.float32)])
    with self.cached_session():
      test_lib._initialize_variables(self, monitored_session.Scaffold())
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[logits_shape: \] \[2 2 3\] \[weights_shape: \] \[2 1\]'):
        spec.loss.eval()

  def test_multi_dim_weights_wrong_outer_dim(self):
    """Logits and labels of shape [2, 2, 3], weights [2, 2, 3]."""
    head = head_lib.MultiLabelHead(n_classes=3, weight_column='weights')

    logits = np.array([[[-10., 10., -10.], [10., -10., 10.]],
                       [[-12., 12., -12.], [12., -12., 12.]]], dtype=np.float32)
    labels = np.array([[[1, 0, 0], [1, 0, 0]],
                       [[0, 1, 1], [0, 1, 1]]], dtype=np.int64)
    weights = np.array([[[1., 1., 1.], [1.5, 1.5, 1.5]],
                        [[2., 2., 2.], [2.5, 2.5, 2.5]]], dtype=np.float32)

    if context.executing_eagerly():
      with self.assertRaisesRegexp(ValueError, 'weights shape'):
        head.loss(
            logits=logits,
            labels=labels,
            features={'weights': weights},
            mode=ModeKeys.TRAIN)
      return

    weights_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    def _train_op_fn(loss):
      del loss
      return control_flow_ops.no_op()
    spec = head.create_estimator_spec(
        features={'weights': weights_placeholder},
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn,
        trainable_variables=[
            variables.Variable([1.0, 2.0], dtype=dtypes.float32)])
    with self.cached_session():
      test_lib._initialize_variables(self, monitored_session.Scaffold())
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[logits_shape: \] \[2 2 3\] \[weights_shape: \] \[2 2 3\]'):
        spec.loss.eval({weights_placeholder: weights})

  def test_multi_dim_weighted_eval(self):
    """Logits and labels of shape [2, 2, 3], weights [2, 2]."""
    head = head_lib.MultiLabelHead(n_classes=3, weight_column='weights')

    logits = np.array([[[-10., 10., -10.], [10., -10., 10.]],
                       [[-12., 12., -12.], [12., -12., 12.]]], dtype=np.float32)
    labels = np.array([[[1, 0, 0], [1, 0, 0]],
                       [[0, 1, 1], [0, 1, 1]]], dtype=np.int64)
    weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)
    # loss = [[10 + 10 + 0, 0 + 0 + 10], [0 + 0 + 12, 12 + 12 + 0]] / 3
    #      = [[20/3, 10/3], [4, 8]]
    # loss = (1*20/3 + 1.5*10/3 + 2*4 + 2.5*8) / 4 = 9.9167
    expected_loss = 9.9167
    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: expected_loss * (4. / np.sum(weights)),
        # auc and auc_pr cannot be reliably calculated for only 4 samples, but
        # this assert tests that the algorithm remains consistent.
        keys.AUC: 0.4977,
        keys.AUC_PR: 0.5461,
    }
    self._test_eval(
        head=head,
        features={'weights': weights},
        logits=logits,
        labels=labels,
        expected_loss=expected_loss,
        expected_metrics=expected_metrics)


@test_util.deprecated_graph_mode_only
class MultiLabelHeadForEstimator(test.TestCase):
  """Tests for create_estimator_spec running in Graph mode only."""

  def test_invalid_trainable_variables(self):
    head = head_lib.MultiLabelHead(n_classes=2)

    class _Optimizer(optimizer_v2.OptimizerV2):

      def get_updates(self, loss, params):
        del params
        return [string_ops.string_join([
            constant_op.constant('my_train_op'),
            string_ops.as_string(loss, precision=2)
        ])]

      def get_config(self):
        config = super(_Optimizer, self).get_config()
        return config

    with self.assertRaisesRegexp(
        ValueError, r'trainable_variables cannot be None'):
      head.create_estimator_spec(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=ModeKeys.TRAIN,
          logits=np.array([[-10., 10.], [-15., 10.]], dtype=np.float32),
          labels=np.array([[1, 0], [1, 1]], dtype=np.int64),
          optimizer=_Optimizer('my_optimizer'),
          trainable_variables=None)
    with self.assertRaisesRegexp(
        ValueError, r'trainable_variables should be a list or a tuple'):
      head.create_estimator_spec(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=ModeKeys.TRAIN,
          logits=np.array([[-10., 10.], [-15., 10.]], dtype=np.float32),
          labels=np.array([[1, 0], [1, 1]], dtype=np.int64),
          optimizer=_Optimizer('my_optimizer'),
          trainable_variables={
              'var_list': [
                  variables.Variable([1.0, 2.0], dtype=dtypes.float32)]})

  def test_train_with_optimizer(self):
    head = head_lib.MultiLabelHead(n_classes=2)
    logits = np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)
    labels = np.array([[1, 0], [1, 1]], dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # For large logits, sigmoid cross entropy loss is approximated as:
    # loss = labels * (logits < 0) * (-logits) +
    #        (1 - labels) * (logits > 0) * logits =>
    # expected_unweighted_loss = [[10., 10.], [15., 0.]]
    # Average over classes, sum over examples, divide by batch_size.
    # loss = ((10 + 10) / 2 + (15 + 0) / 2 ) / 2
    expected_loss = 8.75
    expected_train_result = 'my_train_op'

    class _Optimizer(optimizer_v2.OptimizerV2):

      def get_updates(self, loss, params):
        del params
        return [string_ops.string_join(
            [constant_op.constant(expected_train_result),
             string_ops.as_string(loss, precision=3)])]

      def get_config(self):
        config = super(_Optimizer, self).get_config()
        return config

    spec = head.create_estimator_spec(
        features=features,
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        optimizer=_Optimizer('my_optimizer'),
        trainable_variables=[
            variables.Variable([1.0, 2.0], dtype=dtypes.float32)])

    tol = 1e-3
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.3f}'.format(expected_train_result, expected_loss)),
          train_result)

  def test_predict_with_label_vocabulary(self):
    n_classes = 4
    head = head_lib.MultiLabelHead(
        n_classes, label_vocabulary=['foo', 'bar', 'foobar', 'barfoo'])

    logits = np.array(
        [[0., 1., 2., -1.], [-1., -2., -3., 1.]], dtype=np.float32)
    expected_export_classes = [[b'foo', b'bar', b'foobar', b'barfoo']] * 2

    spec = head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=ModeKeys.PREDICT,
        logits=logits,
        trainable_variables=[
            variables.Variable([1.0, 2.0], dtype=dtypes.float32)])

    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertAllEqual(
          expected_export_classes,
          sess.run(spec.export_outputs[test_lib._DEFAULT_SERVING_KEY].classes))

  def test_train_with_update_ops(self):
    with ops.Graph().as_default():
      w = variables.Variable(1)
      update_op = w.assign_add(1)

      t = variables.Variable('')
      expected_train_result = b'my_train_op'
      def _train_op_fn(loss):
        del loss
        return t.assign(expected_train_result)

      head = head_lib.MultiLabelHead(n_classes=2)
      spec = head.create_estimator_spec(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=ModeKeys.TRAIN,
          logits=np.array([[-10., 10.], [-15., 10.]], dtype=np.float32),
          labels=np.array([[1, 0], [1, 1]], dtype=np.int64),
          train_op_fn=_train_op_fn,
          update_ops=[update_op],
          trainable_variables=[
              variables.Variable([1.0, 2.0], dtype=dtypes.float32)])

      with self.cached_session() as sess:
        test_lib._initialize_variables(self, spec.scaffold)
        sess.run(spec.train_op)
        w_value, t_value = sess.run([w, t])
        self.assertEqual(2, w_value)
        self.assertEqual(expected_train_result, t_value)

  def test_lookup_tables_in_graph(self):
    n_classes = 2
    head = head_lib.MultiLabelHead(
        n_classes=n_classes, label_vocabulary=['class0', 'class1'])

    feature_columns = [feature_column.numeric_column('x')]
    # Create dnn estimator.
    est = dnn.DNNEstimatorV2(
        head=head,
        hidden_units=(2, 2),
        feature_columns=feature_columns)

    def input_fn():
      return (
          {'x': np.array(((42,), (43,),), dtype=np.int32)},
          np.array([[1, 0], [1, 1]], dtype=np.int64))

    # Train.
    num_steps = 1
    est.train(input_fn, steps=num_steps)
    # Eval.
    eval_results = est.evaluate(input_fn, steps=num_steps)
    self.assertEqual(num_steps, eval_results[ops.GraphKeys.GLOBAL_STEP])
    self.assertIn('loss', six.iterkeys(eval_results))
    # Predict.
    est.predict(input_fn)


if __name__ == '__main__':
  test.main()
