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
"""Tests for multi_class_head.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column_lib as feature_column
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import monitored_session
from tensorflow_estimator.python.estimator.canned import dnn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.head import head_utils as test_lib
from tensorflow_estimator.python.estimator.head import multi_class_head as head_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys


class MultiClassHead(test.TestCase):

  def test_n_classes_is_none(self):
    with self.assertRaisesRegexp(ValueError, 'n_classes must be > 2'):
      head_lib.MultiClassHead(n_classes=None)

  def test_n_classes_is_2(self):
    with self.assertRaisesRegexp(ValueError, 'n_classes must be > 2'):
      head_lib.MultiClassHead(n_classes=2)

  def test_invalid_loss_reduction(self):
    with self.assertRaisesRegexp(
        ValueError, r'Invalid loss_reduction: invalid_loss_reduction'):
      head_lib.MultiClassHead(
          n_classes=3, loss_reduction='invalid_loss_reduction')
    with self.assertRaisesRegexp(ValueError, r'Invalid loss_reduction: none'):
      head_lib.MultiClassHead(
          n_classes=3, loss_reduction=losses_utils.ReductionV2.NONE)

  def test_loss_fn_arg_labels_missing(self):

    def _loss_fn(logits):
      del logits  # Unused

    with self.assertRaisesRegexp(
        ValueError, r'loss_fn must contain argument: labels\. '
        r'Given arguments: \(\'logits\',\)'):
      head_lib.MultiClassHead(n_classes=3, loss_fn=_loss_fn)

  def test_loss_fn_arg_logits_missing(self):

    def _loss_fn(labels):
      del labels  # unused

    with self.assertRaisesRegexp(
        ValueError, r'loss_fn must contain argument: logits\. '
        r'Given arguments: \(\'labels\',\)'):
      head_lib.MultiClassHead(n_classes=3, loss_fn=_loss_fn)

  def test_loss_fn_arg_features_ok(self):

    def _loss_fn(labels, logits, features):
      del labels, logits, features  # Unused

    head_lib.MultiClassHead(n_classes=3, loss_fn=_loss_fn)

  def test_loss_fn_arg_invalid(self):

    def _loss_fn(labels, logits, name=None):
      del labels, logits, name  # Unused

    with self.assertRaisesRegexp(ValueError,
                                 r'loss_fn has unexpected args: \[\'name\'\]'):
      head_lib.MultiClassHead(n_classes=3, loss_fn=_loss_fn)

  def test_invalid_logits_shape(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes)
    self.assertEqual(n_classes, head.logits_dimension)

    # Logits should be shape (batch_size, 3).
    logits_2x2 = np.array((
        (45., 44.),
        (41., 42.),
    ))
    pred_key = prediction_keys.PredictionKeys.PROBABILITIES
    # Static shape.
    with self.assertRaisesRegexp(ValueError, 'logits shape'):
      preds = head.predictions(logits_2x2, [pred_key])
      self.evaluate(preds[pred_key])
    if context.executing_eagerly():
      return

    # Dynamic shape only works in Graph mode.
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    spec = head.create_estimator_spec(
        features={'x': np.array((
            (30.,),
            (42.,),
        ))},
        mode=ModeKeys.PREDICT,
        logits=logits_placeholder)
    with self.cached_session():
      with self.assertRaisesRegexp(errors.OpError, 'logits shape'):
        spec.predictions[pred_key].eval({logits_placeholder: logits_2x2})

  def test_invalid_labels_shape(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes)
    self.assertEqual(n_classes, head.logits_dimension)

    # Logits should be shape (batch_size, 3).
    # Labels should be shape (batch_size, 1).
    labels_2x2 = np.array((
        (1, 2),
        (0, 1),
    ), dtype=np.int)
    logits_2x3 = np.array((
        (1., 2., 3.),
        (1., 2., 3.),
    ))
    features = {'x': np.array(((42.,),))}

    # Static shape.
    with self.assertRaisesRegexp(ValueError, 'Mismatched label shape'):
      training_loss = head.loss(
          logits=logits_2x3,
          labels=labels_2x2,
          features=features,
          mode=ModeKeys.EVAL)
      self.evaluate(training_loss)
    if context.executing_eagerly():
      return

    # Dynamic shape only works in Graph mode.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.int64)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    training_loss = head.loss(
        logits=logits_placeholder,
        labels=labels_placeholder,
        features=features,
        mode=ModeKeys.EVAL)
    with self.cached_session():
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[expected_labels_shape: \] \[2 1\] \[labels_shape: \] \[2 2\]'):
        training_loss.eval({
            logits_placeholder: logits_2x3,
            labels_placeholder: labels_2x2
        })

  def test_invalid_labels_type(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes)
    self.assertEqual(n_classes, head.logits_dimension)

    # Logits should be shape (batch_size, 3).
    # Labels should be shape (batch_size, 1).
    labels_2x1 = np.array((
        (1.,),
        (1.,),
    ))
    logits_2x3 = np.array((
        (1., 2., 3.),
        (1., 2., 3.),
    ))
    features = {'x': np.array(((42.,),))}

    # Static shape.
    with self.assertRaisesRegexp(ValueError, 'Labels dtype'):
      head.loss(
          logits=logits_2x3,
          labels=labels_2x1,
          features=features,
          mode=ModeKeys.EVAL)
    if context.executing_eagerly():
      return

    # Dynamic shape only works in Graph mode.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    with self.assertRaisesRegexp(ValueError, 'Labels dtype'):
      head.loss(
          logits=logits_placeholder,
          labels=labels_placeholder,
          features=features,
          mode=ModeKeys.EVAL)

  def test_invalid_labels_values(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes)
    self.assertEqual(n_classes, head.logits_dimension)

    labels_2x1_with_large_id = np.array((
        (45,),
        (1,),
    ), dtype=np.int)
    labels_2x1_with_negative_id = np.array((
        (-5,),
        (1,),
    ), dtype=np.int)
    logits_2x3 = np.array((
        (1., 2., 4.),
        (1., 2., 3.),
    ))
    features = {'x': np.array(((42.,),))}

    if context.executing_eagerly():
      with self.assertRaisesRegexp(ValueError, 'Labels must be <= 3 - 1'):
        training_loss = head.loss(
            logits=logits_2x3,
            labels=labels_2x1_with_large_id,
            features=features,
            mode=ModeKeys.EVAL)

      with self.assertRaisesRegexp(ValueError, 'Labels must be >= 0'):
        training_loss = head.loss(
            logits=logits_2x3,
            labels=labels_2x1_with_negative_id,
            features=features,
            mode=ModeKeys.EVAL)
      return

    # Dynamic shape only works in Graph mode.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.int64)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    training_loss = head.loss(
        logits=logits_placeholder,
        labels=labels_placeholder,
        features=features,
        mode=ModeKeys.EVAL)
    with self.cached_session():
      with self.assertRaisesOpError('Labels must be <= n_classes - 1'):
        training_loss.eval({
            labels_placeholder: labels_2x1_with_large_id,
            logits_placeholder: logits_2x3
        })

    with self.cached_session():
      with self.assertRaisesOpError('Labels must be >= 0'):
        training_loss.eval({
            labels_placeholder: labels_2x1_with_negative_id,
            logits_placeholder: logits_2x3
        })

  def test_invalid_labels_sparse_tensor(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes)
    self.assertEqual(n_classes, head.logits_dimension)

    labels_2x1 = sparse_tensor.SparseTensor(
        values=['english', 'italian'],
        indices=[[0, 0], [1, 0]],
        dense_shape=[2, 1])
    logits_2x3 = np.array((
        (1., 2., 4.),
        (1., 2., 3.),
    ))

    with self.assertRaisesRegexp(ValueError,
                                 'SparseTensor labels are not supported.'):
      loss = head.loss(
          logits=logits_2x3,
          labels=labels_2x1,
          features={'x': np.array(((42.,),))},
          mode=ModeKeys.EVAL)
      self.evaluate(loss)

  def test_incompatible_labels_shape(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes)
    self.assertEqual(n_classes, head.logits_dimension)

    # Logits should be shape (batch_size, 3).
    # Labels should be shape (batch_size, 1).
    # Here batch sizes are different.
    values_3x1 = np.array((
        (1,),
        (1,),
        (1,),
    ))
    values_2x3 = np.array((
        (1., 2., 3.),
        (1., 2., 3.),
    ))
    features = {'x': values_2x3}

    # Static shape.
    # Eager mode.
    if context.executing_eagerly():
      with self.assertRaisesRegexp(ValueError, 'labels shape'):
        head.loss(
            logits=values_2x3,
            labels=values_3x1,
            features=features,
            mode=ModeKeys.EVAL)
      return
    # Graph mode.
    with self.assertRaisesRegexp(
        ValueError,
        r'Shape mismatch: The shape of labels \(received \(3,\)\) should equal '
        r'the shape of logits except for the last dimension '
        r'\(received \(2, 3\)\)\.'):
      head.loss(
          logits=values_2x3,
          labels=values_3x1,
          features=features,
          mode=ModeKeys.EVAL)

    # Dynamic shape only works in Graph mode.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.int64)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    training_loss = head.loss(
        logits=logits_placeholder,
        labels=labels_placeholder,
        features=features,
        mode=ModeKeys.EVAL)
    with self.cached_session():
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[expected_labels_shape: \] \[2 1\] \[labels_shape: \] \[3 1\]'):
        training_loss.eval({
            labels_placeholder: values_3x1,
            logits_placeholder: values_2x3
        })

  def test_predict(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes)
    self.assertEqual(n_classes, head.logits_dimension)

    logits = [[1., 0., 0.], [0., 0., 1.]]
    expected_probabilities = [[0.576117, 0.2119416, 0.2119416],
                              [0.2119416, 0.2119416, 0.576117]]
    expected_class_ids = [[0], [2]]
    expected_classes = [[b'0'], [b'2']]
    expected_export_classes = [[b'0', b'1', b'2']] * 2

    keys = prediction_keys.PredictionKeys
    preds = head.predictions(logits)
    self.assertAllClose(logits, self.evaluate(preds[keys.LOGITS]))
    self.assertAllClose(expected_probabilities,
                        self.evaluate(preds[keys.PROBABILITIES]))
    self.assertAllClose(expected_class_ids,
                        self.evaluate(preds[keys.CLASS_IDS]))
    self.assertAllEqual(expected_classes, self.evaluate(preds[keys.CLASSES]))
    if context.executing_eagerly():
      return

    spec = head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=ModeKeys.PREDICT,
        logits=logits)

    self.assertItemsEqual(
        (test_lib._DEFAULT_SERVING_KEY, 'predict', 'classification'),
        spec.export_outputs.keys())

    # Assert predictions and export_outputs.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      predictions = sess.run(spec.predictions)
      self.assertAllClose(logits, predictions[keys.LOGITS])
      self.assertAllClose(expected_probabilities,
                          predictions[keys.PROBABILITIES])
      self.assertAllClose(expected_class_ids, predictions[keys.CLASS_IDS])
      self.assertAllEqual(expected_classes, predictions[keys.CLASSES])

      self.assertAllClose(
          expected_probabilities,
          sess.run(spec.export_outputs[test_lib._DEFAULT_SERVING_KEY].scores))
      self.assertAllEqual(
          expected_export_classes,
          sess.run(spec.export_outputs[test_lib._DEFAULT_SERVING_KEY].classes))

  def test_predict_with_invalid_keys(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes)
    self.assertEqual(n_classes, head.logits_dimension)

    logits = [[1., 0., 0.], [0., 0., 1.]]
    with self.assertRaisesRegexp(
        ValueError,
        r'Prediction key must be in PredictionKeys, given: some_invalid_key'):
      preds = head.predictions(logits, ['some_invalid_key'])
      self.evaluate(preds)

  def test_predict_with_vocabulary_list(self):
    n_classes = 3
    head = head_lib.MultiClassHead(
        n_classes, label_vocabulary=['aang', 'iroh', 'zuko'])

    logits = [[1., 0., 0.], [0., 0., 1.]]
    expected_classes = [[b'aang'], [b'zuko']]
    expected_export_classes = [[b'aang', b'iroh', b'zuko']] * 2
    pred_key = prediction_keys.PredictionKeys.CLASSES
    if context.executing_eagerly():
      preds = head.predictions(logits, [pred_key])
      self.assertAllEqual(expected_classes,
                          preds[prediction_keys.PredictionKeys.CLASSES])
      return

    spec = head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=ModeKeys.PREDICT,
        logits=logits)

    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertAllEqual(expected_classes,
                          sess.run(spec.predictions[pred_key]))
      self.assertAllEqual(
          expected_export_classes,
          sess.run(spec.export_outputs[test_lib._DEFAULT_SERVING_KEY].classes))

  def test_weight_should_not_impact_prediction(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes, weight_column='label_weights')
    logits = [[1., 0., 0.], [0., 0., 1.]]
    expected_probabilities = [[0.576117, 0.2119416, 0.2119416],
                              [0.2119416, 0.2119416, 0.576117]]
    weights_2x1 = [[1.], [2.]]
    features = {
        'x': np.array(((42,),), dtype=np.int32),
        'label_weights': weights_2x1,
    }

    keys = prediction_keys.PredictionKeys
    preds = head.predictions(logits, [keys.LOGITS, keys.PROBABILITIES])
    self.assertAllClose(logits, self.evaluate(preds[keys.LOGITS]))
    self.assertAllClose(expected_probabilities,
                        self.evaluate(preds[keys.PROBABILITIES]))
    if context.executing_eagerly():
      return

    spec = head.create_estimator_spec(
        features=features, mode=ModeKeys.PREDICT, logits=logits)

    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      predictions = sess.run(spec.predictions)
      self.assertAllClose(logits, predictions[keys.LOGITS])
      self.assertAllClose(expected_probabilities,
                          predictions[keys.PROBABILITIES])

  def test_eval_create_loss(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes)

    # logits: [2, 3], labels: [2, 1]
    logits = np.array((
        (10, 0, 0),
        (0, 10, 0),
    ), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # loss = sum(cross_entropy(labels, logits)) / batch_size = 10 / 2 = 5.
    expected_training_loss = 5.
    # Create loss.
    training_loss = head.loss(
        logits=logits,
        labels=labels,
        features=features,
        mode=ModeKeys.EVAL)
    self.assertAllClose(
        expected_training_loss,
        self.evaluate(training_loss),
        rtol=1e-2,
        atol=1e-2)

  def test_eval_create_loss_loss_fn(self):
    """Tests head.loss for eval mode and custom loss_fn."""
    loss = np.array([[1.], [2.]], dtype=np.float32)
    logits_input = np.array([[-10., 10., 0.], [-15., 10., 0]], dtype=np.float32)
    labels_input = np.array([[1], [2]], dtype=np.int64)

    def _loss_fn(labels, logits):
      check_labels = control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(labels, labels_input)),
          data=[labels])
      check_logits = control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(logits, logits_input)),
          data=[logits])
      with ops.control_dependencies([check_labels, check_logits]):
        return constant_op.constant(loss)

    head = head_lib.MultiClassHead(n_classes=3, loss_fn=_loss_fn)

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

    head = head_lib.MultiClassHead(n_classes=3, loss_fn=_loss_fn)

    logits = np.array([[-10., 10., 0.], [-15., 10., 0.]], dtype=np.float32)
    labels = np.array([[1], [2]], dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    if context.executing_eagerly():
      with self.assertRaisesRegexp(ValueError, 'loss_shape'):
        head.loss(logits=logits, labels=labels, features=features)
    else:
      actual_training_loss = head.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=ModeKeys.EVAL)
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[loss_fn must return Tensor of shape \[D0, D1, ... DN, 1\]\. \] '
          r'\[logits_shape: \] \[2 3\] \[loss_shape: \] \[2\]'):
        self.evaluate(actual_training_loss)

  def test_eval_labels_none(self):
    """Tests that error is raised when labels is None."""
    head = head_lib.MultiClassHead(n_classes=3)

    with self.assertRaisesRegexp(
        ValueError, r'You must provide a labels Tensor\. Given: None\.'):
      head.loss(
          logits=np.array((
              (10, 0, 0),
              (0, 10, 0),
          ), dtype=np.float32),
          labels=None,
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=ModeKeys.EVAL)

  def test_eval(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes)
    logits = np.array((
        (10, 0, 0),
        (0, 10, 0),
    ), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # loss = sum(cross_entropy(labels, logits)) / batch_size
    #      = sum(10, 0) / 2 = 5.
    expected_loss = 5.
    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: expected_loss,
        keys.ACCURACY: 0.5,  # 1 of 2 labels is correct.
    }
    tol = 1e-2

    if context.executing_eagerly():
      eval_metrics = head.metrics()
      updated_metrics = head.update_metrics(eval_metrics, features, logits,
                                            labels)
      self.assertItemsEqual(expected_metrics.keys(), updated_metrics.keys())
      self.assertAllClose(
          expected_metrics,
          {k: updated_metrics[k].result() for k in updated_metrics},
          rtol=tol,
          atol=tol)
      loss = head.loss(
          logits, labels, features=features, mode=ModeKeys.EVAL)
      self.assertIsNotNone(loss)
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      return

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=ModeKeys.EVAL,
        logits=logits,
        labels=labels)

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

  def test_eval_metric_ops_with_head_name(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes, name='some_multiclass_head')
    logits = np.array((
        (10, 0, 0),
        (0, 10, 0),
    ), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    expected_metric_keys = [
        '{}/some_multiclass_head'.format(metric_keys.MetricKeys.LOSS_MEAN),
        '{}/some_multiclass_head'.format(metric_keys.MetricKeys.ACCURACY)
    ]

    eval_metrics = head.metrics()
    updated_metrics = head.update_metrics(eval_metrics, features, logits,
                                          labels)
    self.assertItemsEqual(expected_metric_keys, updated_metrics.keys())

  def test_eval_with_regularization_losses(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes)
    logits = np.array((
        (10, 0, 0),
        (0, 10, 0),
    ), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    regularization_losses = [1.5, 0.5]
    expected_regularization_loss = 2.
    # unregularized_loss = sum(cross_entropy(labels, logits)) / batch_size
    #                    = sum(10, 0) / 2 = 5.
    expected_unregularized_loss = 5.
    expected_regularized_loss = (
        expected_unregularized_loss + expected_regularization_loss)

    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: expected_unregularized_loss,
        keys.LOSS_REGULARIZATION: expected_regularization_loss,
        keys.ACCURACY: 0.5,  # 1 of 2 labels is correct.
    }
    tol = 1e-2
    if context.executing_eagerly():
      eval_metrics = head.metrics(regularization_losses=regularization_losses)
      updated_metrics = head.update_metrics(
          eval_metrics,
          features,
          logits,
          labels,
          regularization_losses=regularization_losses)
      # Assert metrics.
      self.assertAllClose(
          expected_metrics,
          {k: updated_metrics[k].result() for k in updated_metrics},
          rtol=tol,
          atol=tol)
      return

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=ModeKeys.EVAL,
        logits=logits,
        labels=labels,
        regularization_losses=regularization_losses)

    # Assert predictions, loss, and metrics.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, _ = sess.run((spec.loss, update_ops))
      self.assertAllClose(expected_regularized_loss, loss, rtol=tol, atol=tol)
      # Check results of value ops (in `metrics`).
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval() for k in value_ops},
          rtol=tol,
          atol=tol)

  def test_eval_with_label_vocabulary_create_loss(self):
    n_classes = 3
    head = head_lib.MultiClassHead(
        n_classes,
        label_vocabulary=['aang', 'iroh', 'zuko'])
    logits = [[10., 0, 0], [0, 10, 0]]
    labels = [[b'iroh'], [b'iroh']]
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # loss = sum(cross_entropy(labels, logits)) / batch_size = [5.0, 0].
    expected_training_loss = 5.
    if context.executing_eagerly():
      training_loss = head.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=ModeKeys.EVAL)
      self.assertAllClose(
          expected_training_loss, training_loss, rtol=1e-2, atol=1e-2)
    else:
      training_loss = head.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=ModeKeys.EVAL)
      with self.cached_session():
        test_lib._initialize_variables(self, monitored_session.Scaffold())
        self.assertAllClose(
            expected_training_loss, training_loss.eval(), rtol=1e-2, atol=1e-2)

  def test_eval_with_label_vocabulary(self):
    n_classes = 3
    head = head_lib.MultiClassHead(
        n_classes,
        label_vocabulary=['aang', 'iroh', 'zuko'])

    logits = [[10., 0, 0], [0, 10, 0]]
    labels = [[b'iroh'], [b'iroh']]
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # loss = sum(cross_entropy(labels, logits))  / batch_size
    #      = sum(10, 0) / 2 = 5.
    expected_loss = 5.
    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: expected_loss,
        keys.ACCURACY: 0.5,  # 1 of 2 labels is correct.
    }
    tol = 1e-2
    if context.executing_eagerly():
      loss = head.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=ModeKeys.EVAL)
      self.assertAllClose(
          expected_loss, self.evaluate(loss), rtol=tol, atol=tol)
      eval_metrics = head.metrics()
      updated_metrics = head.update_metrics(eval_metrics, features, logits,
                                            labels)
      self.assertAllClose(
          expected_metrics,
          {k: updated_metrics[k].result() for k in updated_metrics},
          rtol=tol,
          atol=tol)
      return

    spec = head.create_estimator_spec(
        features=features,
        mode=ModeKeys.EVAL,
        logits=logits,
        labels=labels)

    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, _ = sess.run((spec.loss, update_ops))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      # Check results of value ops (in `metrics`).
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval() for k in value_ops},
          rtol=tol,
          atol=tol)

  def test_weighted_multi_example_eval(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes, weight_column='label_weights')

    # Create estimator spec.
    logits = np.array((
        (10, 0, 0),
        (0, 10, 0),
        (0, 0, 10),
    ), dtype=np.float32)
    labels = np.array(((1,), (2,), (2,)), dtype=np.int64)
    weights_3x1 = np.array(((1.,), (2.,), (3.,)), dtype=np.float64)
    # weighted_loss = sum(cross_entropy(labels, logits) *  weights)
    #      = sum([10, 10, 0] * [1, 2, 3])
    #      = sum([10, 20, 0]) = 30.
    # loss = weighted_loss  / batch_size = 30 / 3 = 10
    # loss_mean = weighted_loss / sum(weights) = 30 / 6 = 5
    expected_loss = 10.
    features = {
        'x': np.array(((42,),), dtype=np.int32),
        'label_weights': weights_3x1
    }
    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: 30. / np.sum(weights_3x1),
        # Weighted accuracy is 1 * 3.0 / sum weights = 0.5
        keys.ACCURACY: 0.5,
    }

    tol = 1e-2
    if context.executing_eagerly():
      loss = head.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=ModeKeys.EVAL)
      self.assertIsNotNone(loss)
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      eval_metrics = head.metrics()
      updated_metrics = head.update_metrics(eval_metrics, features, logits,
                                            labels)
      self.assertItemsEqual(expected_metrics.keys(), updated_metrics.keys())
      self.assertAllClose(
          expected_metrics,
          {k: updated_metrics[k].result() for k in updated_metrics},
          rtol=tol,
          atol=tol)
      return

    spec = head.create_estimator_spec(
        features=features,
        mode=ModeKeys.EVAL,
        logits=logits,
        labels=labels)
    # Assert spec contains expected tensors.
    self.assertIsNotNone(spec.loss)
    self.assertItemsEqual(expected_metrics.keys(), spec.eval_metric_ops.keys())
    self.assertIsNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    test_lib._assert_no_hooks(self, spec)

    # Assert loss, and metrics.
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

  def test_train_create_loss(self):
    head = head_lib.MultiClassHead(n_classes=3)

    logits = np.array((
        (10, 0, 0),
        (0, 10, 0),
    ), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # unreduced_loss = cross_entropy(labels, logits) = [10, 0].
    expected_unreduced_loss = [[10.], [0.]]
    # Weights default to 1.
    expected_weights = 1.
    # training_loss = (1 * 10 + 1 * 0) / 2 = 5.
    expected_training_loss = 5.
    tol = 1e-2
    if context.executing_eagerly():
      training_loss = head.loss(logits, labels, features)
      self.assertAllClose(
          expected_training_loss, training_loss, rtol=tol, atol=tol)
      unreduced_loss, actual_weights = head._unweighted_loss_and_weights(
          logits, labels, features)
      self.assertAllClose(
          expected_unreduced_loss, unreduced_loss, rtol=tol, atol=tol)
      self.assertAllClose(expected_weights, actual_weights)
      return

    training_loss = head.loss(logits, labels, features)
    unreduced_loss, actual_weights = head._unweighted_loss_and_weights(
        logits, labels, features)
    with self.cached_session():
      test_lib._initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)
      self.assertAllClose(
          expected_unreduced_loss, unreduced_loss.eval(), rtol=tol, atol=tol)
      self.assertAllClose(expected_weights, actual_weights)

  def test_train_create_loss_loss_reduction(self):
    """Tests create_loss with loss_reduction."""
    head = head_lib.MultiClassHead(
        n_classes=3, loss_reduction=losses_utils.ReductionV2.SUM)

    logits = np.array((
        (10, 0, 0),
        (0, 10, 0),
    ), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}

    # unreduced_loss = cross_entropy(labels, logits) = [10, 0].
    expected_unreduced_loss = [[10.], [0.]]
    # Weights default to 1.
    expected_weights = 1.
    # training_loss = 1 * 10 + 1 * 0
    expected_training_loss = 10.
    tol = 1e-2
    if context.executing_eagerly():
      training_loss = head.loss(logits, labels, features)
      self.assertAllClose(
          expected_training_loss, training_loss, rtol=tol, atol=tol)
      unreduced_loss, actual_weights = head._unweighted_loss_and_weights(
          logits, labels, features)
      self.assertAllClose(
          expected_unreduced_loss, unreduced_loss, rtol=tol, atol=tol)
      self.assertAllClose(expected_weights, actual_weights)
      return

    training_loss = head.loss(logits, labels, features)
    unreduced_loss, actual_weights = head._unweighted_loss_and_weights(
        logits, labels, features)
    with self.cached_session():
      test_lib._initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)
      self.assertAllClose(
          expected_unreduced_loss, unreduced_loss.eval(), rtol=tol, atol=tol)
      self.assertAllClose(expected_weights, actual_weights)

  def test_train_labels_none(self):
    """Tests that error is raised when labels is None."""
    head = head_lib.MultiClassHead(n_classes=3)

    with self.assertRaisesRegexp(
        ValueError, r'You must provide a labels Tensor\. Given: None\.'):
      head.loss(
          logits=np.array((
              (10, 0, 0),
              (0, 10, 0),
          ), dtype=np.float32),
          labels=None,
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=ModeKeys.TRAIN)

  def test_train(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes)

    logits = np.array((
        (10, 0, 0),
        (0, 10, 0),
    ), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}

    # loss = sum(cross_entropy(labels, logits)) / batch_size
    #      = sum(10, 0) / 2 = 5.
    expected_loss = 5.
    tol = 1e-2
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
      return string_ops.string_join([
          constant_op.constant(expected_train_result),
          string_ops.as_string(loss, precision=2)
      ])

    spec = head.create_estimator_spec(
        features=features,
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)

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
          six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)),
          train_result)
      test_lib._assert_simple_summaries(
          self, {
              metric_keys.MetricKeys.LOSS: expected_loss,
          }, summary_str, tol)

  def test_train_with_regularization_losses(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes)

    logits = np.array((
        (10, 0, 0),
        (0, 10, 0),
    ), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}

    regularization_losses = [1.5, 0.5]
    expected_regularization_loss = 2.
    # unregularized_loss = sum(cross_entropy(labels, logits)) / batch_size
    #                    = sum(10, 0) / 2 = 5.
    # loss = unregularized_loss + regularization_loss = 7.
    expected_loss = 7.
    tol = 1e-2
    if context.executing_eagerly():
      loss = head.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=ModeKeys.TRAIN,
          regularization_losses=regularization_losses)
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      return

    expected_train_result = 'my_train_op'

    def _train_op_fn(loss):
      return string_ops.string_join([
          constant_op.constant(expected_train_result),
          string_ops.as_string(loss, precision=2)
      ])

    spec = head.create_estimator_spec(
        features=features,
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn,
        regularization_losses=regularization_losses)

    # Assert predictions, loss, train_op, and summaries.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str = sess.run((spec.loss, spec.train_op,
                                                  spec.scaffold.summary_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)),
          train_result)
      test_lib._assert_simple_summaries(
          self, {
              metric_keys.MetricKeys.LOSS:
                  expected_loss,
              metric_keys.MetricKeys.LOSS_REGULARIZATION: (
                  expected_regularization_loss),
          }, summary_str, tol)

  def test_train_one_dim_create_loss(self):
    """Tests create_loss with 1D labels and weights (shape [batch_size])."""
    head = head_lib.MultiClassHead(
        n_classes=3, weight_column='label_weights')

    logits = np.array((
        (10, 0, 0),
        (0, 10, 0),
        (0, 0, 10),
    ), dtype=np.float32)
    labels_rank_1 = np.array((
        1,
        2,
        2,
    ), dtype=np.int64)
    weights_rank_1 = np.array((
        1.,
        2.,
        3.,
    ), dtype=np.float64)
    features = {
        'x': np.array(((42,),), dtype=np.float32),
        'label_weights': weights_rank_1
    }

    # unreduced_loss = cross_entropy(labels, logits) = [10, 10, 0].
    # weights are reshaped to [3, 1] to match logits.
    # training_loss = sum(1 * 10 + 2 * 10 + 3 * 0) / batch_size = 30. / 3 = 10.
    expected_training_loss = 10.
    tol = 1e-2

    if context.executing_eagerly():
      training_loss = head.loss(logits, labels_rank_1, features)
      self.assertAllClose(
          expected_training_loss, training_loss, rtol=tol, atol=tol)
      return

    training_loss = head.loss(logits, labels_rank_1, features)
    with self.cached_session():
      test_lib._initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)

  def test_train_one_dim(self):
    """Tests train with 1D labels and weights (shape [batch_size])."""
    head = head_lib.MultiClassHead(
        n_classes=3, weight_column='label_weights')

    logits = np.array((
        (10, 0, 0),
        (0, 10, 0),
        (0, 0, 10),
    ), dtype=np.float32)
    labels_rank_1 = np.array((
        1,
        2,
        2,
    ), dtype=np.int64)
    weights_rank_1 = np.array((
        1.,
        2.,
        3.,
    ), dtype=np.float64)

    self.assertEqual((3,), labels_rank_1.shape)
    self.assertEqual((3,), weights_rank_1.shape)
    # loss = sum(cross_entropy(labels, logits) * [1, 2, 3]) / batch_size
    #      = sum([10, 10, 0] * [1, 2, 3]) / 3 = 30 / 3 = 10.
    expected_loss = 10.
    features = {
        'x': np.array(((42,),), dtype=np.float32),
        'label_weights': weights_rank_1
    }
    tol = 1e-2
    if context.executing_eagerly():
      loss = head.loss(
          logits=logits,
          labels=labels_rank_1,
          features=features,
          mode=ModeKeys.TRAIN)
      self.assertIsNotNone(loss)
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      return

    expected_train_result = 'my_train_op'

    def _train_op_fn(loss):
      return string_ops.string_join([
          constant_op.constant(expected_train_result),
          string_ops.as_string(loss, precision=2)
      ])

    spec = head.create_estimator_spec(
        features=features,
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels_rank_1,
        train_op_fn=_train_op_fn)

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
          six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)),
          train_result)
      test_lib._assert_simple_summaries(
          self, {
              metric_keys.MetricKeys.LOSS:
                  expected_loss,
          }, summary_str, tol)

  def test_train_with_vocabulary_create_loss(self):
    n_classes = 3
    head = head_lib.MultiClassHead(
        n_classes, label_vocabulary=['aang', 'iroh', 'zuko'])

    logits = [[10., 0, 0], [0, 10, 0]]
    labels = [[b'iroh'], [b'iroh']]
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # loss = sum(cross_entropy(labels, logits)) / batch_size = 10 / 2 = 5.
    expected_training_loss = 5.
    if context.executing_eagerly():
      training_loss = head.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=ModeKeys.TRAIN)
      self.assertAllClose(
          expected_training_loss, training_loss, rtol=1e-2, atol=1e-2)
      return

    training_loss = head.loss(
        logits=logits,
        labels=labels,
        features=features,
        mode=ModeKeys.TRAIN)
    with self.cached_session():
      test_lib._initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=1e-2, atol=1e-2)

  def test_train_with_vocabulary(self):
    n_classes = 3
    head = head_lib.MultiClassHead(
        n_classes, label_vocabulary=['aang', 'iroh', 'zuko'])

    logits = [[10., 0, 0], [0, 10, 0]]
    labels = [[b'iroh'], [b'iroh']]
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # loss = sum(cross_entropy(labels, logits)) / batch_size
    #      = sum(10, 0) / 2 = 5.
    expected_loss = 5.
    tol = 1e-2
    if context.executing_eagerly():
      loss = head.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=ModeKeys.TRAIN)
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      return

    def _train_op_fn(loss):
      del loss
      return control_flow_ops.no_op()

    spec = head.create_estimator_spec(
        features=features,
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      loss = sess.run(spec.loss)
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)

  def test_weighted_multi_example_train(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes, weight_column='label_weights')

    logits = np.array((
        (10, 0, 0),
        (0, 10, 0),
        (0, 0, 10),
    ), dtype=np.float32)
    labels = np.array(((1,), (2,), (2,)), dtype=np.int64)
    weights_3x1 = np.array(((1.,), (2.,), (3.,)), dtype=np.float64)
    expected_train_result = 'my_train_op'
    # loss = sum(cross_entropy(labels, logits) * [1, 2, 3]) / batch_size
    #      = sum([10, 10, 0] * [1, 2, 3]) / 3 = 30 / 3 = 10
    expected_loss = 10.
    tol = 1e-2
    features = {
        'x': np.array(((42,),), dtype=np.float32),
        'label_weights': weights_3x1
    }
    if context.executing_eagerly():
      loss = head.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=ModeKeys.TRAIN)
      self.assertIsNotNone(loss)
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      return

    def _train_op_fn(loss):
      return string_ops.string_join([
          constant_op.constant(expected_train_result),
          string_ops.as_string(loss, precision=2)
      ])

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)

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
          six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)),
          train_result)
      test_lib._assert_simple_summaries(
          self,
          {
              metric_keys.MetricKeys.LOSS:
                  expected_loss,
          },
          summary_str,
          tol)

  def test_multi_dim_weighted_train_create_loss(self):
    """Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 2]."""
    head = head_lib.MultiClassHead(
        n_classes=3,
        weight_column='weights')

    logits = np.array([[[10, 0, 0], [12, 0, 0]], [[0, 10, 0], [0, 15, 0]]],
                      dtype=np.float32)
    labels = np.array([[[0], [1]], [[1], [2]]], dtype=np.int64)
    weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)

    # unreduced_loss = cross_entropy(labels, logits) = [[0, 12], [0, 15]].
    # weights are reshaped to [2, 2, 1] to match logits.
    # training_loss = sum(1*0 + 1.5*12 + 2*0 + 2.5*15) / batch_size
    #               = 55.5 / (2*2) = 13.875
    expected_training_loss = 13.875
    tol = 1e-2
    if context.executing_eagerly():
      training_loss = head.loss(logits, labels, features={'weights': weights})
      self.assertAllClose(
          expected_training_loss, training_loss, rtol=tol, atol=tol)
      return

    training_loss = head.loss(logits, labels, features={'weights': weights})
    with self.cached_session():
      test_lib._initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)

  def test_multi_dim_weighted_train(self):
    """Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 2]."""
    head = head_lib.MultiClassHead(
        n_classes=3,
        weight_column='weights')

    logits = np.array([[[10, 0, 0], [12, 0, 0]], [[0, 10, 0], [0, 15, 0]]],
                      dtype=np.float32)
    labels = np.array([[[0], [1]], [[1], [2]]], dtype=np.int64)
    weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)
    tol = 1e-2
    # loss = cross_entropy(labels, logits) = [[0, 12], [0, 15]].
    # weighted_sum_loss = (1*0 + 1.5*12 + 2*0 + 2.5*15) = 55.5
    # training_loss = weighted_sum_loss / batch_size  = 55.5 / (2*2) = 13.875
    expected_loss = 13.875
    if context.executing_eagerly():
      loss = head.loss(
          logits=logits,
          labels=labels,
          features={'weights': weights},
          mode=ModeKeys.TRAIN)
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      return

    expected_train_result = 'my_train_op'

    def _train_op_fn(loss):
      return string_ops.string_join([
          constant_op.constant(expected_train_result),
          string_ops.as_string(loss, precision=2)
      ])

    spec = head.create_estimator_spec(
        features={'weights': weights},
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)

    # Assert predictions, loss, train_op, and summaries.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)),
          train_result)

  def test_multi_dim_train_weights_wrong_inner_dim(self):
    """Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 1]."""
    head = head_lib.MultiClassHead(n_classes=3, weight_column='weights')
    logits = np.array([[[10, 0, 0], [12, 0, 0]], [[0, 10, 0], [0, 15, 0]]],
                      dtype=np.float32)
    labels = np.array([[[0], [1]], [[1], [2]]], dtype=np.int64)
    weights = np.array([[1.], [2.]], dtype=np.float32)

    if context.executing_eagerly():
      with self.assertRaisesRegexp(ValueError, 'weights shape'):
        head.loss(
            logits=logits,
            labels=labels,
            features={'weights': weights},
            mode=ModeKeys.TRAIN)
      return

    def _no_op_train_fn(loss):
      del loss
      return control_flow_ops.no_op()

    spec = head.create_estimator_spec(
        features={'weights': weights},
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_no_op_train_fn)
    with self.cached_session():
      test_lib._initialize_variables(self, monitored_session.Scaffold())
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[logits_shape: \] \[2 2 3\] \[weights_shape: \] \[2 1\]'):
        spec.loss.eval()

  def test_multi_dim_train_weights_wrong_outer_dim(self):
    """Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 2, 3]."""
    head = head_lib.MultiClassHead(n_classes=3, weight_column='weights')
    logits = np.array([[[10, 0, 0], [12, 0, 0]], [[0, 10, 0], [0, 15, 0]]],
                      dtype=np.float32)
    labels = np.array([[[0], [1]], [[1], [2]]], dtype=np.int64)
    weights = np.array([[[1., 1.1, 1.2], [1.5, 1.6, 1.7]],
                        [[2., 2.1, 2.2], [2.5, 2.6, 2.7]]])

    if context.executing_eagerly():
      with self.assertRaisesRegexp(ValueError, 'weights shape'):
        head.loss(
            logits=logits,
            labels=labels,
            features={'weights': weights},
            mode=ModeKeys.TRAIN)
      return

    weights_placeholder = array_ops.placeholder(dtype=dtypes.float32)

    def _no_op_train_fn(loss):
      del loss
      return control_flow_ops.no_op()

    spec = head.create_estimator_spec(
        features={'weights': weights_placeholder},
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_no_op_train_fn)
    with self.cached_session():
      test_lib._initialize_variables(self, monitored_session.Scaffold())
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[logits_shape: \]\s\[2 2 3\]\s\[weights_shape: \]\s\[2 2 3\]'):
        spec.loss.eval({weights_placeholder: weights})

  def test_multi_dim_weighted_eval(self):
    """Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 2]."""
    head = head_lib.MultiClassHead(
        n_classes=3,
        weight_column='weights')
    logits = np.array([[[10, 0, 0], [12, 0, 0]], [[0, 10, 0], [0, 15, 0]]],
                      dtype=np.float32)
    labels = np.array([[[0], [1]], [[1], [2]]], dtype=np.int64)
    weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)
    # loss = cross_entropy(labels, logits) = [[0, 12], [0, 15]].
    # weighted_sum_loss = 1*0 + 1.5*12 + 2*0 + 2.5*15 = 55.5
    # training_loss = weighted_sum_loss / batch_size = 55.5 / (2*2) = 13.875
    expected_loss = 13.875
    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN:
            55.5 / np.sum(weights),
        keys.ACCURACY:
            (1. * 1. + 1.5 * 0. + 2. * 1. + 2.5 * 0.) / np.sum(weights),
    }
    tol = 1e-2
    if context.executing_eagerly():
      loss = head.loss(
          logits=logits,
          labels=labels,
          features={'weights': weights},
          mode=ModeKeys.TRAIN)
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)

      eval_metrics = head.metrics()
      updated_metrics = head.update_metrics(
          eval_metrics,
          features={'weights': weights},
          logits=logits,
          labels=labels)
      # Assert metrics.
      self.assertAllClose(
          expected_metrics,
          {k: updated_metrics[k].result() for k in updated_metrics},
          rtol=tol,
          atol=tol)
      return

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features={'weights': weights},
        mode=ModeKeys.EVAL,
        logits=logits,
        labels=labels)
    # Assert predictions, loss, and metrics.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, _ = sess.run((spec.loss, update_ops))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      # Check results of value ops (in `metrics`).
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval() for k in value_ops},
          rtol=tol,
          atol=tol)


@test_util.deprecated_graph_mode_only
class MultiClassHeadForEstimator(test.TestCase):
  """Tests for create_estimator_spec running in Graph mode only."""

  def test_train_with_optimizer(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes)

    logits = np.array((
        (10, 0, 0),
        (0, 10, 0),
    ), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    expected_train_result = 'my_train_op'

    class _Optimizer(object):

      def minimize(self, loss, global_step):
        del global_step
        return string_ops.string_join([
            constant_op.constant(expected_train_result),
            string_ops.as_string(loss, precision=2)
        ])

    # loss = sum(cross_entropy(labels, logits)) / batch_size
    #      = sum(10, 0) / 2 = 5.
    expected_loss = 5.
    spec = head.create_estimator_spec(
        features=features,
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        optimizer=_Optimizer())

    tol = 1e-2
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)),
          train_result)

  def test_train_with_update_ops(self):
    n_classes = 3
    with ops.Graph().as_default():
      w = variables.Variable(1)
      update_op = w.assign_add(1)

      t = variables.Variable('')
      expected_train_result = b'my_train_op'

      def _train_op_fn(loss):
        del loss
        return t.assign(expected_train_result)
      head = head_lib.MultiClassHead(n_classes, update_ops=[update_op])

      spec = head.create_estimator_spec(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=ModeKeys.TRAIN,
          logits=np.array((
              (10, 0, 0),
              (0, 10, 0),
          ), dtype=np.float32),
          labels=np.array(((1,), (1,)), dtype=np.int64),
          train_op_fn=_train_op_fn)

      with self.cached_session() as sess:
        test_lib._initialize_variables(self, spec.scaffold)
        sess.run(spec.train_op)
        w_value, t_value = sess.run([w, t])
        self.assertEqual(2, w_value)
        self.assertEqual(expected_train_result, t_value)

  def test_train_summaries_with_head_name(self):
    n_classes = 3
    head = head_lib.MultiClassHead(n_classes, name='some_multiclass_head')

    logits = np.array((
        (10, 0, 0),
        (0, 10, 0),
    ), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    # loss = sum(cross_entropy(labels, logits)) / batch_size= sum(10, 0) / 2 = 5
    expected_loss = 5.
    features = {'x': np.array(((42,),), dtype=np.int32)}

    def _train_op_fn(loss):
      del loss
      return control_flow_ops.no_op()

    spec = head.create_estimator_spec(
        features=features,
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)

    # Assert summaries.
    tol = 1e-2
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      summary_str = sess.run(spec.scaffold.summary_op)
      test_lib._assert_simple_summaries(
          self, {
              '{}/some_multiclass_head'.format(metric_keys.MetricKeys.LOSS):
                  expected_loss,
          }, summary_str, tol)

  def test_lookup_tables_in_graph(self):
    n_classes = 3
    head = head_lib.MultiClassHead(
        n_classes,
        label_vocabulary=['aang', 'iroh', 'zuko'])

    feature_columns = [feature_column.numeric_column('x')]
    # Create dnn estimator.
    est = dnn.DNNEstimator(
        head=head,
        hidden_units=(2, 2),
        feature_columns=feature_columns)

    def input_fn():
      return (
          {'x': np.array(((42,), (43,),), dtype=np.int32)},
          [[b'iroh'], [b'iroh']])

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
