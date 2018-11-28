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
"""Tests for binary_class_head.py."""

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
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.training import monitored_session
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import dnn
from tensorflow_estimator.python.estimator.canned import dnn_testing_utils
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.head import base_head_test as test_lib
from tensorflow_estimator.python.estimator.head import binary_class_head as head_lib


@test_util.run_all_in_graph_and_eager_modes
class BinaryClassHeadTest(test.TestCase):

  def test_threshold_too_small(self):
    with self.assertRaisesRegexp(ValueError, r'thresholds not in \(0, 1\)'):
      head_lib.BinaryClassHead(thresholds=(0., 0.5))

  def test_threshold_too_large(self):
    with self.assertRaisesRegexp(ValueError, r'thresholds not in \(0, 1\)'):
      head_lib.BinaryClassHead(thresholds=(0.5, 1.))

  def test_invalid_loss_reduction(self):
    with self.assertRaisesRegexp(
        ValueError, r'Invalid loss_reduction: invalid_loss_reduction'):
      head_lib.BinaryClassHead(loss_reduction='invalid_loss_reduction')
    with self.assertRaisesRegexp(
        ValueError, r'Invalid loss_reduction: none'):
      head_lib.BinaryClassHead(loss_reduction=losses.Reduction.NONE)

  def test_loss_fn_arg_labels_missing(self):
    def _loss_fn(logits):
      del logits  # Unused
    with self.assertRaisesRegexp(
        ValueError,
        r'loss_fn must contain argument: labels\. '
        r'Given arguments: \(\'logits\',\)'):
      head_lib.BinaryClassHead(loss_fn=_loss_fn)

  def test_loss_fn_arg_logits_missing(self):
    def _loss_fn(labels):
      del labels  # unused
    with self.assertRaisesRegexp(
        ValueError,
        r'loss_fn must contain argument: logits\. '
        r'Given arguments: \(\'labels\',\)'):
      head_lib.BinaryClassHead(loss_fn=_loss_fn)

  def test_loss_fn_arg_features_ok(self):
    def _loss_fn(labels, logits, features):
      del labels, logits, features  # Unused
      head_lib.BinaryClassHead(loss_fn=_loss_fn)

  def test_loss_fn_arg_invalid(self):
    def _loss_fn(labels, logits, name=None):
      del labels, logits, name  # Unused
    with self.assertRaisesRegexp(
        ValueError,
        r'loss_fn has unexpected args: \[\'name\'\]'):
      head_lib.BinaryClassHead(loss_fn=_loss_fn)

  def test_invalid_logits_shape(self):
    head = head_lib.BinaryClassHead()
    self.assertEqual(1, head.logits_dimension)

    # Logits should be shape (batch_size, 1).
    logits_2x2 = np.array(((45., 44.), (41., 42.),))

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
        features={'x': np.array(((42.,),))},
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits_placeholder)
    with self.cached_session():
      with self.assertRaisesRegexp(errors.OpError, 'logits shape'):
        spec.predictions[pred_key].eval({
            logits_placeholder: logits_2x2
        })

  def test_invalid_labels_shape(self):
    head = head_lib.BinaryClassHead()
    self.assertEqual(1, head.logits_dimension)

    # Labels and logits should be shape (batch_size, 1).
    labels_2x2 = np.array(((45., 44.), (41., 42.),))
    logits_2x1 = np.array(((45.,), (41.,),))
    features = {'x': np.array(((42.,),))}

    # Static shape.
    with self.assertRaisesRegexp(ValueError, 'Mismatched label shape'):
      training_loss = head.loss(
          logits=logits_2x1,
          labels=labels_2x2,
          features=features,
          mode=model_fn.ModeKeys.EVAL)
      self.evaluate(training_loss)
    if context.executing_eagerly():
      return

    # Dynamic shape only works in Graph mode.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    training_loss = head.loss(
        logits=logits_placeholder,
        labels=labels_placeholder,
        features=features,
        mode=model_fn.ModeKeys.EVAL)
    with self.cached_session():
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[expected_labels_shape: \] \[2 1\] \[labels_shape: \] \[2 2\]'):
        training_loss.eval({
            logits_placeholder: logits_2x1,
            labels_placeholder: labels_2x2
        })

  def test_incompatible_labels_shape(self):
    head = head_lib.BinaryClassHead()
    self.assertEqual(1, head.logits_dimension)

    # Both logits and labels should be shape (batch_size, 1).
    values_2x1 = np.array(((0.,), (1.,),))
    values_3x1 = np.array(((0.,), (1.,), (0.,),))
    features = {'x': values_2x1}

    # Static shape for eager mode.
    if context.executing_eagerly():
      with self.assertRaisesRegexp(
          ValueError, 'labels shape'):
        head.loss(
            logits=values_2x1,
            labels=values_3x1,
            features=features,
            mode=model_fn.ModeKeys.EVAL)
      with self.assertRaisesRegexp(
          ValueError, 'labels shape'):
        head.loss(
            logits=values_3x1,
            labels=values_2x1,
            features=features,
            mode=model_fn.ModeKeys.EVAL)
      return

    # Static shape for Graph mode.
    with self.assertRaisesRegexp(
        ValueError, 'logits and labels must have the same shape'):
      head.loss(
          logits=values_2x1,
          labels=values_3x1,
          features=features,
          mode=model_fn.ModeKeys.EVAL)
    with self.assertRaisesRegexp(
        ValueError, 'logits and labels must have the same shape'):
      head.loss(
          logits=values_3x1,
          labels=values_2x1,
          features=features,
          mode=model_fn.ModeKeys.EVAL)
    # Dynamic shape only works in Graph mode.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    training_loss = head.loss(
        logits=logits_placeholder,
        labels=labels_placeholder,
        features=features,
        mode=model_fn.ModeKeys.EVAL)
    with self.cached_session():
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[expected_labels_shape: \] \[3 1\] \[labels_shape: \] \[2 1\]'):
        training_loss.eval({
            labels_placeholder: values_2x1,
            logits_placeholder: values_3x1
        })
    with self.cached_session():
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[expected_labels_shape: \] \[2 1\] \[labels_shape: \] \[3 1\]'):
        training_loss.eval({
            labels_placeholder: values_3x1,
            logits_placeholder: values_2x1
        })

  def test_predict(self):
    head = head_lib.BinaryClassHead()
    self.assertEqual(1, head.logits_dimension)

    logits = [[0.3], [-0.4]]
    expected_logistics = [[0.574443], [0.401312]]
    expected_probabilities = [[0.425557, 0.574443], [0.598688, 0.401312]]
    expected_class_ids = [[1], [0]]
    expected_classes = [[b'1'], [b'0']]
    expected_export_classes = [[b'0', b'1']] * 2

    keys = prediction_keys.PredictionKeys
    preds = head.predictions(
        logits, [keys.LOGITS, keys.LOGISTIC, keys.PROBABILITIES, keys.CLASS_IDS,
                 keys.CLASSES])
    self.assertAllClose(logits, self.evaluate(preds[keys.LOGITS]))
    self.assertAllClose(expected_logistics,
                        self.evaluate(preds[keys.LOGISTIC]))
    self.assertAllClose(expected_probabilities,
                        self.evaluate(preds[keys.PROBABILITIES]))
    self.assertAllClose(expected_class_ids,
                        self.evaluate(preds[keys.CLASS_IDS]))
    self.assertAllEqual(expected_classes, self.evaluate(preds[keys.CLASSES]))
    if context.executing_eagerly():
      return

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits)

    # Assert spec contains expected tensors.
    self.assertIsNone(spec.loss)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNone(spec.train_op)
    self.assertItemsEqual(
        ('classification', 'regression', 'predict',
         test_lib._DEFAULT_SERVING_KEY), spec.export_outputs.keys())
    test_lib._assert_no_hooks(self, spec)

    # Assert predictions.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      predictions = sess.run(spec.predictions)
      self.assertAllClose(logits,
                          predictions[prediction_keys.PredictionKeys.LOGITS])
      self.assertAllClose(expected_logistics,
                          predictions[prediction_keys.PredictionKeys.LOGISTIC])
      self.assertAllClose(
          expected_probabilities,
          predictions[prediction_keys.PredictionKeys.PROBABILITIES])
      self.assertAllClose(expected_class_ids,
                          predictions[prediction_keys.PredictionKeys.CLASS_IDS])
      self.assertAllEqual(expected_classes,
                          predictions[prediction_keys.PredictionKeys.CLASSES])
      self.assertAllClose(
          expected_probabilities,
          sess.run(spec.export_outputs[test_lib._DEFAULT_SERVING_KEY].scores))
      self.assertAllEqual(
          expected_export_classes,
          sess.run(spec.export_outputs[test_lib._DEFAULT_SERVING_KEY].classes))
      self.assertAllClose(expected_logistics,
                          sess.run(spec.export_outputs['regression'].value))

  def test_predict_with_vocabulary_list(self):
    head = head_lib.BinaryClassHead(label_vocabulary=['aang', 'iroh'])

    logits = [[1.], [0.]]
    expected_classes = [[b'iroh'], [b'aang']]

    pred_key = prediction_keys.PredictionKeys.CLASSES
    if context.executing_eagerly():
      preds = head.predictions(logits, [pred_key])
      self.assertAllEqual(expected_classes, preds[pred_key])
      return
    preds = head.predictions(logits, [pred_key])
    with self.cached_session():
      test_lib._initialize_variables(self, monitored_session.Scaffold())
      self.assertAllEqual(
          expected_classes, preds[pred_key].eval())

  def test_eval_create_loss(self):
    head = head_lib.BinaryClassHead()
    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.int32)}

    # loss = sum(cross_entropy(labels, logits)) / batch_size
    #      = sum([0, 41]) / 2 = 20.5
    expected_training_loss = 20.5
    # Create loss.
    training_loss = head.loss(
        logits=logits,
        labels=labels,
        features=features,
        mode=model_fn.ModeKeys.EVAL)
    self.assertAllClose(expected_training_loss, self.evaluate(training_loss),
                        rtol=1e-2, atol=1e-2)

  def test_eval_create_loss_loss_fn(self):
    """Tests head.create_loss for eval mode and custom loss_fn."""
    loss = np.array([[1.], [2.]], dtype=np.float32)
    logits_input = np.array([[-10.], [10.]], dtype=np.float32)
    labels_input = np.array([[1], [0]], dtype=np.int64)
    def _loss_fn(labels, logits):
      check_labels = control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(labels, labels_input)),
          data=[labels])
      check_logits = control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(logits, logits_input)),
          data=[logits])
      with ops.control_dependencies([check_labels, check_logits]):
        return constant_op.constant(loss)
    head = head_lib.BinaryClassHead(loss_fn=_loss_fn)

    actual_training_loss = head.loss(
        logits=logits_input,
        labels=labels_input,
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.EVAL)
    self.assertAllClose(np.sum(loss) / 2., self.evaluate(actual_training_loss))

  def test_eval_create_loss_loss_fn_wrong_shape(self):
    """Tests custom loss_fn that returns Tensor of unexpected shape."""
    loss = np.array([1., 2.], dtype=np.float32)
    def _loss_fn(labels, logits):
      del labels, logits  # Unused
      return constant_op.constant(loss)
    head = head_lib.BinaryClassHead(
        loss_fn=_loss_fn)

    logits = np.array([[-10.], [10.]], dtype=np.float32)
    labels = np.array([[1], [0]], dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}

    if context.executing_eagerly():
      with self.assertRaisesRegexp(ValueError, 'loss_shape'):
        head.loss(logits=logits, labels=labels, features=features,
                  mode=model_fn.ModeKeys.EVAL)
    else:
      actual_training_loss = head.loss(
          logits=logits, labels=labels, features=features,
          mode=model_fn.ModeKeys.EVAL)
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[loss_fn must return Tensor of shape \[D0, D1, ... DN, 1\]\. \] '
          r'\[logits_shape: \] \[2 1\] \[loss_shape: \] \[2\]'):
        with self.cached_session():
          test_lib._initialize_variables(self, monitored_session.Scaffold())
          actual_training_loss.eval()

  def test_eval_labels_none(self):
    """Tests that error is raised when labels is None."""
    head = head_lib.BinaryClassHead()

    with self.assertRaisesRegexp(
        ValueError, r'You must provide a labels Tensor\. Given: None\.'):
      head.loss(
          logits=np.array(((45,), (-41,),), dtype=np.float32),
          labels=None,
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=model_fn.ModeKeys.EVAL)

  def test_eval(self):
    head = head_lib.BinaryClassHead()
    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.int32)}

    # loss = sum(cross_entropy(labels, logits)) / batch_size
    #      = sum(0, 41) / 2 = 41 / 2 = 20.5
    expected_loss = 20.5
    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: expected_loss,
        keys.ACCURACY: 1./2,
        keys.PRECISION: 1.,
        keys.RECALL: 1./2,
        keys.PREDICTION_MEAN: 1./2,
        keys.LABEL_MEAN: 2./2,
        # TODO(b/118843532): update metrics
        # keys.ACCURACY_BASELINE: 2./2,
        # keys.AUC: 0.,
        # keys.AUC_PR: 1.,
    }
    if context.executing_eagerly():
      eval_metrics = head.metrics()
      updated_metrics = head.update_metrics(eval_metrics, features, logits,
                                            labels)
      self.assertItemsEqual(expected_metrics.keys(), updated_metrics.keys())
      self.assertAllClose(
          expected_metrics,
          {k: updated_metrics[k].result() for k in updated_metrics})
      loss = head.loss(
          logits, labels, features=features, mode=model_fn.ModeKeys.EVAL)
      self.assertIsNotNone(loss)
      self.assertAllClose(expected_loss, loss)
      return

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
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
      self.assertAllClose(expected_loss, loss)
      # Check results of value ops (in `metrics`).
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval() for k in value_ops})

  def test_eval_metric_ops_with_head_name(self):
    head = head_lib.BinaryClassHead(
        name='some_binary_head')
    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    keys = metric_keys.MetricKeys
    expected_metric_keys = [
        '{}/some_binary_head'.format(keys.LOSS_MEAN),
        '{}/some_binary_head'.format(keys.ACCURACY),
        '{}/some_binary_head'.format(keys.PRECISION),
        '{}/some_binary_head'.format(keys.RECALL),
        '{}/some_binary_head'.format(keys.PREDICTION_MEAN),
        '{}/some_binary_head'.format(keys.LABEL_MEAN),
        # TODO(b/118843532): update metrics
        # '{}/some_binary_head'.format(keys.ACCURACY_BASELINE),
        # '{}/some_binary_head'.format(keys.AUC),
        # '{}/some_binary_head'.format(keys.AUC_PR),
    ]
    eval_metrics = head.metrics()
    updated_metrics = head.update_metrics(eval_metrics, features, logits,
                                          labels)
    self.assertItemsEqual(expected_metric_keys, updated_metrics.keys())

  def test_eval_with_regularization_losses(self):
    head = head_lib.BinaryClassHead(
        loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)
    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    regularization_losses = [1.5, 0.5]
    expected_regularization_loss = 2.
    # unregularized_loss = sum(cross_entropy(labels, logits)) / batch_size
    #                    = sum(0, 41) / 2 = 20.5
    expected_unregularized_loss = 20.5
    expected_regularized_loss = (
        expected_unregularized_loss + expected_regularization_loss)

    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: expected_unregularized_loss,
        keys.LOSS_REGULARIZATION: expected_regularization_loss,
        keys.ACCURACY: 1./2,
        keys.PRECISION: 1.,
        keys.RECALL: 1./2,
        keys.PREDICTION_MEAN: 1./2,
        keys.LABEL_MEAN: 2./2,
        # TODO(b/118843532): update metrics
        # keys.ACCURACY_BASELINE: 2./2,
        # keys.AUC: 0.,
        # keys.AUC_PR: 1.,
    }
    if context.executing_eagerly():
      eval_metrics = head.metrics(regularization_losses=regularization_losses)
      updated_metrics = head.update_metrics(
          eval_metrics, features, logits, labels,
          regularization_losses=regularization_losses)
      # Assert metrics.
      self.assertAllClose(
          expected_metrics,
          {k: updated_metrics[k].result() for k in updated_metrics})
      return

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
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
      self.assertAllClose(expected_regularized_loss, loss)
      # Check results of value ops (in `metrics`).
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval() for k in value_ops})

  def test_eval_with_vocabulary_list_create_loss(self):
    head = head_lib.BinaryClassHead(
        label_vocabulary=['aang', 'iroh'])
    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = [[b'iroh'], [b'iroh']]
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # loss = sum(cross_entropy(labels, logits)) / batch_size
    #      = sum([0, 41]) / 2 = 20.5
    expected_training_loss = 20.5
    # Create loss.
    if context.executing_eagerly():
      training_loss = head.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=model_fn.ModeKeys.EVAL)
      self.assertAllClose(expected_training_loss, training_loss,
                          rtol=1e-2, atol=1e-2)
      return
    training_loss = head.loss(
        logits=logits,
        labels=labels,
        features=features,
        mode=model_fn.ModeKeys.EVAL)
    with self.cached_session():
      test_lib._initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(expected_training_loss, training_loss.eval())

  def test_eval_with_vocabulary_list(self):
    head = head_lib.BinaryClassHead(
        label_vocabulary=['aang', 'iroh'])
    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = [[b'iroh'], [b'iroh']]
    features = {'x': np.array(((42,),), dtype=np.int32)}

    accuracy_key = metric_keys.MetricKeys.ACCURACY
    if context.executing_eagerly():
      eval_metrics = head.metrics()
      updated_metrics = head.update_metrics(eval_metrics, features, logits,
                                            labels)
      self.assertAllClose(1. / 2, updated_metrics[accuracy_key].result())
      return

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)

    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      sess.run(update_ops)
      self.assertAllClose(1. / 2,
                          value_ops[accuracy_key].eval())

  def test_eval_with_thresholds_create_loss(self):
    thresholds = [0.25, 0.5, 0.75]
    head = head_lib.BinaryClassHead(thresholds=thresholds)
    logits = np.array(((-1,), (1,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # probabilities[i] = 1/(1 + exp(-logits[i])) =>
    # probabilities = [1/(1 + exp(1)), 1/(1 + exp(-1))] = [0.269, 0.731]
    # unreduced_loss = -ln(probabilities[label[i]])) = [-ln(0.269), -ln(0.731)]
    #      = [1.31304389, 0.31334182]
    # weighted sum loss = 1.62638571
    # loss = 0.813192855
    expected_training_loss = 0.813192855
    # Create loss.
    training_loss = head.loss(
        logits=logits,
        labels=labels,
        features=features,
        mode=model_fn.ModeKeys.EVAL)
    self.assertAllClose(expected_training_loss, self.evaluate(training_loss),
                        rtol=1e-2, atol=1e-2)

  def test_eval_with_thresholds(self):
    thresholds = [0.25, 0.5, 0.75]
    head = head_lib.BinaryClassHead(
        thresholds=thresholds)
    logits = np.array(((-1,), (1,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.int32)}

    # probabilities[i] = 1/(1 + exp(-logits[i])) =>
    # probabilities = [1/(1 + exp(1)), 1/(1 + exp(-1))] = [0.269, 0.731]
    # loss = -sum(ln(probabilities[label[i]])) / batch_size
    #      = (-ln(0.269) -ln(0.731)) / 2
    #      = 1.62652338 / 2
    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: 1.62652338 / 2.,
        keys.ACCURACY: 1./2,
        keys.PRECISION: 1.,
        keys.RECALL: .5,
        keys.PREDICTION_MEAN: 1./2,
        keys.LABEL_MEAN: 2./2,
        # TODO(b/118843532): update metrics
        # keys.ACCURACY_BASELINE: 2./2,
        # keys.AUC: 0.,
        # keys.AUC_PR: 1.,
        keys.ACCURACY_AT_THRESHOLD % thresholds[0]: 1.,
        keys.PRECISION_AT_THRESHOLD % thresholds[0]: 1.,
        keys.RECALL_AT_THRESHOLD % thresholds[0]: 1.,
        keys.ACCURACY_AT_THRESHOLD % thresholds[1]: .5,
        keys.PRECISION_AT_THRESHOLD % thresholds[1]: 1.,
        keys.RECALL_AT_THRESHOLD % thresholds[1]: .5,
        keys.ACCURACY_AT_THRESHOLD % thresholds[2]: 0.,
        keys.PRECISION_AT_THRESHOLD % thresholds[2]: 0.,
        keys.RECALL_AT_THRESHOLD % thresholds[2]: 0.,
    }
    tol = 1e-2
    if context.executing_eagerly():
      # Create loss.
      training_loss = head.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=model_fn.ModeKeys.EVAL)
      self.assertAllClose(1.62652338 / 2., self.evaluate(training_loss))
      # Eval metrics.
      eval_metrics = head.metrics()
      updated_metrics = head.update_metrics(
          eval_metrics, features, logits, labels)
      # Assert metrics.
      self.assertItemsEqual(expected_metrics.keys(), updated_metrics.keys())
      self.assertAllClose(
          expected_metrics,
          {k: self.evaluate(
              updated_metrics[k].result()) for k in updated_metrics},
          atol=tol, rtol=tol)
      return

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)
    self.assertItemsEqual(expected_metrics.keys(), spec.eval_metric_ops.keys())
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, _ = sess.run((spec.loss, update_ops))
      self.assertAllClose(1.62652338 / 2., loss)
      # Check results of value ops (in `metrics`).
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval() for k in value_ops},
          atol=tol,
          rtol=tol)

  def test_train_create_loss(self):
    head = head_lib.BinaryClassHead()

    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.float64)
    features = {'x': np.array(((42,),), dtype=np.float32)}
    # unreduced_loss = cross_entropy(labels, logits) = [0, 41]
    # weights default to 1.
    # training loss = (1 * 0 + 1 * 41) / 2 = 20.5
    expected_training_loss = 20.5
    # Create loss.
    if context.executing_eagerly():
      training_loss = head.loss(logits, labels, features)
      self.assertAllClose(
          expected_training_loss, training_loss)
      return

    training_loss = head.loss(logits, labels, features)
    with self.cached_session():
      test_lib._initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(expected_training_loss, training_loss.eval())

  def test_train_create_loss_loss_reduction(self):
    """Tests create_loss with loss_reduction."""
    head = head_lib.BinaryClassHead(
        loss_reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.float64)
    features = {'x': np.array(((42,),), dtype=np.float32)}
    # unreduced_loss = cross_entropy(labels, logits) = [0, 41]
    # weights default to 1.
    # training loss = (1 * 0 + 1 * 41) / num_nonzero_weights
    expected_training_loss = 41. / 2.
    # Create loss.
    if context.executing_eagerly():
      training_loss = head.loss(logits, labels, features)
      self.assertAllClose(
          expected_training_loss, training_loss)
      return

    training_loss = head.loss(logits, labels, features)
    with self.cached_session():
      test_lib._initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(expected_training_loss, training_loss.eval())

  def test_train_labels_none(self):
    """Tests that error is raised when labels is None."""
    head = head_lib.BinaryClassHead()
    with self.assertRaisesRegexp(
        ValueError, r'You must provide a labels Tensor\. Given: None\.'):
      head.loss(
          logits=np.array(((45,), (-41,),), dtype=np.float32),
          labels=None,
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=model_fn.ModeKeys.TRAIN)

  def test_train(self):
    head = head_lib.BinaryClassHead()

    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.float64)
    features = {'x': np.array(((42,),), dtype=np.float32)}
    # loss = sum(cross_entropy(labels, logits)) / batch_size
    #      = sum(0, 41) / 2 = 41 / 2 = 20.5
    expected_loss = 20.5
    if context.executing_eagerly():
      loss = head.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=model_fn.ModeKeys.TRAIN)
      self.assertIsNotNone(loss)
      self.assertAllClose(expected_loss, loss)
      return

    expected_train_result = b'my_train_op'
    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)
    # Assert spec contains expected tensors.
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
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)
      test_lib._assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
      }, summary_str)

  def test_train_one_dim_create_loss(self):
    """Tests create_loss with 1D labels and weights (shape [batch_size])."""
    head = head_lib.BinaryClassHead(
        weight_column='label_weights')

    # Create estimator spec.
    logits = np.array(((45,), (-41,), (44,)), dtype=np.float32)
    labels_rank_1 = np.array((1., 1., 0.,))
    weights_rank_1 = np.array(((1., .1, 1.5,)), dtype=np.float64)
    features = {
        'x': np.array(((42.,), (43.,), (44.,)), dtype=np.float32),
        'label_weights': weights_rank_1,
    }
    # unreduced_loss = cross_entropy(labels, logits) = [0, 41, 44]
    # weights are reshaped to [3, 1] to match logits.
    # training loss = (1 * 0 + .1 * 41 + 1.5 * 44) / 3 = 23.366666667
    expected_training_loss = 23.366666667
    # Create loss.
    if context.executing_eagerly():
      training_loss = head.loss(logits, labels_rank_1, features)
      self.assertAllClose(
          expected_training_loss, training_loss)
      return

    training_loss = head.loss(logits, labels_rank_1, features)
    self.assertAllClose(
        expected_training_loss, self.evaluate(training_loss))

  def test_train_one_dim(self):
    """Tests train with 1D labels and weights (shape [batch_size])."""
    head = head_lib.BinaryClassHead(
        weight_column='label_weights')

    # Create estimator spec.
    logits = np.array(((45,), (-41,), (44,)), dtype=np.float32)
    labels_rank_1 = np.array((1., 1., 0.,))
    weights_rank_1 = np.array(((1., .1, 1.5,)), dtype=np.float64)
    self.assertEqual((3,), labels_rank_1.shape)
    self.assertEqual((3,), weights_rank_1.shape)
    features = {
        'x': np.array(((42.,), (43.,), (44.,)), dtype=np.float32),
        'label_weights': weights_rank_1,
    }
    # losses = label_weights*cross_entropy(labels, logits)
    #        = (1*0 + .1*41 + 1.5*44) = (1, 4.1, 66)
    # loss = sum(losses) / batch_size = (1 + 4.1 + 66) / 3 = 23.366666667
    expected_loss = 23.366666667
    if context.executing_eagerly():
      loss = head.loss(
          logits=logits,
          labels=labels_rank_1,
          features=features,
          mode=model_fn.ModeKeys.TRAIN)
      self.assertIsNotNone(loss)
      self.assertAllClose(expected_loss, loss)
      return

    expected_train_result = b'my_train_op'
    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels_rank_1,
        train_op_fn=_train_op_fn)
    # Assert spec contains expected tensors.
    self.assertIsNotNone(spec.loss)
    self.assertIsNotNone(spec.train_op)
    # Assert predictions, loss, and metrics.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str = sess.run((
          spec.loss, spec.train_op, spec.scaffold.summary_op))
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)
      test_lib._assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
      }, summary_str)

  def test_train_with_regularization_losses(self):
    head = head_lib.BinaryClassHead(
        loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)

    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.float64)
    features = {'x': np.array(((42,),), dtype=np.float32)}
    regularization_losses = [1.5, 0.5]
    expected_regularization_loss = 2.
    # unregularized_loss = sum(cross_entropy(labels, logits)) / batch_size
    #                    = sum(0, 41) / 2 = 20.5
    # loss = unregularized_loss + regularization_loss = 22.5
    expected_loss = 22.5
    if context.executing_eagerly():
      loss = head.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=model_fn.ModeKeys.TRAIN,
          regularization_losses=regularization_losses)
      self.assertAllClose(expected_loss, loss)
      return

    expected_train_result = b'my_train_op'
    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
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
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)
      test_lib._assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
          metric_keys.MetricKeys.LOSS_REGULARIZATION: (
              expected_regularization_loss),
      }, summary_str)

  def test_float_labels_invalid_values(self):
    head = head_lib.BinaryClassHead()

    logits = np.array([[0.5], [-0.3]], dtype=np.float32)
    labels = np.array([[1.2], [0.4]], dtype=np.float32)
    features = {'x': np.array([[42]], dtype=np.float32)}
    if context.executing_eagerly():
      with self.assertRaisesRegexp(
          ValueError, r'Labels must be <= 2 - 1'):
        head.loss(
            logits=logits,
            labels=labels,
            features=features,
            mode=model_fn.ModeKeys.TRAIN)
      return

    training_loss = head.loss(
        logits=logits,
        labels=labels,
        features=features,
        mode=model_fn.ModeKeys.TRAIN)
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        r'Labels must be <= n_classes - 1'):
      with self.cached_session():
        test_lib._initialize_variables(self, monitored_session.Scaffold())
        training_loss.eval()

  def test_float_labels_train_create_loss(self):
    head = head_lib.BinaryClassHead()

    logits = np.array([[0.5], [-0.3]], dtype=np.float32)
    labels = np.array([[0.8], [0.4]], dtype=np.float32)
    features = {'x': np.array([[42]], dtype=np.float32)}
    # loss = cross_entropy(labels, logits)
    #      = -label[i]*sigmoid(logit[i]) -(1-label[i])*sigmoid(-logit[i])
    #      = [-0.8 * log(sigmoid(0.5)) -0.2 * log(sigmoid(-0.5)),
    #         -0.4 * log(sigmoid(-0.3)) -0.6 * log(sigmoid(0.3))]
    #      = [0.57407698418, 0.67435524446]
    # weighted_sum_loss = 0.57407698418 + 0.67435524446
    # training_loss = weighted_sum_loss / 2 = 0.62421611432
    expected_training_loss = 0.62421611432
    # Create loss.
    training_loss = head.loss(
        logits=logits,
        labels=labels,
        features=features,
        mode=model_fn.ModeKeys.TRAIN)
    self.assertAllClose(
        expected_training_loss, self.evaluate(training_loss))

  def test_float_labels_train(self):
    head = head_lib.BinaryClassHead()

    logits = np.array([[0.5], [-0.3]], dtype=np.float32)
    labels = np.array([[0.8], [0.4]], dtype=np.float32)
    expected_train_result = b'my_train_op'
    features = {'x': np.array([[42]], dtype=np.float32)}
    # loss = sum(cross_entropy(labels, logits)) / batch_size
    #      = sum(-label[i]*sigmoid(logit[i]) -(1-label[i])*sigmoid(-logit[i])
    #           ) / batch_size
    #      = -0.8 * log(sigmoid(0.5)) -0.2 * log(sigmoid(-0.5)) / 2
    #        -0.4 * log(sigmoid(-0.3)) -0.6 * log(sigmoid(0.3)) / 2
    #      = 1.2484322 / 2 = 0.6242161
    expected_loss = 0.6242161
    # Create loss.
    training_loss = head.loss(
        logits=logits, labels=labels, features=features,
        mode=model_fn.ModeKeys.TRAIN)
    self.assertAlmostEqual(
        expected_loss, self.evaluate(training_loss), delta=1.e-5)
    if context.executing_eagerly():
      return

    def _train_op_fn(loss):
      with ops.control_dependencies((dnn_testing_utils.assert_close(
          math_ops.to_float(expected_loss), math_ops.to_float(loss)),)):
        return constant_op.constant(expected_train_result)
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)
    # Assert predictions, loss, train_op, and summaries.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAlmostEqual(expected_loss, loss, delta=1.e-5)
      self.assertEqual(expected_train_result, train_result)

  def test_float_labels_eval_create_loss(self):
    head = head_lib.BinaryClassHead()

    logits = np.array([[0.5], [-0.3]], dtype=np.float32)
    labels = np.array([[0.8], [0.4]], dtype=np.float32)
    features = {'x': np.array([[42]], dtype=np.float32)}
    # unreduced_loss = cross_entropy(labels, logits)
    #      = -label[i]*sigmoid(logit[i]) -(1-label[i])*sigmoid(-logit[i])
    #      = [-0.8 * log(sigmoid(0.5)) -0.2 * log(sigmoid(-0.5)),
    #         -0.4 * log(sigmoid(-0.3)) -0.6 * log(sigmoid(0.3))]
    #      = [0.57407698418, 0.67435524446]
    # weighted_sum_loss = 0.57407698418 + 0.67435524446
    # loss = weighted_sum_loss / batch_size = 1.24843222864 / 2 = 0.62421611432
    expected_training_loss = 0.62421611432
    # Create loss.
    training_loss = head.loss(
        logits=logits, labels=labels, features=features,
        mode=model_fn.ModeKeys.EVAL)
    self.assertAllClose(expected_training_loss, self.evaluate(training_loss),
                        rtol=1e-2, atol=1e-2)

  def test_float_labels_eval(self):
    head = head_lib.BinaryClassHead()

    logits = np.array([[0.5], [-0.3]], dtype=np.float32)
    labels = np.array([[0.8], [0.4]], dtype=np.float32)
    features = {'x': np.array([[42]], dtype=np.float32)}

    # loss_sum = sum(cross_entropy(labels, logits))
    #      = sum(-label[i]*sigmoid(logit[i]) -(1-label[i])*sigmoid(-logit[i]))
    #      = -0.8 * log(sigmoid(0.5)) -0.2 * log(sigmoid(-0.5))
    #        -0.4 * log(sigmoid(-0.3)) -0.6 * log(sigmoid(0.3))
    #      = 1.2484322
    # loss = loss_sum / batch_size = 1.2484322 / 2 = 0.6242161
    expected_loss = 0.6242161
    # Create loss.
    training_loss = head.loss(
        logits=logits, labels=labels, features=features,
        mode=model_fn.ModeKeys.EVAL)
    self.assertAlmostEqual(expected_loss, self.evaluate(training_loss),
                           delta=1.e-5)
    # Eval metrics.
    loss_mean_key = metric_keys.MetricKeys.LOSS_MEAN
    if context.executing_eagerly():
      eval_metrics = head.metrics()
      updated_metrics = head.update_metrics(eval_metrics, features, logits,
                                            labels)
      self.assertAlmostEqual(
          expected_loss, updated_metrics[loss_mean_key].result().numpy())
      return

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, _ = sess.run((spec.loss, update_ops))
      self.assertAlmostEqual(expected_loss, loss, delta=1.e-5)
      self.assertAlmostEqual(
          expected_loss, value_ops[loss_mean_key].eval())

  def test_weighted_multi_example_predict(self):
    """3 examples, 1 batch."""
    head = head_lib.BinaryClassHead(weight_column='label_weights')
    # Create estimator spec.
    logits = np.array(((45,), (-41,), (44,)), dtype=np.int32)
    pred_keys = prediction_keys.PredictionKeys
    keys = [pred_keys.LOGITS, pred_keys.LOGISTIC, pred_keys.PROBABILITIES,
            pred_keys.CLASS_IDS, pred_keys.CLASSES]
    predictions = head.predictions(logits, keys)
    self.assertAllClose(
        logits.astype(np.float32), self.evaluate(predictions[pred_keys.LOGITS]))
    self.assertAllClose(
        nn.sigmoid(logits.astype(np.float32)),
        self.evaluate(predictions[pred_keys.LOGISTIC]))
    self.assertAllClose(
        [[0., 1.], [1., 0.], [0., 1.]],
        self.evaluate(predictions[pred_keys.PROBABILITIES]))
    self.assertAllClose([[1], [0], [1]],
                        self.evaluate(predictions[pred_keys.CLASS_IDS]))
    self.assertAllEqual([[b'1'], [b'0'], [b'1']],
                        self.evaluate(predictions[pred_keys.CLASSES]))

  def test_weighted_multi_example_eval(self):
    """3 examples, 1 batch."""
    head = head_lib.BinaryClassHead(weight_column='label_weights')

    logits = np.array(((45,), (-41,), (44,)), dtype=np.int32)
    labels = np.array(((1,), (1,), (0,)), dtype=np.int32)
    features = {
        'x': np.array(((42,), (43,), (44,)), dtype=np.int32),
        'label_weights': np.array(((1.,), (.1,), (1.5,)), dtype=np.float32)}
    # label_mean = (1*1 + .1*1 + 1.5*0)/(1 + .1 + 1.5) = 1.1/2.6
    #            = .42307692307
    expected_label_mean = .42307692307
    # losses = label_weights*cross_entropy(labels, logits)
    #        = (1*0 + .1*41 + 1.5*44) = (1, 4.1, 66)
    # loss = sum(losses) / batch_size = (1 + 4.1 + 66) / 3 = 70.1 / 3 = 23.36667
    expected_loss = 23.366666667
    keys = metric_keys.MetricKeys
    expected_metrics = {
        # loss_mean = loss/sum(label_weights) = 70.1/(1 + .1 + 1.5)
        #           = 70.1/2.6 = 26.9615384615
        keys.LOSS_MEAN: 26.9615384615,
        # accuracy = (1*1 + .1*0 + 1.5*0)/(1 + .1 + 1.5) = 1/2.6 = .38461538461
        keys.ACCURACY: .38461538461,
        keys.PRECISION: 1./2.5,
        keys.RECALL: 1./1.1,
        # prediction_mean = (1*1 + .1*0 + 1.5*1)/(1 + .1 + 1.5) = 2.5/2.6
        #                 = .96153846153
        keys.PREDICTION_MEAN: .96153846153,
        keys.LABEL_MEAN: expected_label_mean,
        # TODO(b/118843532): update metrics
        # keys.ACCURACY_BASELINE: 1 - expected_label_mean,
        # keys.AUC: .45454565,
        # keys.AUC_PR: .6737757325172424,
    }
    if context.executing_eagerly():
      eval_metrics = head.metrics()
      updated_metrics = head.update_metrics(eval_metrics, features, logits,
                                            labels)
      self.assertItemsEqual(expected_metrics.keys(), updated_metrics.keys())
      self.assertAllClose(
          expected_metrics,
          {k: updated_metrics[k].result() for k in updated_metrics})
      loss = head.loss(
          logits, labels, features=features, mode=model_fn.ModeKeys.EVAL)
      self.assertIsNotNone(loss)
      self.assertAllClose(expected_loss, loss)
      return

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)
    # Assert spec contains expected tensors.
    self.assertIsNotNone(spec.loss)
    self.assertItemsEqual(expected_metrics.keys(), spec.eval_metric_ops.keys())
    # Assert predictions, loss, and metrics.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, _ = sess.run((spec.loss, update_ops))
      self.assertAllClose(expected_loss, loss)
      # Check results of value ops (in `metrics`).
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval() for k in value_ops})

  def test_weighted_multi_example_train(self):
    """3 examples, 1 batch."""
    head = head_lib.BinaryClassHead(
        weight_column='label_weights')

    # Create estimator spec.
    logits = np.array(((45,), (-41,), (44,)), dtype=np.float32)
    features = {
        'x': np.array(((42.,), (43.,), (44.,)), dtype=np.float32),
        'label_weights': np.array(((1.,), (.1,), (1.5,)), dtype=np.float64),
    }
    labels = np.array(((1.,), (1.,), (0.,)))
    expected_train_result = b'my_train_op'
    # losses = label_weights*cross_entropy(labels, logits)
    #        = (1*0 + .1*41 + 1.5*44) = (1, 4.1, 66)
    # loss = sum(losses) / batch_size = (1 + 4.1 + 66) / 3 = 23.366666667
    expected_loss = 23.366666667
    if context.executing_eagerly():
      loss = head.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=model_fn.ModeKeys.TRAIN)
      self.assertIsNotNone(loss)
      self.assertAllClose(expected_loss, loss)
      return

    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)
    # Assert spec contains expected tensors.
    self.assertIsNotNone(spec.loss)
    self.assertIsNotNone(spec.train_op)
    # Assert predictions, loss, and metrics.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str = sess.run((
          spec.loss, spec.train_op, spec.scaffold.summary_op))
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)
      test_lib._assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss
      }, summary_str)

  def test_multi_dim_weighted_train_create_loss(self):
    """Logits and labels of shape [2, 2, 1], weights [2, 2]."""
    head = head_lib.BinaryClassHead(
        weight_column='weights')

    logits = np.array([[[10], [-10]], [[12], [-12]]], dtype=np.float32)
    labels = np.array([[[0], [0]], [[1], [1]]], dtype=np.float64)
    weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)
    features = {'weights': weights}
    # unreduced_loss = cross_entropy(labels, logits) = [[10, 0], [0, 12]].
    # Weights are reshaped to [2, 2, 1] to match logits.
    # training_loss = (1*10 + 1.5*0 + 2*0 + 2.5*12) / 2*2 = 40 / 4 = 10
    expected_training_loss = 10.
    tol = 1e-2
    # Create loss.
    if context.executing_eagerly():
      training_loss = head.loss(logits, labels, features,
                                mode=model_fn.ModeKeys.TRAIN)
      self.assertAllClose(
          expected_training_loss, training_loss, rtol=tol, atol=tol)
      return

    training_loss = head.loss(logits, labels, features)
    with self.cached_session():
      test_lib._initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)

  def test_multi_dim_weighted_train(self):
    """Logits and labels of shape [2, 2, 1], weights [2, 2]."""
    head = head_lib.BinaryClassHead(
        weight_column='weights')

    logits = np.array([[[10], [-10]], [[12], [-12]]], dtype=np.float32)
    labels = np.array([[[0], [0]], [[1], [1]]], dtype=np.float64)
    weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)
    features = {'weights': weights}
    # losses = cross_entropy(labels, logits) = [[10, 0], [0, 12]].
    # weighted_sum_loss = 1*10 + 1.5*0 + 2*0 + 2.5*12 = 40
    # loss = weighted_sum_loss / batch_size = 40 / (2*2) = 10
    expected_loss = 10.
    tol = 1e-2
    # Create loss.
    if context.executing_eagerly():
      training_loss = head.loss(logits, labels, features,
                                mode=model_fn.ModeKeys.TRAIN)
      self.assertAllClose(
          expected_loss, training_loss, rtol=tol, atol=tol)
      return

    expected_train_result = 'my_train_op'
    def _train_op_fn(loss):
      return string_ops.string_join(
          [constant_op.constant(expected_train_result),
           string_ops.as_string(loss, precision=2)])
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
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
    """Logits and labels of shape [2, 2, 1], weights [2, 1]."""
    head = head_lib.BinaryClassHead(
        weight_column='weights')

    logits = np.array([[[10], [-10]], [[12], [-12]]], dtype=np.float32)
    labels = np.array([[[0], [0]], [[1], [1]]], dtype=np.float64)
    weights = np.array([[1.], [2.]], dtype=np.float32)
    if context.executing_eagerly():
      with self.assertRaisesRegexp(ValueError, 'weights shape'):
        head.loss(
            logits=logits,
            labels=labels,
            features={'weights': weights},
            mode=model_fn.ModeKeys.TRAIN)
      return

    def _no_op_train_fn(loss):
      del loss
      return control_flow_ops.no_op()
    spec = head.create_estimator_spec(
        features={'weights': weights},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_no_op_train_fn)
    with self.cached_session():
      test_lib._initialize_variables(self, monitored_session.Scaffold())
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[logits_shape: \] \[2 2 1\] \[weights_shape: \] \[2 1\]'):
        spec.loss.eval()

  def test_multi_dim_train_weights_wrong_outer_dim(self):
    """Logits and labels of shape [2, 2, 1], weights [2, 2, 2]."""
    head = head_lib.BinaryClassHead(weight_column='weights')
    logits = np.array([[[10], [-10]], [[12], [-12]]], dtype=np.float32)
    labels = np.array([[[0], [0]], [[1], [1]]], dtype=np.float64)
    weights = np.array([[[1., 1.1], [1.5, 1.6]], [[2., 2.1], [2.5, 2.6]]])
    if context.executing_eagerly():
      with self.assertRaisesRegexp(ValueError, 'weights shape'):
        head.loss(
            logits=logits,
            labels=labels,
            features={'weights': weights},
            mode=model_fn.ModeKeys.TRAIN)
      return
    weights_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    def _no_op_train_fn(loss):
      del loss
      return control_flow_ops.no_op()
    spec = head.create_estimator_spec(
        features={'weights': weights_placeholder},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_no_op_train_fn)
    with self.cached_session():
      test_lib._initialize_variables(self, monitored_session.Scaffold())
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[logits_shape: \]\s\[2 2 1\]\s\[weights_shape: \]\s\[2 2 2\]'):
        spec.loss.eval({weights_placeholder: weights})

  def test_multi_dim_weighted_eval(self):
    """Logits and labels of shape [2, 2, 1], weights [2, 2]."""
    head = head_lib.BinaryClassHead(weight_column='weights')

    logits = np.array([[[10], [-10]], [[12], [-12]]], dtype=np.float32)
    labels = np.array([[[0], [0]], [[1], [1]]], dtype=np.float64)
    weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)
    # losses = cross_entropy(labels, logits) = [[10, 0], [0, 12]].
    # weighted_sum_loss = 1*10 + 1.5*0 + 2*0 + 2.5*12 = 40
    # loss = weighted_sum_loss / batch_size = 40 / (2*2) = 10.
    weighted_sum_loss = 40.
    expected_loss = 10.
    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: weighted_sum_loss / np.sum(weights),
        keys.ACCURACY: (1.*0. + 1.5*1. + 2.*1. + 2.5*0.) / np.sum(weights),
        keys.PRECISION: 2.0/3.0,
        keys.RECALL: 2.0/4.5,
        keys.PREDICTION_MEAN: (1.*1 + 1.5*0 + 2.*1 + 2.5*0) / np.sum(weights),
        keys.LABEL_MEAN: (1.*0 + 1.5*0 + 2.*1 + 2.5*1) / np.sum(weights),
        # TODO(b/118843532): update metrics
        # keys.ACCURACY_BASELINE:
        #     (1.*0 + 1.5*0 + 2.*1 + 2.5*1) / np.sum(weights),
        # keys.AUC: 0.5222,
        # keys.AUC_PR: 0.7341,
    }
    tol = 1e-2
    if context.executing_eagerly():
      loss = head.loss(
          logits=logits,
          labels=labels,
          features={'weights': weights},
          mode=model_fn.ModeKeys.TRAIN)
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
          rtol=tol, atol=tol)
      return

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features={'weights': weights},
        mode=model_fn.ModeKeys.EVAL,
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
          rtol=tol, atol=tol)


class BinaryClassHeadForEstimator(test.TestCase):
  """Tests for create_estimator_spec running in Graph mode only."""

  def test_train_with_optimizer(self):
    head = head_lib.BinaryClassHead()

    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.float64)
    expected_train_result = b'my_train_op'
    features = {'x': np.array(((42,),), dtype=np.float32)}
    # loss = sum(cross_entropy(labels, logits)) / batch_size
    #      = sum(0, 41) / 2 = 41 / 2 = 20.5
    expected_loss = 20.5

    class _Optimizer(object):

      def minimize(self, loss, global_step):
        del global_step
        with ops.control_dependencies((check_ops.assert_equal(
            math_ops.to_float(expected_loss), math_ops.to_float(loss),
            name='assert_loss'),)):
          return constant_op.constant(expected_train_result)

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        optimizer=_Optimizer())

    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)

  def test_train_with_update_ops(self):
    head = head_lib.BinaryClassHead()

    with ops.Graph().as_default():
      w = variables.Variable(1)
      update_op = w.assign_add(1)
      ops.add_to_collection(ops.GraphKeys.UPDATE_OPS, update_op)

      t = variables.Variable('')
      expected_train_result = b'my_train_op'
      def _train_op_fn(loss):
        del loss
        return t.assign(expected_train_result)

      spec = head.create_estimator_spec(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=model_fn.ModeKeys.TRAIN,
          logits=np.array(((45,), (-41,),), dtype=np.float32),
          labels=np.array(((1,), (1,),), dtype=np.float64),
          train_op_fn=_train_op_fn)

      with self.cached_session() as sess:
        test_lib._initialize_variables(self, spec.scaffold)
        sess.run(spec.train_op)
        w_value, t_value = sess.run([w, t])
        self.assertEqual(2, w_value)
        self.assertEqual(expected_train_result, t_value)

  def test_train_summaries_with_head_name(self):
    head = head_lib.BinaryClassHead(
        name='some_binary_head')

    logits = np.array(((45,), (-41,),), dtype=np.float32)
    labels = np.array(((1,), (1,),), dtype=np.float64)
    features = {'x': np.array(((42,),), dtype=np.float32)}
    # loss = sum(cross_entropy(labels, logits)) / batch_size
    #      = sum(0, 41) / 2 = 20.5
    expected_loss = 20.5

    def _train_op_fn(loss):
      del loss
      return control_flow_ops.no_op()

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)
    # Assert summaries.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      summary_str = sess.run(spec.scaffold.summary_op)
      test_lib._assert_simple_summaries(
          self,
          {'{}/some_binary_head'.format(metric_keys.MetricKeys.LOSS):
               expected_loss,
          }, summary_str)

  def test_lookup_tables_in_graph(self):
    head = head_lib.BinaryClassHead(
        label_vocabulary=['aang', 'iroh'])

    feature_columns = [feature_column.numeric_column('x')]
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
    self.assertIn(
        metric_keys.MetricKeys.LOSS_MEAN, six.iterkeys(eval_results))
    # Predict.
    est.predict(input_fn)


if __name__ == '__main__':
  test.main()
