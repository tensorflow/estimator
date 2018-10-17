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
"""Tests for head_v2.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.framework import summary_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import monitored_session
from tensorflow.python.training import queue_runner_impl
from tensorflow_estimator.python.estimator import head_v2 as head_lib
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.inputs import numpy_io


def _initialize_variables(test_case, scaffold):
  scaffold.finalize()
  test_case.assertIsNone(scaffold.init_feed_dict)
  test_case.assertIsNone(scaffold.init_fn)
  scaffold.init_op.run()
  scaffold.ready_for_local_init_op.eval()
  scaffold.local_init_op.run()
  scaffold.ready_op.eval()
  test_case.assertIsNotNone(scaffold.saver)


def _assert_simple_summaries(test_case, expected_summaries, summary_str,
                             tol=1e-6):
  """Assert summary the specified simple values.

  Args:
    test_case: test case.
    expected_summaries: Dict of expected tags and simple values.
    summary_str: Serialized `summary_pb2.Summary`.
    tol: Tolerance for relative and absolute.
  """
  summary = summary_pb2.Summary()
  summary.ParseFromString(summary_str)
  test_case.assertAllClose(expected_summaries, {
      v.tag: v.simple_value for v in summary.value
  }, rtol=tol, atol=tol)


def _assert_no_hooks(test_case, spec):
  test_case.assertAllEqual([], spec.training_chief_hooks)
  test_case.assertAllEqual([], spec.training_hooks)


class CreateEstimatorSpecTest(test.TestCase):

  class _HeadWithTPUSupport(head_lib.Head):
    """Head that overrides _create_tpu_estimator_spec."""

    def name(self):
      return 'HeadWithTPUSupport'

    def logits_dimension(self):
      return None

    def loss_reduction(self):
      return None

    def loss(self, features, mode, logits, labels):
      return None

    def predictions(self, logits):
      return None

    def metrics(self, regularization_losses=None):
      return None

    def update_metrics(self, eval_metrics, features, logits, labels,
                       mode=None, regularization_losses=None):
      return None

    def _create_tpu_estimator_spec(self, features, mode, logits, labels=None,
                                   optimizer=None, train_op_fn=None,
                                   regularization_losses=None):
      return model_fn._TPUEstimatorSpec(
          mode=model_fn.ModeKeys.EVAL,
          loss=constant_op.constant(0.0, dtype=dtypes.float32))

  class _HeadWithOutTPUSupport(head_lib.Head):
    """Head that overrides create_estimator_spec."""

    def name(self):
      return 'HeadWithOutTPUSupport'

    def logits_dimension(self):
      return None

    def loss_reduction(self):
      return None

    def loss(self, features, mode, logits, labels):
      return None

    def predictions(self, logits):
      return None

    def metrics(self, regularization_losses=None):
      return None

    def update_metrics(self, eval_metrics, features, logits, labels,
                       mode=None, regularization_losses=None):
      return None

    def create_estimator_spec(self, features, mode, logits, labels=None,
                              optimizer=None, train_op_fn=None,
                              regularization_losses=None):
      return model_fn.EstimatorSpec(
          mode=model_fn.ModeKeys.EVAL,
          loss=constant_op.constant(0.0, dtype=dtypes.float32))

  class _InvalidHead(head_lib.Head):
    """Head that overrides neither estimator_spec functions."""

    def name(self):
      return 'InvalidHead'

    def logits_dimension(self):
      return None

    def loss_reduction(self):
      return None

    def loss(self, features, mode, logits, labels):
      return None

    def predictions(self, logits):
      return None

    def metrics(self, regularization_losses=None):
      return None

    def update_metrics(self, eval_metrics, features, logits, labels,
                       mode=None, regularization_losses=None):
      return None

  def test_head_override_tpu_estimator_spec(self):
    """Test for `_Head` that overrides _create_tpu_estimator_spec."""
    head = self._HeadWithTPUSupport()

    tpu_spec = head._create_tpu_estimator_spec(
        features=None, mode=None, logits=None)
    self.assertTrue(isinstance(tpu_spec, model_fn._TPUEstimatorSpec))
    est_spec = head.create_estimator_spec(
        features=None, mode=None, logits=None)
    self.assertTrue(isinstance(est_spec, model_fn.EstimatorSpec))

  def test_head_override_estimator_spec(self):
    """Test for `_Head` that overrides create_estimator_spec."""
    head = self._HeadWithOutTPUSupport()

    with self.assertRaisesRegexp(
        NotImplementedError,
        'TPUEstimatorSpec not available for this model head.'):
      _ = head._create_tpu_estimator_spec(
          features=None, mode=None, logits=None)
    est_spec = head.create_estimator_spec(
        features=None, mode=None, logits=None)
    self.assertTrue(isinstance(est_spec, model_fn.EstimatorSpec))

  def test_invalid_head_class(self):
    head = self._InvalidHead()

    with self.assertRaisesRegexp(
        NotImplementedError,
        'TPUEstimatorSpec not available for this model head.'):
      _ = head._create_tpu_estimator_spec(
          features=None, mode=None, logits=None)
    with self.assertRaisesRegexp(
        NotImplementedError,
        r'Subclasses of Head must implement `create_estimator_spec\(\)` or '
        r'_create_tpu_estimator_spec\(\).'):
      _ = head.create_estimator_spec(
          features=None, mode=None, logits=None)


@test_util.run_all_in_graph_and_eager_modes
class RegressionHead(test.TestCase):

  def test_invalid_label_dimension(self):
    with self.assertRaisesRegexp(ValueError, r'Invalid label_dimension'):
      head_lib._regression_head(label_dimension=-1)
    with self.assertRaisesRegexp(ValueError, r'Invalid label_dimension'):
      head_lib._regression_head(label_dimension=0)

  def test_invalid_loss_reduction(self):
    with self.assertRaisesRegexp(
        ValueError, r'Invalid loss_reduction: invalid_loss_reduction'):
      head_lib._regression_head(loss_reduction='invalid_loss_reduction')
    with self.assertRaisesRegexp(
        ValueError, r'Invalid loss_reduction: none'):
      head_lib._regression_head(loss_reduction=losses.Reduction.NONE)

  def test_loss_fn_arg_labels_missing(self):
    def _loss_fn(logits):
      del logits  # Unused
    with self.assertRaisesRegexp(
        ValueError,
        r'loss_fn must contain argument: labels\. '
        r'Given arguments: \(\'logits\',\)'):
      head_lib._regression_head(loss_fn=_loss_fn)

  def test_loss_fn_arg_logits_missing(self):
    def _loss_fn(labels):
      del labels  # unused
    with self.assertRaisesRegexp(
        ValueError,
        r'loss_fn must contain argument: logits\. '
        r'Given arguments: \(\'labels\',\)'):
      head_lib._regression_head(loss_fn=_loss_fn)

  def test_loss_fn_arg_features_ok(self):
    def _loss_fn(labels, logits, features):
      del labels, logits, features  # Unused
      head_lib._regression_head(loss_fn=_loss_fn)

  def test_loss_fn_arg_invalid(self):
    def _loss_fn(labels, logits, name=None):
      del labels, logits, name  # Unused
    with self.assertRaisesRegexp(
        ValueError,
        r'loss_fn has unexpected args: \[\'name\'\]'):
      head_lib._regression_head(loss_fn=_loss_fn)

  def test_invalid_logits(self):
    """Label dimension is 3, logits shape [1, 2, 1]."""
    head = head_lib._regression_head(label_dimension=3)
    self.assertEqual(3, head.logits_dimension)
    logits_1d = np.array(((45.,), (41.,),))

    # Static shape.
    with self.assertRaisesRegexp(ValueError, 'logits shape'):
      pred = head.predictions(logits_1d)
      self.evaluate(pred[prediction_keys.PredictionKeys.PREDICTIONS])
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
        spec.predictions[prediction_keys.PredictionKeys.PREDICTIONS].eval({
            logits_placeholder: logits_1d
        })

  def test_incompatible_labels_eval(self):
    head = head_lib._regression_head(label_dimension=3)
    self.assertEqual(3, head.logits_dimension)
    values_3d = np.array(((45., 46., 47.), (41., 42., 43.),))
    values_1d = np.array(((43.,), (44.,),))

    # Static shape.
    if context.executing_eagerly():
      with self.assertRaisesRegexp(ValueError, 'incompatible'):
        head.loss(
            logits=values_3d,
            labels=values_1d,
            features={'x': values_1d},
            mode=model_fn.ModeKeys.EVAL)
      return

    # Dynamic shape only works in Graph mode.
    with self.assertRaisesRegexp(ValueError, 'logits shape'):
      head.create_estimator_spec(
          features={'x': values_3d}, labels=values_3d,
          mode=model_fn.ModeKeys.EVAL, logits=values_1d, train_op_fn=None)
    labels_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    spec = head.create_estimator_spec(
        features={'x': values_1d},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits_placeholder,
        labels=labels_placeholder)
    with self.cached_session():
      with self.assertRaisesRegexp(errors.OpError, 'logits shape'):
        spec.loss.eval({
            labels_placeholder: values_3d,
            logits_placeholder: values_1d
        })
    regularized_training_loss = head.loss(
        logits=logits_placeholder,
        labels=labels_placeholder,
        features={'x': values_1d},
        mode=model_fn.ModeKeys.EVAL)
    with self.cached_session():
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[expected_labels_shape: \] \[2 3\] \[labels_shape: \] \[2 1\]'):
        regularized_training_loss.eval({
            labels_placeholder: values_1d,
            logits_placeholder: values_3d
        })

  def test_incompatible_labels_train(self):
    head = head_lib._regression_head(label_dimension=3)
    self.assertEqual(3, head.logits_dimension)
    values_3d = np.array(((45., 46., 47.), (41., 42., 43.),))  # shape [2, 3]
    values_1d = np.array(((43.,), (44.,),))  # shape [2, 1]

    # Static shape.
    if context.executing_eagerly():
      with self.assertRaisesRegexp(ValueError, 'incompatible'):
        head.loss(
            logits=values_3d,
            labels=values_1d,
            features={'x': values_1d},
            mode=model_fn.ModeKeys.TRAIN)
      return

    # Dynamic shape only works in Graph mode.
    with self.assertRaisesRegexp(ValueError, 'logits shape'):
      head.create_estimator_spec(
          features={'x': values_3d},
          mode=model_fn.ModeKeys.TRAIN,
          logits=values_1d,
          labels=values_3d,
          train_op_fn=lambda x: x)
    labels_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    spec = head.create_estimator_spec(
        features={'x': values_1d},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits_placeholder,
        labels=labels_placeholder,
        train_op_fn=lambda x: x)
    with self.cached_session():
      with self.assertRaisesRegexp(errors.OpError, 'logits shape'):
        spec.loss.eval({
            labels_placeholder: values_3d,
            logits_placeholder: values_1d
        })
    regularized_training_loss = head.loss(
        logits=logits_placeholder,
        labels=labels_placeholder,
        features={'x': values_1d},
        mode=model_fn.ModeKeys.TRAIN)
    with self.cached_session():
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[expected_labels_shape: \] \[2 3\] \[labels_shape: \] \[2 1\]'):
        regularized_training_loss.eval({
            labels_placeholder: values_1d,
            logits_placeholder: values_3d
        })

  def test_predict(self):
    head = head_lib._regression_head()
    self.assertEqual(1, head.logits_dimension)

    logits = np.array(((45,), (41,),), dtype=np.int32)
    preds = head.predictions(logits)

    prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
    self.assertItemsEqual((prediction_key,), preds.keys())
    predictions = preds[prediction_key]
    self.assertEqual(dtypes.float32, predictions.dtype)
    self.assertAllClose(logits, self.evaluate(predictions))

    if context.executing_eagerly():
      return
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features={'x': np.array(((42.,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.PREDICT,
        logits=np.array(((45,), (41,),), dtype=np.int32))
    self.assertIsNone(spec.loss)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNone(spec.train_op)
    default_serving_key = (
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
    self.assertItemsEqual(
        (default_serving_key, 'predict', 'regression'),
        spec.export_outputs.keys())
    _assert_no_hooks(self, spec)
    # Assert predictions.
    with self.cached_session():
      _initialize_variables(self, spec.scaffold)
      self.assertAllClose(
          logits, spec.export_outputs[default_serving_key].value.eval())
      self.assertAllClose(
          logits, spec.export_outputs['regression'].value.eval())
      self.assertAllClose(
          logits,
          spec.export_outputs['predict'].outputs['predictions'].eval())

  def test_predict_with_inverse_link_fn(self):
    def _inverse_link_fn(logits):
      return logits - 10.
    head = head_lib._regression_head(inverse_link_fn=_inverse_link_fn)

    logits = np.array(((45,), (41,),), dtype=np.int32)
    preds = head.predictions(logits)

    keys = prediction_keys.PredictionKeys
    self.assertItemsEqual(
        (keys.PREDICTIONS, keys.LOGITS), preds.keys())
    self.assertEqual(dtypes.float32, preds[keys.PREDICTIONS].dtype)
    self.assertEqual(dtypes.float32, preds[keys.LOGITS].dtype)

    expected_predictions = np.array(((35,), (31,),), dtype=np.int32)
    self.assertAllClose(
        expected_predictions, self.evaluate(preds[keys.PREDICTIONS]))
    self.assertAllClose(logits, self.evaluate(preds[keys.LOGITS]))

    if context.executing_eagerly():
      return
    # Create estimator spec.
    spec = head.create_estimator_spec(
        features={'x': np.array(((42.,),), dtype=np.int32)},
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits)
    # Assert spec contains expected tensors.
    default_serving_key = (
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
    self.assertItemsEqual(
        (default_serving_key, 'predict', 'regression'),
        spec.export_outputs.keys())
    # Assert predictions.
    with self.cached_session():
      _initialize_variables(self, spec.scaffold)
      self.assertAllClose(
          expected_predictions,
          spec.export_outputs[default_serving_key].value.eval())
      self.assertAllClose(
          expected_predictions,
          spec.export_outputs['regression'].value.eval())
      self.assertAllClose(
          expected_predictions,
          spec.export_outputs['predict'].outputs['predictions'].eval())
      self.assertAllClose(
          logits, spec.export_outputs['predict'].outputs['logits'].eval())

  def test_eval_create_loss(self):
    head = head_lib._regression_head()
    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43,), (44,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.float32)}

    regularized_training_loss = head.loss(
        logits=logits,
        labels=labels,
        features=features)
    self.assertAllClose(13., self.evaluate(regularized_training_loss))

  def test_eval_create_loss_loss_fn(self):
    """Tests head.loss for eval mode and custom loss_fn."""
    loss = np.array([[0., 1.], [2., 3.]], dtype=np.float32)
    logits_input = np.array([[-1., 1.], [-2., 2.]], dtype=np.float32)
    labels_input = np.array([[1., 0.], [2., -1.]], dtype=np.float32)
    def _loss_fn(labels, logits):
      check_labels = control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(labels, labels_input)),
          data=[labels])
      check_logits = control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(logits, logits_input)),
          data=[logits])
      with ops.control_dependencies([check_labels, check_logits]):
        return constant_op.constant(loss)
    head = head_lib._regression_head(label_dimension=2, loss_fn=_loss_fn)
    regularized_training_loss = head.loss(
        logits=logits_input,
        labels=labels_input,
        features={'x': np.array(((42,),), dtype=np.int32)})
    self.assertAllClose(np.sum(loss), self.evaluate(regularized_training_loss))

  def test_eval_create_loss_loss_fn_wrong_shape(self):
    """Tests custom loss_fn that returns Tensor of unexpected shape."""
    loss = np.array([[1.], [2.]], dtype=np.float32)
    def _loss_fn(labels, logits):
      del labels, logits  # Unused
      return constant_op.constant(loss)
    head = head_lib._regression_head(label_dimension=2, loss_fn=_loss_fn)

    features = {'x': np.array(((42,),), dtype=np.int32)}
    logits = np.array([[-1., 1.], [-2., 2.]], dtype=np.float32)
    labels = np.array([[1., 0.], [2., -1.]], dtype=np.float32)

    if context.executing_eagerly():
      with self.assertRaisesRegexp(
          ValueError, 'loss_shape'):
        head.loss(
            logits=logits,
            labels=labels,
            features=features)
    else:
      regularized_training_loss = head.loss(
          logits=logits,
          labels=labels,
          features=features)
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[loss_fn must return Tensor of shape \[D0, D1, ... DN, 2\]\. \] '
          r'\[logits_shape: \] \[2 2\] \[loss_shape: \] \[2 1\]'):
        self.evaluate(regularized_training_loss)

  def test_eval_labels_none(self):
    """Tests that error is raised when labels is None."""
    head = head_lib._regression_head()

    with self.assertRaisesRegexp(
        ValueError, r'You must provide a labels Tensor\. Given: None\.'):
      head.create_estimator_spec(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=model_fn.ModeKeys.EVAL,
          logits=np.array(((45,), (41,),), dtype=np.float32),
          labels=None)

  def test_eval(self):
    head = head_lib._regression_head()
    self.assertEqual(1, head.logits_dimension)

    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43,), (44,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.float32)}

    prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
    preds = head.predictions(logits)
    self.assertItemsEqual((prediction_key,), preds.keys())
    self.assertEqual(dtypes.float32, preds[prediction_key].dtype)
    self.assertAllClose(logits, self.evaluate(preds[prediction_key]))

    # loss_mean = loss/2 = 13/2 = 6.5
    expected_loss_mean = 6.5
    if context.executing_eagerly():
      eval_metrics = head.metrics()
      update_metrics = head.update_metrics(
          eval_metrics, features, logits, labels)
      self.assertItemsEqual((metric_keys.MetricKeys.LOSS_MEAN,
                             metric_keys.MetricKeys.PREDICTION_MEAN,
                             metric_keys.MetricKeys.LABEL_MEAN),
                            update_metrics.keys())
      self.assertAllClose(
          expected_loss_mean,
          update_metrics[metric_keys.MetricKeys.LOSS_MEAN].result())
    else:
      # Create estimator spec.
      spec = head.create_estimator_spec(
          features=features,
          mode=model_fn.ModeKeys.EVAL,
          logits=logits,
          labels=labels)
      # Assert spec contains expected tensors.
      self.assertEqual(dtypes.float32, spec.loss.dtype)
      self.assertItemsEqual((metric_keys.MetricKeys.LOSS_MEAN,
                             metric_keys.MetricKeys.PREDICTION_MEAN,
                             metric_keys.MetricKeys.LABEL_MEAN),
                            spec.eval_metric_ops.keys())
      self.assertIsNone(spec.train_op)
      self.assertIsNone(spec.export_outputs)
      _assert_no_hooks(self, spec)
      # Assert predictions, loss, and metrics.
      with self.cached_session() as sess:
        _initialize_variables(self, spec.scaffold)
        self.assertIsNone(spec.scaffold.summary_op)
        loss_mean_value_op, loss_mean_update_op = spec.eval_metric_ops[
            metric_keys.MetricKeys.LOSS_MEAN]
        loss, _ = sess.run((spec.loss, loss_mean_update_op))
        # loss = (43-45)^2 + (44-41)^2 = 4+9 = 13
        self.assertAllClose(13., loss)
        # Check results of value ops (in `loss_mean`).
        self.assertAllClose(expected_loss_mean, loss_mean_value_op.eval())

  def test_eval_metric_ops_with_head_name_for_regression(self):
    head = head_lib._regression_head(name='some_regression_head')
    logits = np.array(((1,), (9,)), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}

    expected_metric_keys = [
        '{}/some_regression_head'.format(metric_keys.MetricKeys.LOSS_MEAN),
        '{}/some_regression_head'.format(
            metric_keys.MetricKeys.PREDICTION_MEAN),
        '{}/some_regression_head'.format(metric_keys.MetricKeys.LABEL_MEAN),
    ]
    eval_metrics = head.metrics()
    updated_metrics = head.update_metrics(
        eval_metrics, features, logits, labels)
    self.assertItemsEqual(expected_metric_keys, updated_metrics.keys())

  def test_eval_with_regularization_losses(self):
    head = head_lib._regression_head(
        loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)
    self.assertEqual(1, head.logits_dimension)

    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43,), (44,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.float32)}
    regularization_losses = [1.5, 0.5]
    expected_regularization_loss = 2.
    # unregularized_loss = ((43-45)^2 + (44-41)^2) / batch_size
    #                    = (4 + 9) / 2 = 6.5
    expected_unregularized_loss = 6.5
    expected_regularized_loss = (
        expected_unregularized_loss + expected_regularization_loss)

    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS_MEAN: expected_unregularized_loss,
        keys.LOSS_REGULARIZATION: expected_regularization_loss,
        keys.PREDICTION_MEAN: (45 + 41) / 2.0,
        keys.LABEL_MEAN: (43 + 44) / 2.0,
    }
    # Test eval metrics in eager mode
    if context.executing_eagerly():
      eval_metrics = head.metrics(regularization_losses=regularization_losses)
      updated_metrics = head.update_metrics(
          eval_metrics, features, logits, labels,
          regularization_losses=regularization_losses)
      # Assert metrics.
      self.assertAllClose(
          expected_metrics,
          {k: updated_metrics[k].result() for k in updated_metrics})
    else:
      # Create estimator spec.
      spec = head.create_estimator_spec(
          features=features,
          mode=model_fn.ModeKeys.EVAL,
          logits=logits,
          labels=labels,
          regularization_losses=regularization_losses)
      # Assert predictions, loss, and metrics.
      with self.cached_session() as sess:
        _initialize_variables(self, spec.scaffold)
        self.assertIsNone(spec.scaffold.summary_op)
        value_ops = {
            k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
        update_ops = {
            k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
        prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
        predictions, loss, _ = sess.run((
            spec.predictions[prediction_key], spec.loss, update_ops))
        self.assertAllClose(logits, predictions)
        self.assertAllClose(expected_regularized_loss, loss)
        # Check results of value ops (in `metrics`).
        self.assertAllClose(
            expected_metrics, {k: value_ops[k].eval() for k in value_ops})

  def test_train_create_loss(self):
    head = head_lib._regression_head()
    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43,), (44,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.float32)}
    # training_loss = 1 * 4 + 1 * 9 = 13
    expected_training_loss = 13.
    # Create loss.
    training_loss = head.loss(
        logits=logits,
        labels=labels,
        features=features,
        mode=model_fn.ModeKeys.TRAIN)
    self.assertAllClose(expected_training_loss, self.evaluate(training_loss))

  def test_train_create_loss_loss_reduction(self):
    """Tests create_loss with loss_reduction."""
    head = head_lib._regression_head(
        loss_reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43,), (44,),), dtype=np.int32)
    features = {'x': np.array(((42,),), dtype=np.float32)}
    # training_loss = (1 * 4 + 1 * 9) / num_nonzero_weights
    expected_training_loss = 13. / 2.
    # Create loss.
    training_loss = head.loss(
        logits=logits,
        labels=labels,
        features=features,
        mode=model_fn.ModeKeys.TRAIN)
    self.assertAllClose(expected_training_loss, self.evaluate(training_loss))

  def test_train_labels_none(self):
    """Tests that error is raised when labels is None."""
    head = head_lib._regression_head()
    def _no_op_train_fn(loss):
      del loss
      return control_flow_ops.no_op()

    with self.assertRaisesRegexp(
        ValueError, r'You must provide a labels Tensor\. Given: None\.'):
      head.create_estimator_spec(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=model_fn.ModeKeys.TRAIN,
          logits=np.array(((45,), (41,),), dtype=np.float32),
          labels=None,
          train_op_fn=_no_op_train_fn)

  def test_train(self):
    head = head_lib._regression_head()
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43.,), (44.,),), dtype=np.float64)
    expected_train_result = b'my_train_op'
    features = {'x': np.array(((42.,),), dtype=np.float32)}
    # loss = (43-45)^2 + (44-41)^2 = 4 + 9 = 13
    expected_loss = 13
    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)

    preds = head.predictions(logits)
    loss = head.loss(logits, labels, features=features)
    prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
    self.assertItemsEqual((prediction_key,), preds.keys())
    self.assertEqual(dtypes.float32, preds[prediction_key].dtype)
    self.assertEqual(dtypes.float32, loss.dtype)
    self.assertAllClose(logits, self.evaluate(preds[prediction_key]))
    self.assertAllClose(expected_loss, self.evaluate(loss))
    if context.executing_eagerly():
      return

    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)

    # Assert spec contains expected tensors.
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, train_op, and summaries.
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      predictions, loss, train_result, summary_str = sess.run((
          spec.predictions[prediction_key], spec.loss, spec.train_op,
          spec.scaffold.summary_op))
      self.assertAllClose(logits, predictions)
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)
      _assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
          # loss_mean = loss/2 = 13/2 = 6.5
          metric_keys.MetricKeys.LOSS_MEAN: 6.5,
      }, summary_str)

  def test_train_with_regularization_losses(self):
    head = head_lib._regression_head(
        loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)
    self.assertEqual(1, head.logits_dimension)

    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43.,), (44.,),), dtype=np.float64)
    expected_train_result = b'my_train_op'
    features = {'x': np.array(((42.,),), dtype=np.float32)}
    regularization_losses = [1.5, 0.5]
    expected_regularization_loss = 2.
    # unregularized_loss = ((43-45)^2 + (44-41)^2) / batch_size
    #                    = (4 + 9) / 2 = 6.5
    # loss = unregularized_loss + regularization_loss = 8.5
    expected_loss = 8.5
    prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
    loss = head.loss(logits, labels, features=features,
                     mode=model_fn.ModeKeys.TRAIN,
                     regularization_losses=regularization_losses)
    preds = head.predictions(logits)
    self.assertAllClose(logits, self.evaluate(preds[prediction_key]))
    self.assertAllClose(expected_loss, self.evaluate(loss))
    if context.executing_eagerly():
      return

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
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      predictions, loss, train_result, summary_str = sess.run((
          spec.predictions[prediction_key], spec.loss, spec.train_op,
          spec.scaffold.summary_op))
      self.assertAllClose(logits, predictions)
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)
      _assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
          metric_keys.MetricKeys.LOSS_REGULARIZATION: (
              expected_regularization_loss),
      }, summary_str)

  def test_weighted_multi_example_eval(self):
    """1d label, 3 examples, 1 batch."""
    head = head_lib._regression_head(weight_column='label_weights')
    self.assertEqual(1, head.logits_dimension)
    logits = np.array(((45,), (41,), (44,)), dtype=np.int32)
    features = {
        'x': np.array(((42,), (43,), (44,)), dtype=np.int32),
        'label_weights': np.array(((1.,), (.1,), (1.5,)), dtype=np.float32),
    }
    labels = np.array(((35,), (42,), (45,)), dtype=np.int32)

    prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
    preds = head.predictions(logits)
    self.assertItemsEqual((prediction_key,), preds.keys())
    predictions = preds[prediction_key]
    self.assertEqual(dtypes.float32, predictions.dtype)
    self.assertAllClose(logits, self.evaluate(predictions))

    # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.0769231
    expected_loss_mean = 39.0769231
    if context.executing_eagerly():
      eval_metrics = head.metrics()
      updated_metrics = head.update_metrics(
          eval_metrics, features, logits, labels)
      self.assertItemsEqual((metric_keys.MetricKeys.LOSS_MEAN,
                             metric_keys.MetricKeys.PREDICTION_MEAN,
                             metric_keys.MetricKeys.LABEL_MEAN),
                            updated_metrics.keys())
      self.assertAllClose(
          expected_loss_mean,
          updated_metrics[metric_keys.MetricKeys.LOSS_MEAN].result())
    else:
      # Create estimator spec.
      spec = head.create_estimator_spec(
          features=features,
          mode=model_fn.ModeKeys.EVAL,
          logits=logits,
          labels=labels)
      # Assert spec contains expected tensors.
      self.assertEqual(dtypes.float32, spec.loss.dtype)
      self.assertIsNone(spec.train_op)
      self.assertIsNone(spec.export_outputs)
      _assert_no_hooks(self, spec)
      # Assert predictions, loss, and metrics.
      with self.cached_session() as sess:
        _initialize_variables(self, spec.scaffold)
        self.assertIsNone(spec.scaffold.summary_op)
        loss_mean_value_op, loss_mean_update_op = spec.eval_metric_ops[
            metric_keys.MetricKeys.LOSS_MEAN]
        loss, _ = sess.run((spec.loss, loss_mean_update_op))
        # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
        self.assertAllClose(101.6, loss)
        # Check results of value ops (in `loss_mean`).
        self.assertAllClose(expected_loss_mean, loss_mean_value_op.eval())

  def test_weight_with_numeric_column(self):
    """1d label, 3 examples, 1 batch."""
    head = head_lib._regression_head(
        weight_column=feature_column_lib.numeric_column(
            'label_weights', normalizer_fn=lambda x: x + 1.))

    logits = np.array(((45,), (41,), (44,)), dtype=np.int32)
    features = {
        'x':
            np.array(((42,), (43,), (44,)), dtype=np.int32),
        'label_weights':
            np.array(((0.,), (-0.9,), (0.5,)), dtype=np.float32),
    }
    labels = np.array(((35,), (42,), (45,)), dtype=np.int32)

    loss = head.loss(logits, labels, features=features)
    # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
    self.assertAllClose(101.6, self.evaluate(loss))

  def test_weighted_multi_example_train(self):
    """1d label, 3 examples, 1 batch."""
    head = head_lib._regression_head(weight_column='label_weights')
    self.assertEqual(1, head.logits_dimension)

    features = {
        'x': np.array(((42,), (43,), (44,)), dtype=np.float32),
        'label_weights': np.array(((1.,), (.1,), (1.5,)), dtype=np.float64),}
    labels = np.array(((35.,), (42.,), (45.,)), dtype=np.float32)
    logits = np.array(((45,), (41,), (44,)), dtype=np.float32)
    expected_train_result = b'my_train_op'
    # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
    expected_loss = 101.6
    preds = head.predictions(logits)
    loss = head.loss(logits, labels, features=features,
                     mode=model_fn.ModeKeys.TRAIN)

    prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
    self.assertItemsEqual((prediction_key,), preds.keys())
    self.assertEqual(dtypes.float32, preds[prediction_key].dtype)
    self.assertEqual(dtypes.float32, loss.dtype)
    self.assertAllClose(logits, self.evaluate(preds[prediction_key]))
    self.assertAllClose(expected_loss, self.evaluate(loss))
    if context.executing_eagerly():
      return

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
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, train_op, and summaries.
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      predictions, loss, train_result, summary_str = sess.run((
          spec.predictions[prediction_key], spec.loss, spec.train_op,
          spec.scaffold.summary_op))
      self.assertAllClose(logits, predictions)
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)
      _assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
          # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.0769231
          metric_keys.MetricKeys.LOSS_MEAN: 39.0769231,
      }, summary_str)

  def test_train_one_dim_create_loss(self):
    """Tests create_loss with 1D labels and weights (shape [batch_size])."""
    head = head_lib._regression_head(weight_column='label_weights')
    logits = np.array(((45,), (41,), (44,)), dtype=np.float32)
    x_feature_rank_1 = np.array((42., 43., 44.,), dtype=np.float32)
    weight_rank_1 = np.array((1., .1, 1.5,), dtype=np.float64)
    labels_rank_1 = np.array((35., 42., 45.,))
    # training_loss = 100 * 1 + 1 * .1 + 1.5 * 1 = 101.6
    expected_training_loss = 101.6
    features = {'x': x_feature_rank_1, 'label_weights': weight_rank_1}
    # Create loss.
    training_loss = head.loss(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels_rank_1)
    self.assertAllClose(expected_training_loss, self.evaluate(training_loss))

  def test_train_one_dim(self):
    """Tests train with 1D labels and weights (shape [batch_size])."""
    head = head_lib._regression_head(weight_column='label_weights')
    self.assertEqual(1, head.logits_dimension)

    logits = np.array(((45,), (41,), (44,)), dtype=np.float32)
    expected_train_result = b'my_train_op'
    # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
    expected_loss = 101.6
    x_feature_rank_1 = np.array((42., 43., 44.,), dtype=np.float32)
    weight_rank_1 = np.array((1., .1, 1.5,), dtype=np.float64)
    labels_rank_1 = np.array((35., 42., 45.,))
    features = {'x': x_feature_rank_1, 'label_weights': weight_rank_1}
    self.assertEqual((3,), x_feature_rank_1.shape)
    self.assertEqual((3,), weight_rank_1.shape)
    self.assertEqual((3,), labels_rank_1.shape)
    preds = head.predictions(logits)
    loss = head.loss(logits, labels=labels_rank_1, features=features,
                     mode=model_fn.ModeKeys.TRAIN)
    prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
    self.assertItemsEqual((prediction_key,), preds.keys())
    self.assertEqual(dtypes.float32, preds[prediction_key].dtype)
    self.assertEqual(dtypes.float32, loss.dtype)
    self.assertAllClose(logits, self.evaluate(preds[prediction_key]))
    self.assertAllClose(expected_loss, self.evaluate(loss))
    if context.executing_eagerly():
      return

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
        labels=labels_rank_1,
        train_op_fn=_train_op_fn)

    # Assert spec contains expected tensors.
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, train_op, and summaries.
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      predictions, loss, train_result, summary_str = sess.run((
          spec.predictions[prediction_key], spec.loss, spec.train_op,
          spec.scaffold.summary_op))
      self.assertAllClose(logits, predictions)
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)
      _assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
          # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.0769231
          metric_keys.MetricKeys.LOSS_MEAN: 39.0769231,
      }, summary_str)

  def test_weighted_multi_value_eval_create_loss(self):
    """3d label, 1 example, 1 batch."""
    head = head_lib._regression_head(
        weight_column='label_weights', label_dimension=3)
    logits = np.array(((45., 41., 44.),))
    labels = np.array(((35., 42., 45.),))
    features = {
        'x': np.array(((42., 43., 44.),)),
        'label_weights': np.array(((1., .1, 1.5),))
    }
    regularized_training_loss = head.loss(
        logits=logits,
        labels=labels,
        features=features)
    # loss = [(35-45)^2, (42-41)^2, (45-44)^2] = [100, 1, 1].
    # weighted sum loss = 1 * 100 + .1 * 1 + 1.5 * 1 = 101.6
    self.assertAllClose(101.6, self.evaluate(regularized_training_loss))

  def test_weighted_multi_value_eval(self):
    """3d label, 1 example, 1 batch."""
    head = head_lib._regression_head(
        weight_column='label_weights', label_dimension=3)
    self.assertEqual(3, head.logits_dimension)

    logits = np.array(((45., 41., 44.),))
    labels = np.array(((35., 42., 45.),))
    features = {
        'x': np.array(((42., 43., 44.),)),
        'label_weights': np.array(((1., .1, 1.5),))
    }

    prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
    preds = head.predictions(logits)
    self.assertItemsEqual((prediction_key,), preds.keys())
    predictions = preds[prediction_key]
    self.assertEqual(dtypes.float32, predictions.dtype)
    self.assertAllClose(logits, self.evaluate(predictions))

    # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.0769231
    expected_loss_mean = 39.0769231
    if context.executing_eagerly():
      eval_metrics = head.metrics()
      updated_metrics = head.update_metrics(
          eval_metrics, features, logits, labels)
      self.assertItemsEqual((metric_keys.MetricKeys.LOSS_MEAN,
                             metric_keys.MetricKeys.PREDICTION_MEAN,
                             metric_keys.MetricKeys.LABEL_MEAN),
                            updated_metrics.keys())
      self.assertAllClose(
          expected_loss_mean,
          updated_metrics[metric_keys.MetricKeys.LOSS_MEAN].result())
    else:
      # Create estimator spec.
      spec = head.create_estimator_spec(
          features=features,
          mode=model_fn.ModeKeys.EVAL,
          logits=logits,
          labels=labels)
      # Assert spec contains expected tensors.
      self.assertEqual(dtypes.float32, spec.loss.dtype)
      self.assertItemsEqual((metric_keys.MetricKeys.LOSS_MEAN,
                             metric_keys.MetricKeys.PREDICTION_MEAN,
                             metric_keys.MetricKeys.LABEL_MEAN),
                            spec.eval_metric_ops.keys())
      self.assertIsNone(spec.train_op)
      self.assertIsNone(spec.export_outputs)
      _assert_no_hooks(self, spec)
      # Assert predictions, loss, and metrics.
      with self.cached_session() as sess:
        _initialize_variables(self, spec.scaffold)
        self.assertIsNone(spec.scaffold.summary_op)
        loss_mean_value_op, loss_mean_update_op = spec.eval_metric_ops[
            metric_keys.MetricKeys.LOSS_MEAN]
        loss, _ = sess.run((spec.loss, loss_mean_update_op))
        # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
        self.assertAllClose(101.6, loss)
        # Check results of value ops (in `loss_mean`).
        self.assertAllClose(expected_loss_mean, loss_mean_value_op.eval())

  def test_weighted_multi_value_train_create_loss(self):
    """3d label, 1 example, 1 batch."""
    head = head_lib._regression_head(
        weight_column='label_weights', label_dimension=3)
    logits = np.array(((45., 41., 44.),))
    labels = np.array(((35., 42., 45.),))
    features = {
        'x': np.array(((42., 43., 44.),)),
        'label_weights': np.array(((1., .1, 1.5),))
    }

    # Create loss.
    regularized_training_loss = head.loss(
        logits=logits,
        labels=labels,
        features=features,
        mode=model_fn.ModeKeys.TRAIN)
    self.assertAllClose(101.6, self.evaluate(regularized_training_loss))

  def test_weighted_multi_value_train(self):
    """3d label, 1 example, 1 batch."""
    head = head_lib._regression_head(
        weight_column='label_weights', label_dimension=3)
    self.assertEqual(3, head.logits_dimension)

    logits = np.array(((45., 41., 44.),))
    labels = np.array(((35., 42., 45.),))
    expected_train_result = b'my_train_op'
    # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
    expected_loss = 101.6
    def _train_op_fn(loss):
      with ops.control_dependencies((check_ops.assert_equal(
          math_ops.to_float(expected_loss), math_ops.to_float(loss),
          name='assert_loss'),)):
        return constant_op.constant(expected_train_result)

    features = {
        'x': np.array(((42., 43., 44.),)),
        'label_weights': np.array(((1., .1, 1.5),)),
    }
    preds = head.predictions(logits)
    loss = head.loss(logits, labels, features=features,
                     mode=model_fn.ModeKeys.TRAIN)
    prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
    self.assertItemsEqual((prediction_key,), preds.keys())
    self.assertEqual(dtypes.float32, preds[prediction_key].dtype)
    self.assertEqual(dtypes.float32, loss.dtype)
    self.assertAllClose(logits, self.evaluate(preds[prediction_key]))
    self.assertAllClose(expected_loss, self.evaluate(loss))
    if context.executing_eagerly():
      return

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)

    # Assert spec contains expected tensors.
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Evaluate predictions, loss, train_op, and summaries.
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      predictions, loss, train_result, summary_str = sess.run((
          spec.predictions[prediction_key], spec.loss, spec.train_op,
          spec.scaffold.summary_op))
      self.assertAllClose(logits, predictions)
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)
      _assert_simple_summaries(self, {
          metric_keys.MetricKeys.LOSS: expected_loss,
          # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.076923
          metric_keys.MetricKeys.LOSS_MEAN: 39.076923,
      }, summary_str)

  def test_weighted_multi_batch_eval_eager(self):
    """1d label, 1 example, 3 batches."""
    with context.eager_mode():
      head = head_lib._regression_head(weight_column='label_weights')
      self.assertEqual(1, head.logits_dimension)

      logits = np.array(((45.,), (41.,), (44.,)))
      features = {
          'x': np.array(((42.,), (43.,), (44.,))),
          'label_weights': np.array(((1.,), (.1,), (1.5,))),
          # 'logits' is not a feature, but we use `tf.data.Dataset` to make it
          # as a `tensor` (required by `update_metrics`), and access it
          # via `features['logits']` in `update_metrics`
          'logits': logits
      }
      labels = np.array(((35.,), (42.,), (45.,)))

      # losses = [1*(35-45)^2, .1*(42-41)^2, 1.5*(45-44)^2] = [100, .1, 1.5]
      # loss = sum(losses) = 100+.1+1.5 = 101.6
      # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.076923
      expected_metrics = {
          metric_keys.MetricKeys.LOSS_MEAN:
              39.076923,
          metric_keys.MetricKeys.PREDICTION_MEAN:
              (45 + 41 * 0.1 + 44 * 1.5) / 2.6,
          metric_keys.MetricKeys.LABEL_MEAN: (35 + 42 * 0.1 + 45 * 1.5) / 2.6,
      }
      dataset = dataset_ops.Dataset.from_tensor_slices((features, labels))
      dataset = dataset.batch(1)
      eval_metrics = head.metrics()
      for (features, labels) in dataset:
        logits = features['logits']
        updated_metrics = head.update_metrics(
            eval_metrics, features, logits, labels)
        # Assert metrics.
      self.assertAllClose(
          expected_metrics,
          {k: updated_metrics[k].result() for k in updated_metrics})

  def test_weighted_multi_batch_train_eager(self):
    """1d label, 1 example, 3 batches."""
    if context.executing_eagerly():
      head = head_lib._regression_head(weight_column='label_weights')
      self.assertEqual(1, head.logits_dimension)

      logits = np.array(((45.,), (41.,), (44.,)))
      features = {
          'x': np.array(((42.,), (43.,), (44.,))),
          'label_weights': np.array(((1.,), (.1,), (1.5,))),
          # 'logits' is not a feature, but we use `tf.data.Dataset` to make it
          # as a `tensor` (required by `update_metrics`), and access it
          # via `features['logits']` in `update_metrics`
          'logits': logits}
      labels = np.array(((35.,), (42.,), (45.,)))
      dataset = dataset_ops.Dataset.from_tensor_slices((features, labels))
      dataset = dataset.batch(1)
      expected_losses = np.array((100, .1, 1.5))
      for (batch, (features, labels)) in enumerate(dataset):
        logits = features['logits']
        loss = head.loss(logits, labels, features=features)
        self.assertAllClose(expected_losses[batch], loss)

  def test_multi_dim_weighted_train_create_loss(self):
    """Logits, labels of shape [2, 2, 3], weight shape [2, 2]."""
    label_dimension = 3
    head = head_lib._regression_head(
        weight_column='label_weights', label_dimension=label_dimension)
    logits = np.array([[[00., 01., 02.], [10., 11., 12.]],
                       [[20., 21., 22.], [30., 31., 32.]]])
    labels = np.array([[[01., 02., 03.], [12., 13., 14.]],
                       [[23., 24., 25.], [34., 35., 36.]]])
    weights = np.array([[1., 1.5], [2., 2.5]])
    expected_training_loss = np.sum(
        np.array([[[1. * x for x in [1., 1., 1.]],
                   [1.5 * x for x in [4., 4., 4.]]],
                  [[2. * x for x in [9., 9., 9.]],
                   [2.5 * x for x in [16., 16., 16.]]]]))

    # Create loss.
    training_loss = head.loss(
        features={'label_weights': weights},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        regularization_losses=None)
    self.assertAllClose(expected_training_loss, self.evaluate(training_loss))

  def test_multi_dim_weighted_train(self):
    """Logits, labels of shape [2, 2, 3], weight shape [2, 2]."""
    head = head_lib._regression_head(
        weight_column='label_weights', label_dimension=3)
    logits = np.array([[[00., 01., 02.], [10., 11., 12.]],
                       [[20., 21., 22.], [30., 31., 32.]]])
    labels = np.array([[[01., 02., 03.], [12., 13., 14.]],
                       [[23., 24., 25.], [34., 35., 36.]]])
    expected_train_result = b'my_train_op'
    features = {
        'label_weights': np.array([[1., 1.5], [2., 2.5]]),
    }
    # loss = 1*3*1^2 + 1.5*3*2^2 + 2*3*3^2 +2.5*3*4^2 = 195
    expected_loss = 195.
    loss = head.loss(logits, labels, features=features,
                     mode=model_fn.ModeKeys.TRAIN)
    self.assertAllClose(expected_loss, self.evaluate(loss))
    if context.executing_eagerly():
      return

    # Create estimator spec.
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
    with self.cached_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(expected_loss, spec.loss.eval())

  def test_multi_dim_train_weights_wrong_inner_dim(self):
    """Logits, labels of shape [2, 2, 3], weight shape [2, 1]."""
    head = head_lib._regression_head(
        weight_column='label_weights', label_dimension=3)
    logits = np.array([[[00., 01., 02.], [10., 11., 12.]],
                       [[20., 21., 22.], [30., 31., 32.]]])
    labels = np.array([[[01., 02., 03.], [12., 13., 14.]],
                       [[23., 24., 25.], [34., 35., 36.]]])
    features = {
        'label_weights': np.array([[1.], [2]]),
    }
    def _no_op_train_fn(loss):
      del loss
      return control_flow_ops.no_op()

    if context.executing_eagerly():
      with self.assertRaisesRegexp(ValueError, 'weights shape'):
        head.loss(
            features=features,
            mode=model_fn.ModeKeys.TRAIN,
            logits=logits,
            labels=labels,
            regularization_losses=None)
      return

    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_no_op_train_fn)
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        r'\[logits_shape: \] \[2 2 3\] \[weights_shape: \] \[2 1\]'):
      self.evaluate(spec.loss)

  def test_multi_dim_train_weights_wrong_outer_dim(self):
    """Logits, labels of shape [2, 2, 3], weight shape [2, 2, 2]."""
    head = head_lib._regression_head(
        weight_column='label_weights', label_dimension=3)
    logits = np.array([[[00., 01., 02.], [10., 11., 12.]],
                       [[20., 21., 22.], [30., 31., 32.]]])
    labels = np.array([[[01., 02., 03.], [12., 13., 14.]],
                       [[23., 24., 25.], [34., 35., 36.]]])
    def _no_op_train_fn(loss):
      del loss
      return control_flow_ops.no_op()

    if context.executing_eagerly():
      with self.assertRaisesRegexp(ValueError, 'weights shape'):
        head.loss(
            features={'label_weights': np.array([[[1., 1.1], [1.5, 1.6]],
                                                 [[2., 2.1], [2.5, 2.6]]])},
            mode=model_fn.ModeKeys.TRAIN,
            logits=logits,
            labels=labels,
            regularization_losses=None)
      return

    weights_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    features = {
        'label_weights': weights_placeholder,
    }
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_no_op_train_fn)
    with self.cached_session():
      _initialize_variables(self, monitored_session.Scaffold())
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[logits_shape: \]\s\[2 2 3\]\s\[weights_shape: \]\s\[2 2 2\]'):
        spec.loss.eval({
            weights_placeholder: np.array([[[1., 1.1], [1.5, 1.6]],
                                           [[2., 2.1], [2.5, 2.6]]])})


class RegressionHeadForEstimator(test.TestCase):
  """Tests for create_estimator_spec running in Graph mode only."""

  def test_train_with_optimizer(self):
    head = head_lib._regression_head()
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43.,), (44.,),), dtype=np.float64)
    expected_train_result = b'my_train_op'
    features = {'x': np.array(((42.,),), dtype=np.float32)}
    # loss = (43-45)^2 + (44-41)^2 = 4 + 9 = 13
    expected_loss = 13

    class _Optimizer(object):

      def minimize(self, loss, global_step):
        del global_step
        with ops.control_dependencies((check_ops.assert_equal(
            math_ops.to_float(expected_loss), math_ops.to_float(loss),
            name='assert_loss'),)):
          return constant_op.constant(expected_train_result)

    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        optimizer=_Optimizer())

    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAllClose(expected_loss, loss)
      self.assertEqual(expected_train_result, train_result)

  def test_train_summaries_with_head_name(self):
    head = head_lib._regression_head(name='some_regression_head')
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43.,), (44.,),), dtype=np.float64)
    features = {'x': np.array(((42.,),), dtype=np.float32)}
    # loss = (43-45)^2 + (44-41)^2 = 4 + 9 = 13
    expected_loss = 13

    def _train_op_fn(loss):
      del loss
      return control_flow_ops.no_op()

    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)

    # Assert summaries.
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      summary_str = sess.run(spec.scaffold.summary_op)
      _assert_simple_summaries(
          self,
          {
              '{}/some_regression_head'.format(metric_keys.MetricKeys.LOSS):
                  expected_loss,
              # loss_mean = loss/2 = 13/2 = 6.5
              '{}/some_regression_head'
              .format(metric_keys.MetricKeys.LOSS_MEAN):
                  6.5,
          },
          summary_str)

  def test_train_with_update_ops(self):
    head = head_lib._regression_head()

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
          logits=np.array(((45,), (41,),), dtype=np.float32),
          labels=np.array(((43.,), (44.,),), dtype=np.float64),
          train_op_fn=_train_op_fn)

      with self.cached_session() as sess:
        _initialize_variables(self, spec.scaffold)
        sess.run(spec.train_op)
        w_value, t_value = sess.run([w, t])
        self.assertEqual(2, w_value)
        self.assertEqual(expected_train_result, t_value)

  def test_weighted_multi_batch_train(self):
    """1d label, 1 example, 3 batches."""
    # numpy_input_fn is not compitable with eager.
    head = head_lib._regression_head(weight_column='label_weights')
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45.,), (41.,), (44.,)))
    input_fn = numpy_io.numpy_input_fn(
        x={
            'x': np.array(((42.,), (43.,), (44.,))),
            'label_weights': np.array(((1.,), (.1,), (1.5,))),
            # 'logits' is not a feature, but we use `numpy_input_fn` to make a
            # batched version of it, and pop it off before passing to
            # `create_estimator_spec`.
            'logits': logits,
        },
        y=np.array(((35.,), (42.,), (45.,))),
        batch_size=1,
        num_epochs=1,
        shuffle=False)
    batched_features, batched_labels = input_fn()
    batched_logits = batched_features.pop('logits')
    spec = head.create_estimator_spec(
        features=batched_features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=batched_logits,
        labels=batched_labels,
        train_op_fn=lambda loss: loss * -7.)

    # Assert spec contains expected tensors.
    self.assertEqual(dtypes.float32, spec.loss.dtype)
    self.assertIsNotNone(spec.train_op)

    with self.cached_session() as sess:
      # Finalize graph and initialize variables.
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      queue_runner_impl.start_queue_runners()

      results = tuple([
          sess.run((spec.loss, spec.train_op)) for _ in range(len(logits))
      ])

      # losses = [1*(35-45)^2, .1*(42-41)^2, 1.5*(45-44)^2] = [100, .1, 1.5]
      expected_losses = np.array((100, .1, 1.5))
      self.assertAllClose(expected_losses, [r[0] for r in results])
      self.assertAllClose(expected_losses * -7., [r[1] for r in results])

  def test_weighted_multi_batch_eval(self):
    """1d label, 1 example, 3 batches."""
    # numpy_input_fn is not compitable with eager.
    head = head_lib._regression_head(weight_column='label_weights')
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45.,), (41.,), (44.,)))
    input_fn = numpy_io.numpy_input_fn(
        x={
            'x': np.array(((42.,), (43.,), (44.,))),
            'label_weights': np.array(((1.,), (.1,), (1.5,))),
            # 'logits' is not a feature, but we use `numpy_input_fn` to make a
            # batched version of it, and pop it off before passing to
            # `create_estimator_spec`.
            'logits': logits,
        },
        y=np.array(((35.,), (42.,), (45.,))),
        batch_size=1,
        num_epochs=1,
        shuffle=False)
    batched_features, batched_labels = input_fn()
    batched_logits = batched_features.pop('logits')
    spec = head.create_estimator_spec(
        features=batched_features,
        mode=model_fn.ModeKeys.EVAL,
        logits=batched_logits,
        labels=batched_labels,
        train_op_fn=None)

    # losses = [1*(35-45)^2, .1*(42-41)^2, 1.5*(45-44)^2] = [100, .1, 1.5]
    # loss = sum(losses) = 100+.1+1.5 = 101.6
    # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.076923
    expected_metrics = {
        metric_keys.MetricKeys.LOSS_MEAN:
            39.076923,
        metric_keys.MetricKeys.PREDICTION_MEAN:
            (45 + 41 * 0.1 + 44 * 1.5) / 2.6,
        metric_keys.MetricKeys.LABEL_MEAN: (35 + 42 * 0.1 + 45 * 1.5) / 2.6,
    }

    # Assert spec contains expected tensors.
    self.assertEqual(dtypes.float32, spec.loss.dtype)
    self.assertItemsEqual(expected_metrics.keys(), spec.eval_metric_ops.keys())
    self.assertIsNone(spec.train_op)
    _assert_no_hooks(self, spec)

    with self.cached_session() as sess:
      # Finalize graph and initialize variables.
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      queue_runner_impl.start_queue_runners()

      # Run tensors for `steps` steps.
      steps = len(logits)
      results = tuple([
          sess.run((
              spec.loss,
              # The `[1]` gives us the metric update op.
              {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
          )) for _ in range(steps)
      ])

      # Assert losses and metrics.
      self.assertAllClose((100, .1, 1.5), [r[0] for r in results])
      # For metrics, check results of value ops (in `results`).
      self.assertAllClose(expected_metrics, {
          k: spec.eval_metric_ops[k][0].eval() for k in spec.eval_metric_ops
      })


if __name__ == '__main__':
  test.main()
