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
import six

from tensorflow.core.framework import summary_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
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


_DEFAULT_SERVING_KEY = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY


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
class MultiClassHead(test.TestCase):

  def test_n_classes_is_none(self):
    with self.assertRaisesRegexp(ValueError, 'n_classes must be > 2'):
      head_lib._multi_class_head(n_classes=None)

  def test_n_classes_is_2(self):
    with self.assertRaisesRegexp(ValueError, 'n_classes must be > 2'):
      head_lib._multi_class_head(n_classes=2)

  def test_invalid_loss_reduction(self):
    with self.assertRaisesRegexp(
        ValueError, r'Invalid loss_reduction: invalid_loss_reduction'):
      head_lib._multi_class_head(
          n_classes=3, loss_reduction='invalid_loss_reduction')
    with self.assertRaisesRegexp(ValueError, r'Invalid loss_reduction: none'):
      head_lib._multi_class_head(
          n_classes=3, loss_reduction=losses.Reduction.NONE)

  def test_loss_fn_arg_labels_missing(self):

    def _loss_fn(logits):
      del logits  # Unused

    with self.assertRaisesRegexp(
        ValueError, r'loss_fn must contain argument: labels\. '
        r'Given arguments: \(\'logits\',\)'):
      head_lib._multi_class_head(n_classes=3, loss_fn=_loss_fn)

  def test_loss_fn_arg_logits_missing(self):

    def _loss_fn(labels):
      del labels  # unused

    with self.assertRaisesRegexp(
        ValueError, r'loss_fn must contain argument: logits\. '
        r'Given arguments: \(\'labels\',\)'):
      head_lib._multi_class_head(n_classes=3, loss_fn=_loss_fn)

  def test_loss_fn_arg_features_ok(self):

    def _loss_fn(labels, logits, features):
      del labels, logits, features  # Unused

    head_lib._multi_class_head(n_classes=3, loss_fn=_loss_fn)

  def test_loss_fn_arg_invalid(self):

    def _loss_fn(labels, logits, name=None):
      del labels, logits, name  # Unused

    with self.assertRaisesRegexp(ValueError,
                                 r'loss_fn has unexpected args: \[\'name\'\]'):
      head_lib._multi_class_head(n_classes=3, loss_fn=_loss_fn)

  def test_invalid_logits_shape(self):
    n_classes = 3
    head = head_lib._multi_class_head(n_classes)
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
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits_placeholder)
    with self.cached_session():
      with self.assertRaisesRegexp(errors.OpError, 'logits shape'):
        spec.predictions[pred_key].eval({logits_placeholder: logits_2x2})

  def test_invalid_labels_shape(self):
    n_classes = 3
    head = head_lib._multi_class_head(n_classes)
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
          mode=model_fn.ModeKeys.EVAL)
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
        mode=model_fn.ModeKeys.EVAL)
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
    head = head_lib._multi_class_head(n_classes)
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
          mode=model_fn.ModeKeys.EVAL)
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
          mode=model_fn.ModeKeys.EVAL)

  def test_invalid_labels_values(self):
    n_classes = 3
    head = head_lib._multi_class_head(n_classes)
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
            mode=model_fn.ModeKeys.EVAL)

      with self.assertRaisesRegexp(ValueError, 'Labels must be >= 0'):
        training_loss = head.loss(
            logits=logits_2x3,
            labels=labels_2x1_with_negative_id,
            features=features,
            mode=model_fn.ModeKeys.EVAL)
      return

    # Dynamic shape only works in Graph mode.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.int64)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    training_loss = head.loss(
        logits=logits_placeholder,
        labels=labels_placeholder,
        features=features,
        mode=model_fn.ModeKeys.EVAL)
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
    head = head_lib._multi_class_head(n_classes)
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
          mode=model_fn.ModeKeys.EVAL)
      self.evaluate(loss)

  def test_incompatible_labels_shape(self):
    n_classes = 3
    head = head_lib._multi_class_head(n_classes)
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
            mode=model_fn.ModeKeys.EVAL)
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
          mode=model_fn.ModeKeys.EVAL)

    # Dynamic shape only works in Graph mode.
    labels_placeholder = array_ops.placeholder(dtype=dtypes.int64)
    logits_placeholder = array_ops.placeholder(dtype=dtypes.float32)
    training_loss = head.loss(
        logits=logits_placeholder,
        labels=labels_placeholder,
        features=features,
        mode=model_fn.ModeKeys.EVAL)
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
    head = head_lib._multi_class_head(n_classes)
    self.assertEqual(n_classes, head.logits_dimension)

    logits = [[1., 0., 0.], [0., 0., 1.]]
    expected_probabilities = [[0.576117, 0.2119416, 0.2119416],
                              [0.2119416, 0.2119416, 0.576117]]
    expected_class_ids = [[0], [2]]
    expected_classes = [[b'0'], [b'2']]
    expected_export_classes = [[b'0', b'1', b'2']] * 2

    keys = prediction_keys.PredictionKeys
    preds = head.predictions(
        logits, [keys.LOGITS, keys.PROBABILITIES, keys.CLASS_IDS, keys.CLASSES])
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
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits)

    self.assertItemsEqual((_DEFAULT_SERVING_KEY, 'predict', 'classification'),
                          spec.export_outputs.keys())

    # Assert predictions and export_outputs.
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      predictions = sess.run(spec.predictions)
      self.assertAllClose(logits, predictions[keys.LOGITS])
      self.assertAllClose(expected_probabilities,
                          predictions[keys.PROBABILITIES])
      self.assertAllClose(expected_class_ids, predictions[keys.CLASS_IDS])
      self.assertAllEqual(expected_classes, predictions[keys.CLASSES])

      self.assertAllClose(
          expected_probabilities,
          sess.run(spec.export_outputs[_DEFAULT_SERVING_KEY].scores))
      self.assertAllEqual(
          expected_export_classes,
          sess.run(spec.export_outputs[_DEFAULT_SERVING_KEY].classes))

  def test_predict_with_invalid_keys(self):
    n_classes = 3
    head = head_lib._multi_class_head(n_classes)
    self.assertEqual(n_classes, head.logits_dimension)

    logits = [[1., 0., 0.], [0., 0., 1.]]
    with self.assertRaisesRegexp(
        ValueError,
        r'Prediction key must be in PredictionKeys, given: some_invalid_key'):
      preds = head.predictions(logits, ['some_invalid_key'])
      self.evaluate(preds)

  def test_predict_with_vocabulary_list(self):
    n_classes = 3
    head = head_lib._multi_class_head(
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
        mode=model_fn.ModeKeys.PREDICT,
        logits=logits)

    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertAllEqual(expected_classes,
                          sess.run(spec.predictions[pred_key]))
      self.assertAllEqual(
          expected_export_classes,
          sess.run(spec.export_outputs[_DEFAULT_SERVING_KEY].classes))

  def test_weight_should_not_impact_prediction(self):
    n_classes = 3
    head = head_lib._multi_class_head(n_classes, weight_column='label_weights')
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
        features=features, mode=model_fn.ModeKeys.PREDICT, logits=logits)

    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      predictions = sess.run(spec.predictions)
      self.assertAllClose(logits, predictions[keys.LOGITS])
      self.assertAllClose(expected_probabilities,
                          predictions[keys.PROBABILITIES])

  def test_eval_create_loss(self):
    n_classes = 3
    head = head_lib._multi_class_head(n_classes)

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
        mode=model_fn.ModeKeys.EVAL)
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

    head = head_lib._multi_class_head(n_classes=3, loss_fn=_loss_fn)

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

    head = head_lib._multi_class_head(n_classes=3, loss_fn=_loss_fn)

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
          mode=model_fn.ModeKeys.EVAL)
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[loss_fn must return Tensor of shape \[D0, D1, ... DN, 1\]\. \] '
          r'\[logits_shape: \] \[2 3\] \[loss_shape: \] \[2\]'):
        self.evaluate(actual_training_loss)

  def test_eval_labels_none(self):
    """Tests that error is raised when labels is None."""
    head = head_lib._multi_class_head(n_classes=3)

    with self.assertRaisesRegexp(
        ValueError, r'You must provide a labels Tensor\. Given: None\.'):
      head.loss(
          logits=np.array((
              (10, 0, 0),
              (0, 10, 0),
          ), dtype=np.float32),
          labels=None,
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=model_fn.ModeKeys.EVAL)

  def test_eval(self):
    n_classes = 3
    head = head_lib._multi_class_head(n_classes)
    logits = np.array((
        (10, 0, 0),
        (0, 10, 0),
    ), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # loss = sum(cross_entropy(labels, logits)) / batch_size = sum(10, 0) / 2 = 5.
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
          logits, labels, features=features, mode=model_fn.ModeKeys.EVAL)
      self.assertIsNotNone(loss)
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
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
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, and metrics.
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
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
    head = head_lib._multi_class_head(n_classes, name='some_multiclass_head')
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
    head = head_lib._multi_class_head(n_classes)
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
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels,
        regularization_losses=regularization_losses)

    # Assert predictions, loss, and metrics.
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
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
    head = head_lib._multi_class_head(
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
          mode=model_fn.ModeKeys.EVAL)
      self.assertAllClose(
          expected_training_loss, training_loss, rtol=1e-2, atol=1e-2)
    else:
      training_loss = head.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=model_fn.ModeKeys.EVAL)
      with self.cached_session():
        _initialize_variables(self, monitored_session.Scaffold())
        self.assertAllClose(
            expected_training_loss, training_loss.eval(), rtol=1e-2, atol=1e-2)

  def test_eval_with_label_vocabulary(self):
    n_classes = 3
    head = head_lib._multi_class_head(
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
          mode=model_fn.ModeKeys.EVAL)
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
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)

    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
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
    head = head_lib._multi_class_head(n_classes, weight_column='label_weights')

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
          mode=model_fn.ModeKeys.EVAL)
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
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)
    # Assert spec contains expected tensors.
    self.assertIsNotNone(spec.loss)
    self.assertItemsEqual(expected_metrics.keys(), spec.eval_metric_ops.keys())
    self.assertIsNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert loss, and metrics.
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
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
    head = head_lib._multi_class_head(n_classes=3)

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
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)
      self.assertAllClose(
          expected_unreduced_loss, unreduced_loss.eval(), rtol=tol, atol=tol)
      self.assertAllClose(expected_weights, actual_weights)

  def test_train_create_loss_loss_reduction(self):
    """Tests create_loss with loss_reduction."""
    head = head_lib._multi_class_head(
        n_classes=3, loss_reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

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
    # training_loss = 1 * 10 + 1 * 0 / num_nonzero_weights
    expected_training_loss = 10. / 2.
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
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)
      self.assertAllClose(
          expected_unreduced_loss, unreduced_loss.eval(), rtol=tol, atol=tol)
      self.assertAllClose(expected_weights, actual_weights)

  def test_train_labels_none(self):
    """Tests that error is raised when labels is None."""
    head = head_lib._multi_class_head(n_classes=3)

    def _no_op_train_fn(loss):
      del loss
      return control_flow_ops.no_op()

    with self.assertRaisesRegexp(
        ValueError, r'You must provide a labels Tensor\. Given: None\.'):
      head.loss(
          logits=np.array((
              (10, 0, 0),
              (0, 10, 0),
          ), dtype=np.float32),
          labels=None,
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=model_fn.ModeKeys.TRAIN)

  def test_train(self):
    n_classes = 3
    head = head_lib._multi_class_head(n_classes)

    logits = np.array((
        (10, 0, 0),
        (0, 10, 0),
    ), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    features = {'x': np.array(((42,),), dtype=np.int32)}

    # loss = sum(cross_entropy(labels, logits)) / batch_size = sum(10, 0) / 2 = 5.
    expected_loss = 5.
    tol = 1e-2
    if context.executing_eagerly():
      loss = head.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=model_fn.ModeKeys.TRAIN)
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
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)

    self.assertIsNotNone(spec.loss)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, train_op, and summaries.
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str = sess.run((spec.loss, spec.train_op,
                                                  spec.scaffold.summary_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)),
          train_result)
      _assert_simple_summaries(
          self, {
              metric_keys.MetricKeys.LOSS: expected_loss,
          }, summary_str, tol)

  def test_train_with_regularization_losses(self):
    n_classes = 3
    head = head_lib._multi_class_head(n_classes)

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
          mode=model_fn.ModeKeys.TRAIN,
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
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn,
        regularization_losses=regularization_losses)

    # Assert predictions, loss, train_op, and summaries.
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str = sess.run((spec.loss, spec.train_op,
                                                  spec.scaffold.summary_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)),
          train_result)
      _assert_simple_summaries(
          self, {
              metric_keys.MetricKeys.LOSS:
                  expected_loss,
              metric_keys.MetricKeys.LOSS_REGULARIZATION: (
                  expected_regularization_loss),
          }, summary_str, tol)

  def test_train_one_dim_create_loss(self):
    """Tests create_loss with 1D labels and weights (shape [batch_size])."""
    head = head_lib._multi_class_head(
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
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)

  def test_train_one_dim(self):
    """Tests train with 1D labels and weights (shape [batch_size])."""
    head = head_lib._multi_class_head(
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
          mode=model_fn.ModeKeys.TRAIN)
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
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels_rank_1,
        train_op_fn=_train_op_fn)

    self.assertIsNotNone(spec.loss)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, train_op, and summaries.
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str = sess.run((spec.loss, spec.train_op,
                                                  spec.scaffold.summary_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)),
          train_result)
      _assert_simple_summaries(
          self, {
              metric_keys.MetricKeys.LOSS:
                  expected_loss,
          }, summary_str, tol)

  def test_train_with_vocabulary_create_loss(self):
    n_classes = 3
    head = head_lib._multi_class_head(
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
          mode=model_fn.ModeKeys.TRAIN)
      self.assertAllClose(
          expected_training_loss, training_loss, rtol=1e-2, atol=1e-2)
      return

    training_loss = head.loss(
        logits=logits,
        labels=labels,
        features=features,
        mode=model_fn.ModeKeys.TRAIN)
    with self.cached_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=1e-2, atol=1e-2)

  def test_train_with_vocabulary(self):
    n_classes = 3
    head = head_lib._multi_class_head(
        n_classes, label_vocabulary=['aang', 'iroh', 'zuko'])

    logits = [[10., 0, 0], [0, 10, 0]]
    labels = [[b'iroh'], [b'iroh']]
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # loss = sum(cross_entropy(labels, logits)) / batch_size = sum(10, 0) / 2 = 5.
    expected_loss = 5.
    tol = 1e-2
    if context.executing_eagerly():
      loss = head.loss(
          logits=logits,
          labels=labels,
          features=features,
          mode=model_fn.ModeKeys.TRAIN)
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      return

    def _train_op_fn(loss):
      del loss
      return control_flow_ops.no_op()

    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      loss = sess.run(spec.loss)
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)

  def test_weighted_multi_example_train(self):
    n_classes = 3
    head = head_lib._multi_class_head(n_classes, weight_column='label_weights')

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
          mode=model_fn.ModeKeys.TRAIN)
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
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)

    self.assertIsNotNone(spec.loss)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    _assert_no_hooks(self, spec)

    # Assert predictions, loss, train_op, and summaries.
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str = sess.run((spec.loss, spec.train_op,
                                                  spec.scaffold.summary_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)),
          train_result)
      _assert_simple_summaries(
          self,
          {
              metric_keys.MetricKeys.LOSS:
                  expected_loss,
          },
          summary_str,
          tol)

  def test_multi_dim_weighted_train_create_loss(self):
    """Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 2]."""
    head = head_lib._multi_class_head(
        n_classes=3,
        weight_column='weights')

    logits = np.array([[[10, 0, 0], [12, 0, 0]], [[0, 10, 0], [0, 15, 0]]],
                      dtype=np.float32)
    labels = np.array([[[0], [1]], [[1], [2]]], dtype=np.int64)
    weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)

    # unreduced_loss = cross_entropy(labels, logits) = [[0, 12], [0, 15]].
    # weights are reshaped to [2, 2, 1] to match logits.
    # training_loss = sum(1*0 + 1.5*12 + 2*0 + 2.5*15) / batch_size = 55.5 / 2x2 = 13.875
    expected_training_loss = 13.875
    tol = 1e-2
    if context.executing_eagerly():
      training_loss = head.loss(logits, labels, features={'weights': weights})
      self.assertAllClose(
          expected_training_loss, training_loss, rtol=tol, atol=tol)
      return

    training_loss = head.loss(logits, labels, features={'weights': weights})
    with self.cached_session():
      _initialize_variables(self, monitored_session.Scaffold())
      self.assertAllClose(
          expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)

  def test_multi_dim_weighted_train(self):
    """Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 2]."""
    head = head_lib._multi_class_head(
        n_classes=3,
        weight_column='weights')

    logits = np.array([[[10, 0, 0], [12, 0, 0]], [[0, 10, 0], [0, 15, 0]]],
                      dtype=np.float32)
    labels = np.array([[[0], [1]], [[1], [2]]], dtype=np.int64)
    weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)
    tol = 1e-2
    # loss = cross_entropy(labels, logits) = [[0, 12], [0, 15]].
    # weighted_sum_loss = (1*0 + 1.5*12 + 2*0 + 2.5*15) = 55.5
    # training_loss = weighted_sum_loss / batch_size  = 55.5 / 2x2 = 13.875
    expected_loss = 13.875
    if context.executing_eagerly():
      loss = head.loss(
          logits=logits,
          labels=labels,
          features={'weights': weights},
          mode=model_fn.ModeKeys.TRAIN)
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
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)

    # Assert predictions, loss, train_op, and summaries.
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)),
          train_result)

  def test_multi_dim_train_weights_wrong_inner_dim(self):
    """Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 1]."""
    head = head_lib._multi_class_head(n_classes=3, weight_column='weights')
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
      _initialize_variables(self, monitored_session.Scaffold())
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[logits_shape: \] \[2 2 3\] \[weights_shape: \] \[2 1\]'):
        spec.loss.eval()

  def test_multi_dim_train_weights_wrong_outer_dim(self):
    """Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 2, 3]."""
    head = head_lib._multi_class_head(n_classes=3, weight_column='weights')
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
      _initialize_variables(self, monitored_session.Scaffold())
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r'\[logits_shape: \]\s\[2 2 3\]\s\[weights_shape: \]\s\[2 2 3\]'):
        spec.loss.eval({weights_placeholder: weights})

  def test_multi_dim_weighted_eval(self):
    """Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 2]."""
    head = head_lib._multi_class_head(
        n_classes=3,
        weight_column='weights')
    logits = np.array([[[10, 0, 0], [12, 0, 0]], [[0, 10, 0], [0, 15, 0]]],
                      dtype=np.float32)
    labels = np.array([[[0], [1]], [[1], [2]]], dtype=np.int64)
    weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)
    # loss = cross_entropy(labels, logits) = [[0, 12], [0, 15]].
    # weighted_sum_loss = 1*0 + 1.5*12 + 2*0 + 2.5*15 = 55.5
    # training_loss = weighted_sum_loss / batch_size = 55.5 / 2x2 = 13.875
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
          rtol=tol,
          atol=tol)
      return

    # Create estimator spec.
    spec = head.create_estimator_spec(
        features={'weights': weights},
        mode=model_fn.ModeKeys.EVAL,
        logits=logits,
        labels=labels)
    # Assert predictions, loss, and metrics.
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, _ = sess.run((spec.loss, update_ops))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      # Check results of value ops (in `metrics`).
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval() for k in value_ops},
          rtol=tol,
          atol=tol)


class MultiClassHeadForEstimator(test.TestCase):
  """Tests for create_estimator_spec running in Graph mode only."""

  def test_train_with_optimizer(self):
    n_classes = 3
    head = head_lib._multi_class_head(n_classes)

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

    # loss = sum(cross_entropy(labels, logits)) / batch_size = sum(10, 0) / 2 = 5.
    expected_loss = 5.
    spec = head.create_estimator_spec(
        features=features,
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        optimizer=_Optimizer())

    tol = 1e-2
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)),
          train_result)

  def test_train_with_update_ops(self):
    n_classes = 3
    head = head_lib._multi_class_head(n_classes)

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
          logits=np.array((
              (10, 0, 0),
              (0, 10, 0),
          ), dtype=np.float32),
          labels=np.array(((1,), (1,)), dtype=np.int64),
          train_op_fn=_train_op_fn)

      with self.cached_session() as sess:
        _initialize_variables(self, spec.scaffold)
        sess.run(spec.train_op)
        w_value, t_value = sess.run([w, t])
        self.assertEqual(2, w_value)
        self.assertEqual(expected_train_result, t_value)

  def test_train_summaries_with_head_name(self):
    n_classes = 3
    head = head_lib._multi_class_head(n_classes, name='some_multiclass_head')

    logits = np.array((
        (10, 0, 0),
        (0, 10, 0),
    ), dtype=np.float32)
    labels = np.array(((1,), (1,)), dtype=np.int64)
    # loss = sum(cross_entropy(labels, logits)) / batch_size= sum(10, 0) / 2 = 5.
    expected_loss = 5.
    features = {'x': np.array(((42,),), dtype=np.int32)}

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
    tol = 1e-2
    with self.cached_session() as sess:
      _initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      summary_str = sess.run(spec.scaffold.summary_op)
      _assert_simple_summaries(
          self, {
              '{}/some_multiclass_head'.format(metric_keys.MetricKeys.LOSS):
                  expected_loss,
          }, summary_str, tol)


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
      with self.assertRaisesRegexp(ValueError, 'Mismatched label shape'):
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
      with self.assertRaisesRegexp(ValueError, 'Mismatched label shape'):
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
    self.assertAllClose(6.5, self.evaluate(regularized_training_loss))

  def test_eval_create_loss_loss_fn(self):
    """Tests head.loss for eval mode and custom loss_fn."""
    loss = np.array([[0., 1.], [2., 3.]], dtype=np.float32)
    batch_size = 4.
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
    self.assertAllClose(
        np.sum(loss) / batch_size, self.evaluate(regularized_training_loss))

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

    # weighted_loss = (43-45)^2 + (44-41)^2 = 13.
    # loss = weighted_loss / batch_size = (4+9) / 2 = 6.5
    expected_loss = 6.5
    # loss_mean = loss/sum(weights) = 13/2 = 6.5
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
      loss = head.loss(
          logits, labels, features=features, mode=model_fn.ModeKeys.EVAL)
      self.assertIsNotNone(loss)
      self.assertAllClose(expected_loss, loss)
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
        self.assertAllClose(6.5, loss)
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
    head = head_lib._regression_head()
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
    # training_loss = (1 * 4 + 1 * 9) / 2 = 6.5
    expected_training_loss = 6.5
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
    # loss = ((43-45)^2 + (44-41)^2) / 2 = (4 + 9) / 2 = 6.5
    expected_loss = 6.5
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
      }, summary_str)

  def test_train_with_regularization_losses(self):
    head = head_lib._regression_head()
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

    # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
    # expected_loss = loss / batch_size = 33.8666667
    expected_loss = 33.8666667
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
      loss = head.loss(
          logits, labels, features=features, mode=model_fn.ModeKeys.EVAL)
      self.assertIsNotNone(loss)
      self.assertAllClose(expected_loss, loss)
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
        self.assertAllClose(expected_loss, loss)
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
    # weighted_loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
    # loss = weighted_loss / batch_size = 101.6 / 3 = 33.8666667
    self.assertAllClose(33.8666667, self.evaluate(loss))

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
    # weighted_loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
    # expected_loss = weighted_loss / batch_size = 101.6 / 3 = 33.8666667
    expected_loss = 33.8666667
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
      }, summary_str)

  def test_train_one_dim_create_loss(self):
    """Tests create_loss with 1D labels and weights (shape [batch_size])."""
    head = head_lib._regression_head(weight_column='label_weights')
    logits = np.array(((45,), (41,), (44,)), dtype=np.float32)
    x_feature_rank_1 = np.array((42., 43., 44.,), dtype=np.float32)
    weight_rank_1 = np.array((1., .1, 1.5,), dtype=np.float64)
    labels_rank_1 = np.array((35., 42., 45.,))
    # training_loss = (100 * 1 + 1 * .1 + 1.5 * 1) / batch_size = 101.6 / 3 = 33.8666667
    expected_training_loss = 33.8666667
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
    # loss = (1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2) / batch_size
    #      = (100+.1+1.5) / 3 = 101.6 / 3 = 33.8666667
    expected_loss = 33.8666667
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
    # weighted_sum_loss = 1 * 100 + .1 * 1 + 1.5 * 1 = 101.6
    # expected_training_loss = weighted_sum_loss / batch_size = 101.6 / 3 = 33.8666667
    self.assertAllClose(33.8666667, self.evaluate(regularized_training_loss))

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

    # weighted_loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
    # expected_loss = weighted_loss / batch_size = 101.6 / 3 = 33.8666667
    expected_loss = 33.8666667
    # loss_mean = weighted_loss/(1+.1+1.5) = 101.6/2.6 = 39.0769231
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
      loss = head.loss(
          logits, labels, features=features, mode=model_fn.ModeKeys.EVAL)
      self.assertIsNotNone(loss)
      self.assertAllClose(expected_loss, loss)
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
        self.assertAllClose(expected_loss, loss)
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
    self.assertAllClose(33.8666667, self.evaluate(regularized_training_loss))

  def test_weighted_multi_value_train(self):
    """3d label, 1 example, 1 batch."""
    head = head_lib._regression_head(
        weight_column='label_weights', label_dimension=3)
    self.assertEqual(3, head.logits_dimension)

    logits = np.array(((45., 41., 44.),))
    labels = np.array(((35., 42., 45.),))
    expected_train_result = b'my_train_op'
    # loss = (1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2) / batch_size
    #      = (100+.1+1.5) / 3 = 101.6 / 3 = 33.8666667
    expected_loss = 33.8666667
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
    training_loss_weighted_sum = np.sum(
        np.array([[[1. * x for x in [1., 1., 1.]],
                   [1.5 * x for x in [4., 4., 4.]]],
                  [[2. * x for x in [9., 9., 9.]],
                   [2.5 * x for x in [16., 16., 16.]]]]))
    # expected_training_loss = training_loss_weighted_sum / batch_size (2 x 2 x 3)
    # Create loss.
    training_loss = head.loss(
        features={'label_weights': weights},
        mode=model_fn.ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        regularization_losses=None)
    self.assertAllClose(
        training_loss_weighted_sum / 12., self.evaluate(training_loss))

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
    # weighted_loss_sum = (1*3*1^2 + 1.5*3*2^2 + 2*3*3^2 +2.5*3*4^2) = 195
    # loss = weighted_loss_sum / batch_size = 195 / (2 * 2 * 3) = 16.25
    expected_loss = 16.25
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
    # loss = ((43-45)^2 + (44-41)^2) / 2 = (4 + 9) / 2 = 13 / 2 = 6.5
    expected_loss = 6.5

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
          logits=np.array((
              (45,),
              (41,),
          ), dtype=np.float32),
          labels=np.array((
              (43.,),
              (44.,),
          ), dtype=np.float64),
          train_op_fn=_train_op_fn)

      with self.cached_session() as sess:
        _initialize_variables(self, spec.scaffold)
        sess.run(spec.train_op)
        w_value, t_value = sess.run([w, t])
        self.assertEqual(2, w_value)
        self.assertEqual(expected_train_result, t_value)

  def test_train_summaries_with_head_name(self):
    head = head_lib._regression_head(name='some_regression_head')
    self.assertEqual(1, head.logits_dimension)

    # Create estimator spec.
    logits = np.array(((45,), (41,),), dtype=np.float32)
    labels = np.array(((43.,), (44.,),), dtype=np.float64)
    features = {'x': np.array(((42.,),), dtype=np.float32)}
    # loss = ((43-45)^2 + (44-41)^2) / 2 = (4 + 9) / 2 = 6.5
    expected_loss = 6.5

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
          },
          summary_str)

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
