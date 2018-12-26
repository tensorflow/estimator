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
"""Tests for base_head.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.head import base_head

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

  class _HeadWithTPUSupport(base_head.Head):
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

  class _HeadWithOutTPUSupport(base_head.Head):
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

  class _InvalidHead(base_head.Head):
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

  @test_util.run_in_graph_and_eager_modes
  def test_head_override_tpu_estimator_spec(self):
    """Test for `_Head` that overrides _create_tpu_estimator_spec."""
    head = self._HeadWithTPUSupport()

    tpu_spec = head._create_tpu_estimator_spec(
        features=None, mode=None, logits=None)
    self.assertTrue(isinstance(tpu_spec, model_fn._TPUEstimatorSpec))
    est_spec = head.create_estimator_spec(
        features=None, mode=None, logits=None)
    self.assertTrue(isinstance(est_spec, model_fn.EstimatorSpec))

  @test_util.run_in_graph_and_eager_modes
  def test_head_override_estimator_spec(self):
    """Test for `Head` that overrides create_estimator_spec."""
    head = self._HeadWithOutTPUSupport()

    with self.assertRaisesRegexp(
        NotImplementedError,
        'TPUEstimatorSpec not available for this model head.'):
      _ = head._create_tpu_estimator_spec(
          features=None, mode=None, logits=None)
    est_spec = head.create_estimator_spec(
        features=None, mode=None, logits=None)
    self.assertTrue(isinstance(est_spec, model_fn.EstimatorSpec))

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.deprecated_graph_mode_only
  def test_tensor_shape_checking_in_graph_mode(self):
    """Test for shape checking of tensor with partially defined shape."""
    labels_placeholder = array_ops.placeholder(
        dtype=dtypes.float32, shape=(None, 1))
    logits_placeholder = array_ops.placeholder(
        dtype=dtypes.float32, shape=(None, 1))
    labels_input = np.array([[-10.], [10.]], dtype=np.float32)
    logits_input = np.array([[1.], [0.]], dtype=np.float32)

    loss = np.array([[1.], [2.]], dtype=np.float32)
    def _loss_fn(labels, logits):
      check_labels = control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(labels, labels_input)),
          data=[labels])
      check_logits = control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(logits, logits_input)),
          data=[logits])
      with ops.control_dependencies([check_labels, check_logits]):
        return constant_op.constant(loss)

    unweighted_loss = base_head.call_loss_fn(
        loss_fn=_loss_fn,
        labels=labels_placeholder,
        logits=logits_placeholder,
        features={'x': np.array(((42,),), dtype=np.int32)})
    with self.cached_session():
      self.assertAllClose(
          unweighted_loss.eval({
              labels_placeholder: labels_input,
              logits_placeholder: logits_input
          }),
          loss)

if __name__ == '__main__':
  test.main()
