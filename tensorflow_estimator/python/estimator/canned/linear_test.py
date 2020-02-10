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
"""Tests for linear.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator.canned import linear
from tensorflow_estimator.python.estimator.canned import linear_testing_utils


def _linear_regressor_fn(*args, **kwargs):
  return linear.LinearRegressorV2(*args, **kwargs)


def _linear_classifier_fn(*args, **kwargs):
  return linear.LinearClassifierV2(*args, **kwargs)


# Tests for Linear Regressor.


class LinearRegressorEvaluationV2Test(
    linear_testing_utils.BaseLinearRegressorEvaluationTest, tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorEvaluationTest.__init__(
        self, _linear_regressor_fn, fc_lib=feature_column_v2)


class LinearRegressorPredictV2Test(
    linear_testing_utils.BaseLinearRegressorPredictTest, tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorPredictTest.__init__(
        self, _linear_regressor_fn, fc_lib=feature_column_v2)


class LinearRegressorIntegrationV2Test(
    linear_testing_utils.BaseLinearRegressorIntegrationTest, tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorIntegrationTest.__init__(
        self, _linear_regressor_fn, fc_lib=feature_column_v2)


class LinearRegressorTrainingV2Test(
    linear_testing_utils.BaseLinearRegressorTrainingTest, tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorTrainingTest.__init__(
        self, _linear_regressor_fn, fc_lib=feature_column_v2)


# Tests for Linear Classifier.


class LinearClassifierTrainingV2Test(
    linear_testing_utils.BaseLinearClassifierTrainingTest, tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearClassifierTrainingTest.__init__(
        self,
        linear_classifier_fn=_linear_classifier_fn,
        fc_lib=feature_column_v2)


class LinearClassifierEvaluationV2Test(
    linear_testing_utils.BaseLinearClassifierEvaluationTest, tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearClassifierEvaluationTest.__init__(
        self,
        linear_classifier_fn=_linear_classifier_fn,
        fc_lib=feature_column_v2)


class LinearClassifierPredictV2Test(
    linear_testing_utils.BaseLinearClassifierPredictTest, tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearClassifierPredictTest.__init__(
        self,
        linear_classifier_fn=_linear_classifier_fn,
        fc_lib=feature_column_v2)


class LinearClassifierIntegrationV2Test(
    linear_testing_utils.BaseLinearClassifierIntegrationTest, tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearClassifierIntegrationTest.__init__(
        self,
        linear_classifier_fn=_linear_classifier_fn,
        fc_lib=feature_column_v2)


# Tests for Linear logit_fn.


class LinearLogitFnV2Test(linear_testing_utils.BaseLinearLogitFnTest,
                          tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearLogitFnTest.__init__(
        self, fc_lib=feature_column_v2)


# Tests for warm-starting with Linear logit_fn.


class LinearWarmStartingV2Test(linear_testing_utils.BaseLinearWarmStartingTest,
                               tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearWarmStartingTest.__init__(
        self,
        _linear_classifier_fn,
        _linear_regressor_fn,
        fc_lib=feature_column_v2)


class ComputeFractionOfZeroTest(tf.test.TestCase):

  def _assertSparsity(self, expected_sparsity, tensor):
    sparsity = linear._compute_fraction_of_zero([tensor])
    self.assertAllClose(expected_sparsity, sparsity)

  def test_small_float32(self):
    self._assertSparsity(
        0.75, ops.convert_to_tensor([0, 0, 0, 1], dtype=tf.dtypes.float32))
    self._assertSparsity(
        0.5, ops.convert_to_tensor([0, 1, 0, 1], dtype=tf.dtypes.float32))

  def test_small_int32(self):
    self._assertSparsity(
        0.75, ops.convert_to_tensor([0, 0, 0, 1], dtype=tf.dtypes.int32))

  def test_small_float64(self):
    self._assertSparsity(
        0.75, ops.convert_to_tensor([0, 0, 0, 1], dtype=tf.dtypes.float64))

  def test_small_int64(self):
    self._assertSparsity(
        0.75, ops.convert_to_tensor([0, 0, 0, 1], dtype=tf.dtypes.int64))

  def test_nested(self):
    self._assertSparsity(
        0.75, [ops.convert_to_tensor([0, 0]),
               ops.convert_to_tensor([0, 1])])

  def test_none(self):
    with self.assertRaises(ValueError):
      linear._compute_fraction_of_zero([])

  def test_empty(self):
    sparsity = linear._compute_fraction_of_zero([ops.convert_to_tensor([])])
    self.assertTrue(
        self.evaluate(tf.math.is_nan(sparsity)),
        'Expected sparsity=nan, got %s' % sparsity)

  def test_multiple_empty(self):
    sparsity = linear._compute_fraction_of_zero([
        ops.convert_to_tensor([]),
        ops.convert_to_tensor([]),
    ])
    self.assertTrue(
        self.evaluate(tf.math.is_nan(sparsity)),
        'Expected sparsity=nan, got %s' % sparsity)

  def test_some_empty(self):
    with self.test_session():
      self._assertSparsity(0.5, [
          ops.convert_to_tensor([]),
          ops.convert_to_tensor([0.]),
          ops.convert_to_tensor([1.]),
      ])

  def test_mixed_types(self):
    with self.test_session():
      self._assertSparsity(0.6, [
          ops.convert_to_tensor([0, 0, 1, 1, 1], dtype=tf.dtypes.float32),
          ops.convert_to_tensor([0, 0, 0, 0, 1], dtype=tf.dtypes.int32),
      ])

  def test_2_27_zeros__using_512_MiB_of_ram(self):
    self._assertSparsity(1., tf.zeros([int(2**27 * 1.01)],
                                      dtype=tf.dtypes.int8))

  def test_2_27_ones__using_512_MiB_of_ram(self):
    self._assertSparsity(0., tf.ones([int(2**27 * 1.01)], dtype=tf.dtypes.int8))


if __name__ == '__main__':
  tf.test.main()
