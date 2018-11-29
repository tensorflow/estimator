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

import numpy as np

from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow_estimator.python.estimator.canned import linear
from tensorflow_estimator.python.estimator.canned.v1 import linear_testing_utils_v1


def _linear_regressor_fn(*args, **kwargs):
  return linear.LinearRegressor(*args, **kwargs)


def _linear_classifier_fn(*args, **kwargs):
  return linear.LinearClassifier(*args, **kwargs)


# Tests for Linear Regressor.


class LinearRegressorPartitionerTest(
    linear_testing_utils_v1.BaseLinearRegressorPartitionerTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearRegressorPartitionerTest.__init__(
        self, _linear_regressor_fn, fc_lib=feature_column)


class LinearRegressorPartitionerV2Test(
    linear_testing_utils_v1.BaseLinearRegressorPartitionerTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearRegressorPartitionerTest.__init__(
        self, _linear_regressor_fn, fc_lib=feature_column_v2)


class LinearRegressorEvaluationTest(
    linear_testing_utils_v1.BaseLinearRegressorEvaluationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearRegressorEvaluationTest.__init__(
        self, _linear_regressor_fn, fc_lib=feature_column)


class LinearRegressorEvaluationV2Test(
    linear_testing_utils_v1.BaseLinearRegressorEvaluationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearRegressorEvaluationTest.__init__(
        self, _linear_regressor_fn, fc_lib=feature_column_v2)


class LinearRegressorPredictTest(
    linear_testing_utils_v1.BaseLinearRegressorPredictTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearRegressorPredictTest.__init__(
        self, _linear_regressor_fn, fc_lib=feature_column)


class LinearRegressorPredictV2Test(
    linear_testing_utils_v1.BaseLinearRegressorPredictTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearRegressorPredictTest.__init__(
        self, _linear_regressor_fn, fc_lib=feature_column_v2)


class LinearRegressorIntegrationTest(
    linear_testing_utils_v1.BaseLinearRegressorIntegrationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearRegressorIntegrationTest.__init__(
        self, _linear_regressor_fn, fc_lib=feature_column)


class LinearRegressorIntegrationV2Test(
    linear_testing_utils_v1.BaseLinearRegressorIntegrationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearRegressorIntegrationTest.__init__(
        self, _linear_regressor_fn, fc_lib=feature_column_v2)


class LinearRegressorTrainingTest(
    linear_testing_utils_v1.BaseLinearRegressorTrainingTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearRegressorTrainingTest.__init__(
        self, _linear_regressor_fn, fc_lib=feature_column)


class LinearRegressorTrainingV2Test(
    linear_testing_utils_v1.BaseLinearRegressorTrainingTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearRegressorTrainingTest.__init__(
        self, _linear_regressor_fn, fc_lib=feature_column_v2)


# Tests for Linear Classifier.
class LinearClassifierTrainingTest(
    linear_testing_utils_v1.BaseLinearClassifierTrainingTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearClassifierTrainingTest.__init__(
        self, linear_classifier_fn=_linear_classifier_fn, fc_lib=feature_column)


class LinearClassifierTrainingV2Test(
    linear_testing_utils_v1.BaseLinearClassifierTrainingTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearClassifierTrainingTest.__init__(
        self,
        linear_classifier_fn=_linear_classifier_fn,
        fc_lib=feature_column_v2)


class LinearClassifierEvaluationTest(
    linear_testing_utils_v1.BaseLinearClassifierEvaluationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearClassifierEvaluationTest.__init__(
        self, linear_classifier_fn=_linear_classifier_fn, fc_lib=feature_column)


class LinearClassifierEvaluationV2Test(
    linear_testing_utils_v1.BaseLinearClassifierEvaluationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearClassifierEvaluationTest.__init__(
        self,
        linear_classifier_fn=_linear_classifier_fn,
        fc_lib=feature_column_v2)


class LinearClassifierPredictTest(
    linear_testing_utils_v1.BaseLinearClassifierPredictTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearClassifierPredictTest.__init__(
        self, linear_classifier_fn=_linear_classifier_fn, fc_lib=feature_column)


class LinearClassifierPredictV2Test(
    linear_testing_utils_v1.BaseLinearClassifierPredictTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearClassifierPredictTest.__init__(
        self,
        linear_classifier_fn=_linear_classifier_fn,
        fc_lib=feature_column_v2)


class LinearClassifierIntegrationTest(
    linear_testing_utils_v1.BaseLinearClassifierIntegrationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearClassifierIntegrationTest.__init__(
        self, linear_classifier_fn=_linear_classifier_fn, fc_lib=feature_column)


class LinearClassifierIntegrationV2Test(
    linear_testing_utils_v1.BaseLinearClassifierIntegrationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearClassifierIntegrationTest.__init__(
        self,
        linear_classifier_fn=_linear_classifier_fn,
        fc_lib=feature_column_v2)


# Tests for Linear logit_fn.
class LinearLogitFnTest(linear_testing_utils_v1.BaseLinearLogitFnTest,
                        test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearLogitFnTest.__init__(
        self, fc_lib=feature_column)


class LinearLogitFnV2Test(linear_testing_utils_v1.BaseLinearLogitFnTest,
                          test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearLogitFnTest.__init__(
        self, fc_lib=feature_column_v2)


# Tests for warm-starting with Linear logit_fn.
class LinearWarmStartingTest(linear_testing_utils_v1.BaseLinearWarmStartingTest,
                             test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearWarmStartingTest.__init__(
        self,
        _linear_classifier_fn,
        _linear_regressor_fn,
        fc_lib=feature_column)


class LinearWarmStartingV2Test(
    linear_testing_utils_v1.BaseLinearWarmStartingTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearWarmStartingTest.__init__(
        self,
        _linear_classifier_fn,
        _linear_regressor_fn,
        fc_lib=feature_column_v2)


class ComputeFractionOfZeroTest(test.TestCase):

  def _assertSparsity(self, expected_sparsity, tensor):
    sparsity = linear._compute_fraction_of_zero([tensor])
    with self.test_session() as sess:
      self.assertAllClose(expected_sparsity, sess.run(sparsity))

  def test_small_float32(self):
    self._assertSparsity(
        0.75,
        ops.convert_to_tensor([0, 0, 0, 1], dtype=dtypes.float32))
    self._assertSparsity(
        0.5,
        ops.convert_to_tensor([0, 1, 0, 1], dtype=dtypes.float32))

  def test_small_int32(self):
    self._assertSparsity(
        0.75,
        ops.convert_to_tensor([0, 0, 0, 1], dtype=dtypes.int32))

  def test_small_float64(self):
    self._assertSparsity(
        0.75,
        ops.convert_to_tensor([0, 0, 0, 1], dtype=dtypes.float64))

  def test_small_int64(self):
    self._assertSparsity(
        0.75,
        ops.convert_to_tensor([0, 0, 0, 1], dtype=dtypes.int64))

  def test_nested(self):
    self._assertSparsity(
        0.75,
        [ops.convert_to_tensor([0, 0]),
         ops.convert_to_tensor([0, 1])])

  def test_none(self):
    with self.assertRaises(ValueError):
      linear._compute_fraction_of_zero([])

  def test_empty(self):
    sparsity = linear._compute_fraction_of_zero([ops.convert_to_tensor([])])
    with self.test_session() as sess:
      sparsity_np = sess.run(sparsity)
      self.assertTrue(
          np.isnan(sparsity_np),
          'Expected sparsity=nan, got %s' % sparsity_np)

  def test_multiple_empty(self):
    sparsity = linear._compute_fraction_of_zero([
        ops.convert_to_tensor([]),
        ops.convert_to_tensor([]),
    ])
    with self.test_session() as sess:
      sparsity_np = sess.run(sparsity)
      self.assertTrue(
          np.isnan(sparsity_np),
          'Expected sparsity=nan, got %s' % sparsity_np)

  def test_some_empty(self):
    with self.test_session():
      self._assertSparsity(
          0.5,
          [
              ops.convert_to_tensor([]),
              ops.convert_to_tensor([0.]),
              ops.convert_to_tensor([1.]),
          ])

  def test_mixed_types(self):
    with self.test_session():
      self._assertSparsity(
          0.6,
          [
              ops.convert_to_tensor([0, 0, 1, 1, 1], dtype=dtypes.float32),
              ops.convert_to_tensor([0, 0, 0, 0, 1], dtype=dtypes.int32),
          ])

  def test_2_27_zeros__using_512_MiB_of_ram(self):
    self._assertSparsity(
        1.,
        array_ops.zeros([int(2**27 * 1.01)], dtype=dtypes.int8))

  def test_2_27_ones__using_512_MiB_of_ram(self):
    self._assertSparsity(
        0.,
        array_ops.ones([int(2**27 * 1.01)], dtype=dtypes.int8))


if __name__ == '__main__':
  test.main()
