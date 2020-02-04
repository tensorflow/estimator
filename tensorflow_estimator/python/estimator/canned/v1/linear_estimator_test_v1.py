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
"""Tests for LinearEstimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile

import tensorflow as tf
import numpy as np
import six

from tensorflow.python.feature_column import feature_column_lib as feature_column
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer_cache
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import linear
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.canned.v1 import linear_testing_utils_v1
from tensorflow_estimator.python.estimator.export import export
from tensorflow_estimator.python.estimator.inputs import numpy_io


def _linear_estimator_fn(
    weight_column=None, label_dimension=1, **kwargs):
  """Returns a LinearEstimator that uses regression_head."""
  return linear.LinearEstimator(
      head=head_lib._regression_head(
          weight_column=weight_column,
          label_dimension=label_dimension,
          # Tests in core (from which this test inherits) test the sum loss.
          loss_reduction=tf.compat.v1.losses.Reduction.SUM),
      **kwargs)


@test_util.run_v1_only("Tests v1 only symbols")
class LinearEstimatorEvaluateTest(
    linear_testing_utils_v1.BaseLinearRegressorEvaluationTest, tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearRegressorEvaluationTest.__init__(
        self, _linear_estimator_fn)


@test_util.run_v1_only("Tests v1 only symbols")
class LinearEstimatorPredictTest(
    linear_testing_utils_v1.BaseLinearRegressorPredictTest, tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearRegressorPredictTest.__init__(
        self, _linear_estimator_fn)


@test_util.run_v1_only("Tests v1 only symbols")
class LinearEstimatorTrainTest(
    linear_testing_utils_v1.BaseLinearRegressorTrainingTest, tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    linear_testing_utils_v1.BaseLinearRegressorTrainingTest.__init__(
        self, _linear_estimator_fn)


@test_util.run_v1_only("Tests v1 only symbols")
class LinearEstimatorIntegrationTest(tf.test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      tf.compat.v1.summary.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _test_complete_flow(
      self, train_input_fn, eval_input_fn, predict_input_fn, input_dimension,
      label_dimension, batch_size):
    feature_columns = [
        tf.feature_column.numeric_column('x', shape=(input_dimension,))]
    est = linear.LinearEstimator(
        head=head_lib._regression_head(label_dimension=label_dimension),
        feature_columns=feature_columns,
        model_dir=self._model_dir)

    # Train
    num_steps = 10
    est.train(train_input_fn, steps=num_steps)

    # Evaluate
    scores = est.evaluate(eval_input_fn)
    self.assertEqual(num_steps, scores[tf.compat.v1.GraphKeys.GLOBAL_STEP])
    self.assertIn('loss', six.iterkeys(scores))

    # Predict
    predictions = np.array([
        x[prediction_keys.PredictionKeys.PREDICTIONS]
        for x in est.predict(predict_input_fn)
    ])
    self.assertAllEqual((batch_size, label_dimension), predictions.shape)

    # Export
    feature_spec = tf.compat.v1.feature_column.make_parse_example_spec(feature_columns)
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    export_dir = est.export_saved_model(tempfile.mkdtemp(),
                                        serving_input_receiver_fn)
    self.assertTrue(tf.compat.v1.gfile.Exists(export_dir))

  def test_numpy_input_fn(self):
    """Tests complete flow with numpy_input_fn."""
    label_dimension = 2
    batch_size = 10
    data = np.linspace(0., 2., batch_size * label_dimension, dtype=np.float32)
    data = data.reshape(batch_size, label_dimension)
    # learn y = x
    train_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size,
        shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        batch_size=batch_size,
        shuffle=False)

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        input_dimension=label_dimension,
        label_dimension=label_dimension,
        batch_size=batch_size)


if __name__ == '__main__':
  tf.test.main()
