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
"""Tests for DNNLinearCombinedEstimatorV2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile

import numpy as np
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator.util import tf_keras
from tensorflow_estimator.python.estimator.canned import dnn_linear_combined
from tensorflow_estimator.python.estimator.canned import dnn_testing_utils
from tensorflow_estimator.python.estimator.canned import linear_testing_utils
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.export import export
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.inputs import numpy_io


def _dnn_only_estimator_fn(hidden_units,
                           feature_columns,
                           model_dir=None,
                           label_dimension=1,
                           weight_column=None,
                           optimizer='Adagrad',
                           activation_fn=tf.nn.relu,
                           dropout=None,
                           config=None):
  return dnn_linear_combined.DNNLinearCombinedEstimatorV2(
      head=regression_head.RegressionHead(
          weight_column=weight_column,
          label_dimension=label_dimension,
          # Tests in core (from which this test inherits) test the sum loss.
          loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE),
      model_dir=model_dir,
      dnn_feature_columns=feature_columns,
      dnn_optimizer=optimizer,
      dnn_hidden_units=hidden_units,
      dnn_activation_fn=activation_fn,
      dnn_dropout=dropout,
      config=config)


class DNNOnlyEstimatorEvaluateTest(
    dnn_testing_utils.BaseDNNRegressorEvaluateTest, tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNRegressorEvaluateTest.__init__(
        self, _dnn_only_estimator_fn)


class DNNOnlyEstimatorPredictTest(dnn_testing_utils.BaseDNNRegressorPredictTest,
                                  tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNRegressorPredictTest.__init__(
        self, _dnn_only_estimator_fn)


class DNNOnlyEstimatorTrainTest(dnn_testing_utils.BaseDNNRegressorTrainTest,
                                tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNRegressorTrainTest.__init__(
        self, _dnn_only_estimator_fn)


def _linear_only_estimator_fn(feature_columns,
                              model_dir=None,
                              label_dimension=1,
                              weight_column=None,
                              optimizer='Ftrl',
                              config=None,
                              sparse_combiner='sum'):
  return dnn_linear_combined.DNNLinearCombinedEstimatorV2(
      head=regression_head.RegressionHead(
          weight_column=weight_column,
          label_dimension=label_dimension,
          # Tests in core (from which this test inherits) test the sum loss.
          loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE),
      model_dir=model_dir,
      linear_feature_columns=feature_columns,
      linear_optimizer=optimizer,
      config=config,
      linear_sparse_combiner=sparse_combiner)


class LinearOnlyEstimatorEvaluateTest(
    linear_testing_utils.BaseLinearRegressorEvaluationTest, tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorEvaluationTest.__init__(
        self, _linear_only_estimator_fn)


class LinearOnlyEstimatorPredictTest(
    linear_testing_utils.BaseLinearRegressorPredictTest, tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorPredictTest.__init__(
        self, _linear_only_estimator_fn)


class LinearOnlyEstimatorTrainTest(
    linear_testing_utils.BaseLinearRegressorTrainingTest, tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorTrainingTest.__init__(
        self, _linear_only_estimator_fn)


class DNNLinearCombinedEstimatorIntegrationTest(tf.test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      tf.compat.v1.summary.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _test_complete_flow(self,
                          train_input_fn,
                          eval_input_fn,
                          predict_input_fn,
                          input_dimension,
                          label_dimension,
                          batch_size,
                          dnn_optimizer='Adagrad',
                          linear_optimizer='Ftrl'):
    linear_feature_columns = [
        tf.feature_column.numeric_column('x', shape=(input_dimension,))
    ]
    dnn_feature_columns = [
        tf.feature_column.numeric_column('x', shape=(input_dimension,))
    ]
    feature_columns = linear_feature_columns + dnn_feature_columns
    est = dnn_linear_combined.DNNLinearCombinedEstimatorV2(
        head=regression_head.RegressionHead(label_dimension=label_dimension),
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        dnn_hidden_units=(2, 2),
        model_dir=self._model_dir,
        dnn_optimizer=dnn_optimizer,
        linear_optimizer=linear_optimizer)

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
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    export_dir = est.export_saved_model(tempfile.mkdtemp(),
                                        serving_input_receiver_fn)
    self.assertTrue(tf.compat.v1.gfile.Exists(export_dir))

  def _create_input_fn(self, label_dimension, batch_size):
    """Creates input_fn for integration test."""
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
        x={'x': data}, y=data, batch_size=batch_size, shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, batch_size=batch_size, shuffle=False)

    return train_input_fn, eval_input_fn, predict_input_fn

  def test_numpy_input_fn(self):
    """Tests complete flow with numpy_input_fn."""
    label_dimension = 2
    batch_size = 10
    train_input_fn, eval_input_fn, predict_input_fn = self._create_input_fn(
        label_dimension, batch_size)

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        input_dimension=label_dimension,
        label_dimension=label_dimension,
        batch_size=batch_size)

  def test_numpy_input_fn_with_optimizer_instance(self):
    """Tests complete flow with optimizer_v2 instance."""
    label_dimension = 2
    batch_size = 10
    train_input_fn, eval_input_fn, predict_input_fn = self._create_input_fn(
        label_dimension, batch_size)

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        input_dimension=label_dimension,
        label_dimension=label_dimension,
        batch_size=batch_size,
        dnn_optimizer=tf_keras.optimizers.legacy.Adagrad(0.01),
        linear_optimizer=tf_keras.optimizers.legacy.Ftrl(0.01))


if __name__ == '__main__':
  tf.test.main()
