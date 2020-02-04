# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for training ar_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import math

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned.timeseries import ar_model
from tensorflow_estimator.python.estimator.canned.timeseries.estimators import LSTMAutoRegressor
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import PredictionFeatures
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures


class InputFnBuilder(object):

  def __init__(self,
               noise_stddev,
               periods,
               window_size,
               batch_size,
               num_samples=200):
    self.window_size = window_size
    self.batch_size = batch_size

    split = int(num_samples * 0.8)
    self.initialize_data = lambda: self.initialize_data_with_properties(
        noise_stddev, periods, num_samples, split)

  def initialize_data_with_properties(self, noise_stddev, periods, num_samples,
                                      split):
    time = 1 + 3 * tf.range(num_samples, dtype=tf.dtypes.int64)
    time_offset = 2 * math.pi * tf.cast(time % periods[0],
                                              tf.dtypes.float32) / periods[0]
    time_offset = time_offset[:, None]
    if len(periods) > 1:
      time_offset2 = tf.cast(time % periods[1],
                                   tf.dtypes.float32) / periods[1]
      time_offset2 = time_offset2[:, None]
      data1 = tf.math.sin(time_offset / 2.0)**2 * (1 + time_offset2)
    else:
      data1 = tf.math.sin(2 * time_offset) + tf.math.cos(3 * time_offset)
    data1_noise = noise_stddev / 4. * tf.random.normal([num_samples],
                                                               1)[:, None]
    data1 = tf.math.add(data1, data1_noise)

    data2 = tf.math.sin(3 * time_offset) + tf.math.cos(5 * time_offset)
    data2_noise = noise_stddev / 3. * tf.random.normal([num_samples],
                                                               1)[:, None]
    data2 = tf.math.add(data2, data2_noise)
    data = tf.concat((4 * data1, 3 * data2), 1)
    self.train_data, self.test_data = data[0:split], data[split:]
    self.train_time, self.test_time = time[0:split], time[split:]

  def train_or_test_input_fn(self, time, data):

    def map_to_dict(time, data):
      return {TrainEvalFeatures.TIMES: time, TrainEvalFeatures.VALUES: data}

    def batch_windows(time, data):
      return tf.compat.v1.data.Dataset.zip((time, data)).batch(
          self.window_size, drop_remainder=True)

    dataset = tf.compat.v1.data.Dataset.from_tensor_slices((time, data))
    dataset = dataset.window(self.window_size, shift=1, drop_remainder=True)
    dataset = dataset.shuffle(1000, seed=2).repeat()
    dataset = dataset.flat_map(batch_windows).batch(
        self.batch_size).map(map_to_dict)
    return dataset

  def train_input_fn(self):
    self.initialize_data()
    return self.train_or_test_input_fn(self.train_time, self.train_data)

  def test_input_fn(self):
    self.initialize_data()
    return self.train_or_test_input_fn(self.test_time, self.test_data)

  def prediction_input_fn(self):

    def map_to_dict(predict_times, predict_true_values, state_times,
                    state_values, state_exogenous):
      return ({
          PredictionFeatures.TIMES:
              predict_times[None, :],
          TrainEvalFeatures.VALUES:
              predict_true_values[None, :],
          PredictionFeatures.STATE_TUPLE: (state_times[None, :],
                                           state_values[None, :],
                                           state_exogenous[None, :])
      }, {})

    self.initialize_data()
    predict_times = tf.concat(
        [self.train_time[self.window_size:], self.test_time], 0)[None, :]
    predict_true_values = tf.concat(
        [self.train_data[self.window_size:], self.test_data], 0)[None, :]
    state_times = tf.cast(self.train_time[:self.window_size][None, :],
                                tf.dtypes.float32)
    state_values = tf.cast(self.train_data[:self.window_size, :][None, :],
                                 tf.dtypes.float32)
    state_exogenous = state_times[:, :, None][:, :, :0]

    dataset = tf.compat.v1.data.Dataset.from_tensor_slices(
        (predict_times, predict_true_values, state_times, state_values,
         state_exogenous))
    dataset = dataset.map(map_to_dict)
    return dataset

  def true_values(self):
    self.initialize_data()
    predict_true_values = tf.concat(
        [self.train_data[self.window_size:], self.test_data], 0)[None, :]
    true_values = predict_true_values[0, :, 0]
    return true_values


@test_util.run_v1_only("Currently incompatible with ResourceVariable")
class ARModelTrainingTest(tf.test.TestCase):

  def train_helper(self, input_window_size, loss, max_loss=None, periods=(25,)):
    data_noise_stddev = 0.2
    if max_loss is None:
      if loss == ar_model.ARModel.NORMAL_LIKELIHOOD_LOSS:
        max_loss = 1.0
      else:
        max_loss = 0.05 / (data_noise_stddev**2)
    output_window_size = 10
    window_size = input_window_size + output_window_size
    input_fn_builder = InputFnBuilder(
        noise_stddev=data_noise_stddev,
        periods=periods,
        window_size=window_size,
        batch_size=64)

    class _RunConfig(estimator_lib.RunConfig):

      @property
      def tf_random_seed(self):
        return 3

    estimator = LSTMAutoRegressor(
        periodicities=periods,
        input_window_size=input_window_size,
        output_window_size=output_window_size,
        num_features=2,
        num_timesteps=20,
        num_units=16,
        loss=loss,
        config=_RunConfig())

    # Test training
    # Note that most models will require many more steps to fully converge. We
    # have used a small number of steps here to keep the running time small.
    estimator.train(input_fn=input_fn_builder.train_input_fn, steps=75)
    test_evaluation = estimator.evaluate(
        input_fn=input_fn_builder.test_input_fn, steps=1)
    test_loss = test_evaluation["loss"]
    tf.compat.v1.logging.warn("Final test loss: %f", test_loss)
    self.assertLess(test_loss, max_loss)
    if loss == ar_model.ARModel.SQUARED_LOSS:
      # Test that the evaluation loss is reported without input scaling.
      self.assertAllClose(
          test_loss,
          tf.math.reduce_mean(
              (test_evaluation["mean"] - test_evaluation["observed"])**2))

    # Test predict
    (predictions,) = tuple(
        estimator.predict(input_fn=input_fn_builder.prediction_input_fn))
    predicted_mean = predictions["mean"][:, 0]

    if loss == ar_model.ARModel.NORMAL_LIKELIHOOD_LOSS:
      variances = predictions["covariance"][:, 0]
      standard_deviations = tf.math.sqrt(variances)
      # Note that we may get tighter bounds with more training steps.
      true_values = input_fn_builder.true_values()
      errors = tf.math.abs(predicted_mean -
                            true_values) > 4 * standard_deviations
      fraction_errors = tf.math.reduce_mean(
          tf.cast(errors, tf.dtypes.float32))
      tf.compat.v1.logging.warn("Fraction errors: %f", self.evaluate(fraction_errors))

  def test_autoregression_squared(self):
    self.train_helper(input_window_size=15,
                      loss=ar_model.ARModel.SQUARED_LOSS)

  def test_autoregression_short_input_window(self):
    self.train_helper(input_window_size=8,
                      loss=ar_model.ARModel.SQUARED_LOSS)

  def test_autoregression_normal(self):
    self.train_helper(
        input_window_size=10,
        loss=ar_model.ARModel.NORMAL_LIKELIHOOD_LOSS,
        max_loss=50.)  # Just make sure there are no exceptions.

  def test_autoregression_normal_multiple_periods(self):
    self.train_helper(
        input_window_size=10,
        loss=ar_model.ARModel.NORMAL_LIKELIHOOD_LOSS,
        max_loss=2.0,
        periods=(25, 55))


if __name__ == "__main__":
  tf.test.main()
