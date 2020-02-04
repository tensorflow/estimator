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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tempfile

import tensorflow as tf
import six

from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.feature_column import feature_column_lib as feature_column
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned.timeseries import ar_model
from tensorflow_estimator.python.estimator.canned.timeseries import estimators
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
from tensorflow_estimator.python.estimator.canned.timeseries import saved_model_utils


class _SeedRunConfig(estimator_lib.RunConfig):

  @property
  def tf_random_seed(self):
    return 3


def _generate_data():
  time = tf.range(20, dtype=tf.dtypes.int64)
  data = tf.reshape(tf.range(20, dtype=tf.dtypes.float32), (20, 1))
  exogenous = data
  return time, data, exogenous


def _build_input_fn_with_seed(seed):

  def map_to_dict(time, data, exogenous):
    return {
        feature_keys.TrainEvalFeatures.TIMES: time,
        feature_keys.TrainEvalFeatures.VALUES: data,
        "exogenous": exogenous
    }

  def batch_windows(time, data, exogenous):
    return tf.compat.v1.data.Dataset.zip((time, data, exogenous)).batch(
        16, drop_remainder=True)

  def input_fn():
    dataset = tf.compat.v1.data.Dataset.from_tensor_slices(_generate_data())
    dataset = dataset.window(16, shift=1, drop_remainder=True)
    dataset = dataset.shuffle(1000, seed=seed).repeat()
    dataset = dataset.flat_map(batch_windows).batch(16).map(map_to_dict)
    return dataset

  return input_fn


@test_util.run_v1_only("Currently incompatible with ResourceVariable")
class TimeSeriesRegressorTest(tf.test.TestCase):

  def _fit_restore_fit_test_template(self, estimator_fn, test_saved_model):
    """Tests restoring previously fit models."""
    temp_dir = self.get_temp_dir()
    model_dir = tempfile.mkdtemp(dir=temp_dir)
    exogenous_feature_columns = (
        tf.feature_column.numeric_column("exogenous"),
    )
    first_estimator = estimator_fn(model_dir, exogenous_feature_columns)
    train_input_fn = _build_input_fn_with_seed(2)
    eval_input_fn = _build_input_fn_with_seed(3)
    first_estimator.train(input_fn=train_input_fn, steps=1)
    first_evaluation = first_estimator.evaluate(
        input_fn=eval_input_fn, steps=1)
    first_loss_before_fit = first_evaluation["loss"]
    self.assertAllEqual(first_loss_before_fit, first_evaluation["average_loss"])
    self.assertAllEqual([], first_loss_before_fit.shape)
    first_estimator.train(input_fn=train_input_fn, steps=1)
    first_loss_after_fit = first_estimator.evaluate(
        input_fn=eval_input_fn, steps=1)["loss"]
    self.assertAllEqual([], first_loss_after_fit.shape)
    second_estimator = estimator_fn(model_dir, exogenous_feature_columns)
    second_estimator.train(input_fn=train_input_fn, steps=1)
    second_evaluation = second_estimator.evaluate(
        input_fn=eval_input_fn, steps=1)
    exogenous_values_ten_steps = {
        "exogenous": tf.range(10, dtype=tf.dtypes.float32)[None, :, None]
    }
    input_receiver_fn = first_estimator.build_raw_serving_input_receiver_fn()
    export_location = first_estimator.export_saved_model(temp_dir,
                                                         input_receiver_fn)
    if not test_saved_model:
      return
    with tf.Graph().as_default():
      with tf.compat.v1.Session() as sess:
        signatures = tf.compat.v1.saved_model.load(sess, [tf.saved_model.SERVING], export_location)
        # Test that prediction and filtering can continue from evaluation output
        _ = saved_model_utils.predict_continuation(
            continue_from=second_evaluation,
            steps=10,
            exogenous_features=exogenous_values_ten_steps,
            signatures=signatures,
            session=sess)
        times, values, _ = _generate_data()
        first_filtering = saved_model_utils.filter_continuation(
            continue_from=second_evaluation,
            features={
                feature_keys.FilteringFeatures.TIMES: times[None, -1] + 2,
                feature_keys.FilteringFeatures.VALUES: values[None, -1] + 2.,
                "exogenous": values[None, -1, None] + 12.
            },
            signatures=signatures,
            session=sess)
        # Test that prediction and filtering can continue from filtering output
        second_saved_prediction = saved_model_utils.predict_continuation(
            continue_from=first_filtering,
            steps=1,
            exogenous_features={
                "exogenous":
                    tf.range(1, dtype=tf.dtypes.float32)[None, :, None]
            },
            signatures=signatures,
            session=sess)
        self.assertEqual(
            times[-1] + 3,
            tf.compat.v1.squeeze(
                second_saved_prediction[feature_keys.PredictionResults.TIMES]))
        saved_model_utils.filter_continuation(
            continue_from=first_filtering,
            features={
                feature_keys.FilteringFeatures.TIMES: times[-1] + 3,
                feature_keys.FilteringFeatures.VALUES: values[-1] + 3.,
                "exogenous": values[-1, None] + 13.
            },
            signatures=signatures,
            session=sess)

        # Test cold starting
        six.assertCountEqual(
            self,
            [feature_keys.FilteringFeatures.TIMES,
             feature_keys.FilteringFeatures.VALUES,
             "exogenous"],
            signatures.signature_def[
                feature_keys.SavedModelLabels.COLD_START_FILTER].inputs.keys())
        batched_times = tf.tile(
            tf.range(30, dtype=tf.dtypes.int64)[None, :], (10, 1))
        batched_values = tf.ones([10, 30, 1])
        state = saved_model_utils.cold_start_filter(
            signatures=signatures,
            session=sess,
            features={
                feature_keys.FilteringFeatures.TIMES: batched_times,
                feature_keys.FilteringFeatures.VALUES: batched_values,
                "exogenous": 10. + batched_values
            })
        predict_times = math_ops.tile(
            tf.range(30, 45, dtype=tf.dtypes.int64)[None, :], (10, 1))
        predictions = saved_model_utils.predict_continuation(
            continue_from=state,
            times=predict_times,
            exogenous_features={
                "exogenous":
                    math_ops.tile(
                        tf.range(15, dtype=tf.dtypes.float32), (10,))
                    [None, :, None]
            },
            signatures=signatures,
            session=sess)
        self.assertAllEqual([10, 15, 1], predictions["mean"].shape)

  def disabled_test_time_series_regressor(self):
    def _estimator_fn(model_dir, exogenous_feature_columns):
      return estimators.TimeSeriesRegressor(
          model=ar_model.ARModel(
              periodicities=10, input_window_size=10, output_window_size=6,
              num_features=1,
              exogenous_feature_columns=exogenous_feature_columns,
              prediction_model_factory=functools.partial(
                  ar_model.LSTMPredictionModel,
                  num_units=10)),
          config=_SeedRunConfig(),
          model_dir=model_dir)

    self._fit_restore_fit_test_template(_estimator_fn, test_saved_model=True)

  def test_ar_lstm_regressor(self):
    def _estimator_fn(model_dir, exogenous_feature_columns):
      return estimators.LSTMAutoRegressor(
          periodicities=10,
          input_window_size=10,
          output_window_size=6,
          model_dir=model_dir,
          num_features=1,
          extra_feature_columns=exogenous_feature_columns,
          num_units=10,
          config=_SeedRunConfig())

    # LSTMAutoRegressor uses OneShotPredictionHead which does not work with
    # saved models.
    self._fit_restore_fit_test_template(_estimator_fn, test_saved_model=False)


if __name__ == "__main__":
  tf.test.main()
