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
"""Tests for ar_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.python.client import session
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned.timeseries import ar_model
from tensorflow_estimator.python.estimator.canned.timeseries.estimators import LSTMAutoRegressor
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import PredictionFeatures
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures


@test_util.run_v1_only("Currently incompatible with ResourceVariable")
class ARModelTest(test.TestCase):

  def test_wrong_window_size(self):
    estimator = LSTMAutoRegressor(
        periodicities=10,
        input_window_size=10,
        output_window_size=6,
        num_features=1)

    def _bad_window_size_input_fn():
      return ({TrainEvalFeatures.TIMES: [[1]],
               TrainEvalFeatures.VALUES: [[[1.]]]},
              None)
    def _good_data():
      return ({
          TrainEvalFeatures.TIMES:
              math_ops.range(16)[None, :],
          TrainEvalFeatures.VALUES:
              array_ops.reshape(math_ops.range(16), [1, 16, 1])
      }, None)

    with self.assertRaisesRegexp(ValueError, "set window_size=16"):
      estimator.train(input_fn=_bad_window_size_input_fn, steps=1)
    # Get a checkpoint for evaluation
    estimator.train(input_fn=_good_data, steps=1)
    with self.assertRaisesRegexp(ValueError, "requires a window of at least"):
      estimator.evaluate(input_fn=_bad_window_size_input_fn, steps=1)

  def test_predictions_direct_lstm(self):
    model = ar_model.ARModel(periodicities=2,
                             num_features=1,
                             num_time_buckets=10,
                             input_window_size=2,
                             output_window_size=2,
                             prediction_model_factory=functools.partial(
                                 ar_model.LSTMPredictionModel,
                                 num_units=16))
    with session.Session():
      predicted_values = model.predict({
          PredictionFeatures.TIMES: [[4, 6, 10]],
          PredictionFeatures.STATE_TUPLE: (
              [[1, 2]], [[[1.], [2.]]], [[[], []]])
      })
      variables.global_variables_initializer().run()
      self.assertAllEqual(predicted_values["mean"].eval().shape,
                          [1, 3, 1])

  def test_long_eval(self):
    model = ar_model.ARModel(periodicities=2,
                             num_features=1,
                             num_time_buckets=10,
                             input_window_size=2,
                             output_window_size=1)
    raw_features = {
        TrainEvalFeatures.TIMES: [[1, 3, 5, 7, 11]],
        TrainEvalFeatures.VALUES: [[[1.], [2.], [3.], [4.], [5.]]]}
    model.initialize_graph()
    with variable_scope.variable_scope("armodel"):
      raw_evaluation = model.define_loss(
          raw_features, mode=estimator_lib.ModeKeys.EVAL)
    with session.Session() as sess:
      variables.global_variables_initializer().run()
      raw_evaluation_evaled = sess.run(raw_evaluation)
      self.assertAllEqual([[5, 7, 11]],
                          raw_evaluation_evaled.prediction_times)
      for feature_name in raw_evaluation.predictions:
        self.assertAllEqual(
            [1, 3, 1],  # batch, window, num_features. The window size has 2
                        # cut off for the first input_window.
            raw_evaluation_evaled.predictions[feature_name].shape)

  def test_long_eval_discard_indivisible(self):
    model = ar_model.ARModel(periodicities=2,
                             num_features=1,
                             num_time_buckets=10,
                             input_window_size=2,
                             output_window_size=2)
    raw_features = {
        TrainEvalFeatures.TIMES: [[1, 3, 5, 7, 11]],
        TrainEvalFeatures.VALUES: [[[1.], [2.], [3.], [4.], [5.]]]}
    model.initialize_graph()
    raw_evaluation = model.define_loss(
        raw_features, mode=estimator_lib.ModeKeys.EVAL)
    with session.Session() as sess:
      variables.global_variables_initializer().run()
      raw_evaluation_evaled = sess.run(raw_evaluation)
      self.assertAllEqual([[7, 11]],
                          raw_evaluation_evaled.prediction_times)
      for feature_name in raw_evaluation.predictions:
        self.assertAllEqual(
            [1, 2, 1],  # batch, window, num_features. The window has two cut
                        # off for the first input window and one discarded so
                        # that the remainder is divisible into output windows.
            raw_evaluation_evaled.predictions[feature_name].shape)


if __name__ == "__main__":
  test.main()
