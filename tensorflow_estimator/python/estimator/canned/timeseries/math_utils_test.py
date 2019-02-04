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
"""Tests for math_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator as coordinator_lib
from tensorflow.python.training import queue_runner_impl
from tensorflow_estimator.python.estimator.canned.timeseries import math_utils
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures


class InputStatisticsTests(test.TestCase):

  def _input_statistics_test_template(self,
                                      stat_object,
                                      num_features,
                                      dtype,
                                      warmup_iterations=0,
                                      rtol=1e-6,
                                      data_length=4):
    graph = ops.Graph()
    with graph.as_default():
      data_length_range = math_ops.range(data_length, dtype=dtype)
      num_features_range = math_ops.range(num_features, dtype=dtype)
      times = 2 * data_length_range[None, :] - 3
      values = (
          data_length_range[:, None] + num_features_range[None, :])[None, ...]
      features = {
          TrainEvalFeatures.TIMES: times,
          TrainEvalFeatures.VALUES: values,
      }
      statistics = stat_object.initialize_graph(features=features)
      with self.session(graph=graph) as session:
        variables.global_variables_initializer().run()
        coordinator = coordinator_lib.Coordinator()
        queue_runner_impl.start_queue_runners(session, coord=coordinator)
        for _ in range(warmup_iterations):
          # A control dependency should ensure that, for queue-based statistics,
          # a use of any statistic is preceded by an update of all adaptive
          # statistics.
          self.evaluate(statistics.total_observation_count)
        self.assertAllClose(
            range(num_features) + math_ops.reduce_mean(data_length_range)[None],
            self.evaluate(statistics.series_start_moments.mean),
            rtol=rtol)
        self.assertAllClose(
            array_ops.tile(
                math_ops.reduce_variance(data_length_range)[None],
                [num_features]),
            self.evaluate(statistics.series_start_moments.variance),
            rtol=rtol)
        self.assertAllClose(
            math_ops.reduce_mean(values[0], axis=0),
            self.evaluate(statistics.overall_feature_moments.mean),
            rtol=rtol)
        self.assertAllClose(
            math_ops.reduce_variance(values[0], axis=0),
            self.evaluate(statistics.overall_feature_moments.variance),
            rtol=rtol)
        self.assertAllClose(-3, self.evaluate(statistics.start_time), rtol=rtol)
        self.assertAllClose(
            data_length,
            self.evaluate(statistics.total_observation_count),
            rtol=rtol)
        coordinator.request_stop()
        coordinator.join()

  def test_queue(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      for num_features in [1, 2, 3]:
        self._input_statistics_test_template(
            math_utils.InputStatisticsFromMiniBatch(
                num_features=num_features, dtype=dtype),
            num_features=num_features,
            dtype=dtype,
            warmup_iterations=1000,
            rtol=0.1)


if __name__ == "__main__":
  test.main()
