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

import numpy

from tensorflow.contrib.timeseries.python.timeseries import input_pipeline
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator as coordinator_lib
from tensorflow.python.training import queue_runner_impl
from tensorflow_estimator.python.estimator.canned.timeseries import math_utils
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures


class InputStatisticsTests(test.TestCase):

  def _input_statistics_test_template(
      self, stat_object, num_features, dtype, give_full_data,
      warmup_iterations=0, rtol=1e-6, data_length=500, chunk_size=4):
    graph = ops.Graph()
    with graph.as_default():
      numpy_dtype = dtype.as_numpy_dtype
      values = (
          (numpy.arange(data_length, dtype=numpy_dtype)[..., None]
           + numpy.arange(num_features, dtype=numpy_dtype)[None, ...])[None])
      times = 2 * (numpy.arange(data_length)[None]) - 3
      if give_full_data:
        stat_object.set_data((times, values))
      features = {TrainEvalFeatures.TIMES: times,
                  TrainEvalFeatures.VALUES: values}
      input_fn = input_pipeline.RandomWindowInputFn(
          batch_size=16, window_size=chunk_size,
          time_series_reader=input_pipeline.NumpyReader(features))
      statistics = stat_object.initialize_graph(
          features=input_fn()[0])
      with self.session(graph=graph) as session:
        variables.global_variables_initializer().run()
        coordinator = coordinator_lib.Coordinator()
        queue_runner_impl.start_queue_runners(session, coord=coordinator)
        for _ in range(warmup_iterations):
          # A control dependency should ensure that, for queue-based statistics,
          # a use of any statistic is preceded by an update of all adaptive
          # statistics.
          statistics.total_observation_count.eval()
        self.assertAllClose(
            range(num_features) + numpy.mean(numpy.arange(chunk_size))[None],
            statistics.series_start_moments.mean.eval(),
            rtol=rtol)
        self.assertAllClose(
            numpy.tile(numpy.var(numpy.arange(chunk_size))[None],
                       [num_features]),
            statistics.series_start_moments.variance.eval(),
            rtol=rtol)
        self.assertAllClose(
            numpy.mean(values[0], axis=0),
            statistics.overall_feature_moments.mean.eval(),
            rtol=rtol)
        self.assertAllClose(
            numpy.var(values[0], axis=0),
            statistics.overall_feature_moments.variance.eval(),
            rtol=rtol)
        self.assertAllClose(
            -3,
            statistics.start_time.eval(),
            rtol=rtol)
        self.assertAllClose(
            data_length,
            statistics.total_observation_count.eval(),
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
            give_full_data=False,
            warmup_iterations=1000,
            rtol=0.1)


if __name__ == "__main__":
  test.main()
