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
"""extenders tests."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.feature_column import feature_column_lib as fc
from tensorflow.python.framework import constant_op
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.platform import test
from tensorflow_estimator.python.estimator import extenders
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator.canned import linear


def get_input_fn(x, y):

  def input_fn():
    dataset = tf.compat.v1.data.Dataset.from_tensor_slices({'x': x, 'y': y})
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    features = iterator.get_next()
    labels = features.pop('y')
    return features, labels

  return input_fn


class AddMetricsTest(tf.test.TestCase):

  def test_should_add_metrics(self):
    def _test_metric_fn(metric_fn):
      input_fn = get_input_fn(
          x=np.arange(4)[:, None, None], y=np.ones(4)[:, None])
      config = run_config.RunConfig(log_step_count_steps=1)
      estimator = linear.LinearClassifierV2([tf.feature_column.numeric_column('x')],
                                            config=config)

      estimator = extenders.add_metrics(estimator, metric_fn)

      estimator.train(input_fn=input_fn)
      metrics = estimator.evaluate(input_fn=input_fn)
      self.assertIn('mean_x', metrics)
      self.assertEqual(1.5, metrics['mean_x'])
      # assert that it keeps original estimators metrics
      self.assertIn('auc', metrics)

    def metric_fn(features):
      metric = metrics_module.Mean()
      metric.update_state(features['x'])
      return {'mean_x': metric}

    _test_metric_fn(metric_fn)

  def test_should_error_out_for_not_recognized_args(self):
    estimator = linear.LinearClassifierV2([tf.feature_column.numeric_column('x')])

    def metric_fn(features, not_recognized):
      _, _ = features, not_recognized
      return {}

    with self.assertRaisesRegexp(ValueError, 'not_recognized'):
      estimator = extenders.add_metrics(estimator, metric_fn)

  def test_all_supported_args(self):
    input_fn = get_input_fn(x=[[[0.]]], y=[[[1]]])
    estimator = linear.LinearClassifierV2([tf.feature_column.numeric_column('x')])

    def metric_fn(features, predictions, labels, config):
      self.assertIn('x', features)
      self.assertIsNotNone(labels)
      self.assertIn('logistic', predictions)
      self.assertTrue(isinstance(config, run_config.RunConfig))
      return {}

    estimator = extenders.add_metrics(estimator, metric_fn)

    estimator.train(input_fn=input_fn)
    estimator.evaluate(input_fn=input_fn)

  def test_all_supported_args_in_different_order(self):
    input_fn = get_input_fn(x=[[[0.]]], y=[[[1]]])
    estimator = linear.LinearClassifierV2([tf.feature_column.numeric_column('x')])

    def metric_fn(labels, config, features, predictions):
      self.assertIn('x', features)
      self.assertIsNotNone(labels)
      self.assertIn('logistic', predictions)
      self.assertTrue(isinstance(config, run_config.RunConfig))
      return {}

    estimator = extenders.add_metrics(estimator, metric_fn)

    estimator.train(input_fn=input_fn)
    estimator.evaluate(input_fn=input_fn)

  def test_all_args_are_optional(self):
    def _test_metric_fn(metric_fn):
      input_fn = get_input_fn(x=[[[0.]]], y=[[[1]]])
      estimator = linear.LinearClassifierV2([tf.feature_column.numeric_column('x')])
      estimator = extenders.add_metrics(estimator, metric_fn)

      estimator.train(input_fn=input_fn)
      metrics = estimator.evaluate(input_fn=input_fn)
      self.assertEqual(2., metrics['two'])

    def metric_fn():
      metric = metrics_module.Mean()
      metric.update_state(tf.constant([2.]))
      return {'two': metric}

    _test_metric_fn(metric_fn)

  def test_overrides_existing_metrics(self):
    def _test_metric_fn(metric_fn):
      input_fn = get_input_fn(x=[[[0.]]], y=[[[1]]])
      estimator = linear.LinearClassifierV2([tf.feature_column.numeric_column('x')])
      estimator.train(input_fn=input_fn)
      metrics = estimator.evaluate(input_fn=input_fn)
      self.assertNotEqual(2., metrics['auc'])

      estimator = extenders.add_metrics(estimator, metric_fn)
      metrics = estimator.evaluate(input_fn=input_fn)
      self.assertEqual(2., metrics['auc'])

    def metric_fn():
      metric = metrics_module.Mean()
      metric.update_state(tf.constant([2.]))
      return {'auc': metric}

    _test_metric_fn(metric_fn)


if __name__ == '__main__':
  tf.test.main()
