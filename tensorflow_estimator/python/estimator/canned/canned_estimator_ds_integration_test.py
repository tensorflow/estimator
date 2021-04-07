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
"""Tests canned estimators with distribution strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import tempfile

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator.canned import dnn
from tensorflow_estimator.python.estimator.canned import dnn_linear_combined
from tensorflow_estimator.python.estimator.canned import linear
from tensorflow_estimator.python.estimator.extenders import add_metrics


class CannedEstimatorDistributionStrategyTest(tf.test.TestCase,
                                              parameterized.TestCase):

  def setUp(self):
    super(CannedEstimatorDistributionStrategyTest, self).setUp()
    np.random.seed(1337)
    tf.compat.v1.random.set_random_seed(1337)

    self._model_dir = tempfile.mkdtemp()

  def dataset_input_fn(self, x, y, batch_size, shuffle):

    def input_fn():
      dataset = tf.compat.v1.data.Dataset.from_tensor_slices((x, y))
      if shuffle:
        dataset = dataset.shuffle(batch_size)
      dataset = dataset.repeat(10).batch(batch_size)
      return dataset

    return input_fn

  @tf.__internal__.distribute.combinations.generate(
      tf.__internal__.test.combinations.combine(
          mode=['graph', 'eager'],
          distribution=[
              tf.__internal__.distribute.combinations.one_device_strategy,
              tf.__internal__.distribute.combinations
              .mirrored_strategy_with_gpu_and_cpu,
              tf.__internal__.distribute.combinations
              .mirrored_strategy_with_two_gpus,
          ],
          estimator_cls=[
              dnn_linear_combined.DNNLinearCombinedRegressorV2,
              dnn.DNNRegressorV2,
              linear.LinearRegressorV2,
          ]))
  def test_canned_estimator(self, distribution, estimator_cls):
    label_dimension = 2
    batch_size = 10
    # Adding one extra row (+ label_dimension) to test the last partial batch
    # use case.
    data = np.linspace(
        0.,
        2.,
        batch_size * label_dimension + label_dimension,
        dtype=np.float32)
    data = data.reshape(batch_size + 1, label_dimension)
    fc = tf.feature_column.numeric_column('x', shape=(2,))

    # Set kwargs based on the current canned estimator class.
    estimator_kw_args = {
        'model_dir': self._model_dir,
        'label_dimension': 2,
    }

    cls_args = inspect.getargspec(estimator_cls.__init__).args
    if 'hidden_units' in cls_args:
      estimator_kw_args['hidden_units'] = [2, 2]
    elif 'dnn_hidden_units' in cls_args:
      estimator_kw_args['dnn_hidden_units'] = [2, 2]

    if 'optimizer' in cls_args:
      estimator_kw_args['optimizer'] = 'SGD'
    else:
      estimator_kw_args['linear_optimizer'] = 'SGD'
      estimator_kw_args['dnn_optimizer'] = 'SGD'

    if 'feature_columns' in cls_args:
      estimator_kw_args['feature_columns'] = [fc]
    else:
      estimator_kw_args['linear_feature_columns'] = [fc]
      estimator_kw_args['dnn_feature_columns'] = [fc]

    def my_metrics(features):
      metric = tf.keras.metrics.Mean()
      metric.update_state(features['x'])
      return {'mean_x': metric}

    # Create a canned estimator and train to save a checkpoint.
    input_fn = self.dataset_input_fn(
        x={'x': data}, y=data, batch_size=batch_size, shuffle=False)
    canned_est = estimator_cls(**estimator_kw_args)
    canned_est.train(input_fn=input_fn)

    # Create a second canned estimator, warm-started from the first.
    del estimator_kw_args['model_dir']
    estimator_kw_args['warm_start_from'] = canned_est.model_dir
    warm_started_canned_est = estimator_cls(**estimator_kw_args)
    warm_started_canned_est.train(input_fn=input_fn)

    # Create a third canned estimator, warm-started from the first.
    input_fn = self.dataset_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size // distribution.num_replicas_in_sync,
        shuffle=False)
    estimator_kw_args['config'] = run_config.RunConfig(
        train_distribute=distribution, eval_distribute=distribution)
    warm_started_canned_est_with_ds = estimator_cls(**estimator_kw_args)
    warm_started_canned_est_with_ds.train(input_fn=input_fn)

    for variable_name in warm_started_canned_est.get_variable_names():
      self.assertAllClose(
          warm_started_canned_est_with_ds.get_variable_value(variable_name),
          warm_started_canned_est.get_variable_value(variable_name))

    warm_started_canned_est = add_metrics(warm_started_canned_est, my_metrics)
    warm_started_canned_est_with_ds = add_metrics(
        warm_started_canned_est_with_ds, my_metrics)

    scores = warm_started_canned_est.evaluate(input_fn)
    scores_with_ds = warm_started_canned_est_with_ds.evaluate(input_fn)
    self.assertAlmostEqual(scores['loss'], scores_with_ds['loss'], 5)
    self.assertAlmostEqual(scores['mean_x'], scores_with_ds['mean_x'], 5)


if __name__ == '__main__':
  tf.test.main()
