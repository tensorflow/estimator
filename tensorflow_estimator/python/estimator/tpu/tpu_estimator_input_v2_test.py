# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TPUEstimator."""

from absl import flags
from absl.testing import parameterized
import numpy as np
import six
import tensorflow.compat.v1 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python import data as dataset_lib
from tensorflow.python.layers import layers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.training import training
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.tpu import tpu_config
from tensorflow_estimator.python.estimator.tpu import tpu_estimator

FLAGS = flags.FLAGS

_TRAIN = model_fn_lib.ModeKeys.TRAIN
_EVAL = model_fn_lib.ModeKeys.EVAL
_PREDICT = model_fn_lib.ModeKeys.PREDICT


def create_run_config(iterations_per_loop, **kwargs):
  return tpu_config.RunConfig(
      master='',
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=iterations_per_loop,
          **kwargs),
  )


def dense_computation(features):
  return layers.dense(
      features, 1, kernel_initializer=init_ops.zeros_initializer())


def model_fn_global_step_incrementer(features, labels, mode, params):
  del params
  loss = None
  train_op = None
  predictions = dense_computation(features)
  if mode != _PREDICT:
    loss = losses.mean_squared_error(labels, predictions)
    optimizer = tf.tpu.CrossShardOptimizer(
        training.GradientDescentOptimizer(learning_rate=0.5))
    train_op = optimizer.minimize(loss, training.get_global_step())
  return tpu_estimator.TPUEstimatorSpec(
      mode,
      loss=loss,
      train_op=train_op,
      predictions={'predictions': predictions},
      export_outputs={
          'test': export_output.PredictOutput({
              'prediction': predictions
          })
      })


def dummy_input_fn_with_dataset(batch_size, fea_len=1, repeat=True, x=None):
  if x is None:
    x = np.random.normal(size=[batch_size, fea_len]).astype(np.float32)
  labels = [[2.0]] * batch_size

  dataset1 = dataset_lib.Dataset.from_tensor_slices(x)
  dataset2 = dataset_lib.Dataset.from_tensor_slices(labels)
  dataset = dataset_lib.Dataset.zip((dataset1, dataset2))
  if repeat:
    dataset = dataset.repeat()
  dataset = dataset.batch(batch_size, drop_remainder=True)

  def _map(x, y):
    return x, y

  return dataset.map(_map)


class TpuEstimatorInputV2Test(parameterized.TestCase, test.TestCase):

  @parameterized.parameters((2, 1), (None, 2))
  def test_batch_size(self, num_cores_per_replica, num_shards):
    input_fn_call_count = [0]
    run_config = create_run_config(
        iterations_per_loop=4,
        num_cores_per_replica=num_cores_per_replica,
        num_shards=num_shards,
        per_host_input_for_training=tpu_config.InputPipelineConfig.PER_HOST_V2)

    def _input_fn(params):
      input_fn_call_count[0] += 1
      expected_batch_size = 128 // num_shards
      self.assertEqual(expected_batch_size, params['batch_size'])
      return dummy_input_fn_with_dataset(batch_size=params['batch_size'])

    est = tpu_estimator.TPUEstimator(
        model_fn=model_fn_global_step_incrementer,
        config=run_config,
        train_batch_size=128)
    self.assertEqual(0, input_fn_call_count[0])
    est.train(_input_fn, steps=1)
    self.assertEqual(1, input_fn_call_count[0])

  def test_run_spatial_partition(self):
    input_fn_call_count = [0]
    run_config = create_run_config(
        iterations_per_loop=4,
        num_cores_per_replica=2,
        num_shards=1,
        input_partition_dims=[[1, 2], None],
        per_host_input_for_training=(
            tpu_config.InputPipelineConfig.PER_HOST_V2))

    def _input_fn(params):
      input_fn_call_count[0] += 1
      return dummy_input_fn_with_dataset(
          batch_size=params['batch_size'], fea_len=2)

    est = tpu_estimator.TPUEstimator(
        model_fn=model_fn_global_step_incrementer,
        config=run_config,
        train_batch_size=128)
    self.assertEqual(0, input_fn_call_count[0])
    est.train(_input_fn, steps=1)
    self.assertEqual(1, input_fn_call_count[0])

  def test_predict_mode(self):
    input_fn_call_count = [0]
    predict_batch_size = 128
    run_config = create_run_config(
        iterations_per_loop=4,
        num_cores_per_replica=2,
        num_shards=1,
        input_partition_dims=[[1, 2], None],
        per_host_input_for_training=(
            tpu_config.InputPipelineConfig.PER_HOST_V2))

    def _input_fn(params):
      input_fn_call_count[0] += 1
      return dummy_input_fn_with_dataset(
          batch_size=params['batch_size'], fea_len=2)

    est = tpu_estimator.TPUEstimator(
        model_fn=model_fn_global_step_incrementer,
        config=run_config,
        train_batch_size=128,
        predict_batch_size=predict_batch_size)

    self.assertEqual(0, input_fn_call_count[0])

    predictor = est.predict(_input_fn, yield_single_examples=False)
    prediction = six.next(predictor)

    self.assertEqual(1, input_fn_call_count[0])
    self.assertIn('predictions', prediction)
    self.assertEqual((predict_batch_size, 1), prediction['predictions'].shape)

    predictor = est.predict(_input_fn, yield_single_examples=True)
    prediction = six.next(predictor)

    self.assertEqual(2, input_fn_call_count[0])
    self.assertIn('predictions', prediction)
    self.assertEqual((1,), prediction['predictions'].shape)

  def test_evaluate_mode(self):
    input_fn_call_count = [0]
    eval_batch_size = 128
    run_config = create_run_config(
        iterations_per_loop=4,
        num_cores_per_replica=2,
        num_shards=1,
        input_partition_dims=[[1, 2], None],
        per_host_input_for_training=(
            tpu_config.InputPipelineConfig.PER_HOST_V2))

    def _input_fn(params):
      input_fn_call_count[0] += 1
      return dummy_input_fn_with_dataset(
          batch_size=params['batch_size'], fea_len=2)

    est = tpu_estimator.TPUEstimator(
        model_fn=model_fn_global_step_incrementer,
        config=run_config,
        train_batch_size=128,
        eval_batch_size=eval_batch_size)

    self.assertEqual(0, input_fn_call_count[0])
    est.evaluate(_input_fn, steps=1)
    self.assertEqual(1, input_fn_call_count[0])

if __name__ == '__main__':
  tf.disable_v2_behavior()
  test.main()
