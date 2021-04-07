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
"""Tests for TPUEstimator with model parallelism."""

from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python import data as dataset_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.layers import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.training import evaluation
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training
from tensorflow.python.tpu.device_assignment import device_assignment
from tensorflow.python.tpu.topology import Topology
from tensorflow.python.tpu import tpu_feed
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.tpu import tpu_config
from tensorflow_estimator.python.estimator.tpu import tpu_estimator
# pylint: enable=g-direct-tensorflow-import

FLAGS = flags.FLAGS

_TRAIN = model_fn_lib.ModeKeys.TRAIN
_EVAL = model_fn_lib.ModeKeys.EVAL
_PREDICT = model_fn_lib.ModeKeys.PREDICT

_PER_HOST = 'per_host_sharding'
_PER_SHARD = 'per_shard_sharding'
_UNSHARDED = 'unsharded'
_INPUT_PIPELINE_WITH_QUEUE_RUNNER = (
    'Input pipeline contains one or more QueueRunners')


def dense_computation(features):
  return layers.dense(
      features['x'], 1, kernel_initializer=init_ops.zeros_initializer())


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


def dummy_input_fn_with_dataset(batch_size, repeat=True, x=None):
  if x is None:
    x = np.random.normal(size=[batch_size, 1]).astype(np.float32)
  labels = [[2.0]] * batch_size

  dataset1 = dataset_lib.Dataset.from_tensor_slices(x)
  dataset2 = dataset_lib.Dataset.from_tensor_slices(labels)
  dataset = dataset_lib.Dataset.zip((dataset1, dataset2))
  if repeat:
    dataset = dataset.repeat()
  dataset = dataset.batch(batch_size, drop_remainder=True)

  def _map(x, y):
    return {'x': x}, y

  return dataset.map(_map)


def dummy_input_fn(batch_size, repeat=True):
  dataset = dummy_input_fn_with_dataset(batch_size, repeat)
  iterator = dataset_ops.make_one_shot_iterator(dataset)
  return iterator.get_next()


def create_run_config(iterations_per_loop, num_shards, num_cores_per_replica,
                      **kwargs):
  return tpu_config.RunConfig(
      master='',
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=iterations_per_loop,
          num_shards=num_shards,
          num_cores_per_replica=num_cores_per_replica,
          **kwargs))


class TPUEstimatorModelParallelismConstructorTest(test.TestCase):

  def test_fail_model_parallelism_for_per_core_input(self):
    run_config = create_run_config(
        iterations_per_loop=4,
        num_shards=1,
        num_cores_per_replica=2,
        per_host_input_for_training=False)
    with self.assertRaisesRegex(ValueError, 'Model parallelism only supports'):
      tpu_estimator.TPUEstimator(
          model_fn=model_fn_global_step_incrementer,
          config=run_config,
          train_batch_size=128)


class TPUEstimatorModelParallelismTrainingTest(test.TestCase):

  def _train_and_return_global_steps(self,
                                     iterations_per_loop,
                                     steps=None,
                                     max_steps=None,
                                     pre_train_steps=None,
                                     **kwargs):
    """Trains the model and returns the list of global steps after each loop."""

    def input_fn(params):
      return dummy_input_fn(params['batch_size'])

    def _model_fn(features, labels, mode, params):
      return model_fn_global_step_incrementer(features, labels, mode, params)

    run_config = create_run_config(
        iterations_per_loop=iterations_per_loop,
        num_shards=1,
        num_cores_per_replica=2,
        **kwargs)
    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        config=run_config,
        train_batch_size=16,
        eval_batch_size=16)

    class _TrainStepCheckHook(session_run_hook.SessionRunHook):
      """Check eval step counter after one session.run."""

      def __init__(self):
        """Constructs the run hook."""
        self._global_steps = []

      @property
      def global_steps(self):
        return self._global_steps

      def after_run(self, run_context, run_values):
        global_step = run_context.session.run(training.get_global_step())
        self._global_steps.append(global_step)

    if pre_train_steps:
      est.train(input_fn, steps=pre_train_steps)

    hook = _TrainStepCheckHook()
    est.train(input_fn, steps=steps, max_steps=max_steps, hooks=[hook])
    return hook.global_steps

  def test_train_steps_with_model_parallelism(self):
    # From scratch.
    global_steps_per_loop = self._train_and_return_global_steps(
        iterations_per_loop=40, steps=12)
    self.assertEqual([12], global_steps_per_loop)

    # From existing checkpoint.
    global_steps_per_loop = self._train_and_return_global_steps(
        iterations_per_loop=40, steps=12, pre_train_steps=3)
    self.assertEqual([15], global_steps_per_loop)


class TPUEstimatorModelParallelismEvaluationTest(test.TestCase):

  def _create_input_fn(self):

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    return _input_fn

  def _create_head(self, mode, loss, eval_metrics):
    """Creates a head returning `TPUEstimatorSpec` based on mode."""
    if mode == _EVAL:
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode, eval_metrics=eval_metrics, loss=loss)
    # Train
    optimizer = tf.tpu.CrossShardOptimizer(
        training.GradientDescentOptimizer(learning_rate=0.5))
    train_op = optimizer.minimize(loss, global_step=training.get_global_step())
    return tpu_estimator.TPUEstimatorSpec(
        mode=mode, train_op=train_op, loss=loss)

  def _metric_fn_on_cpu(self, labels, predictions):
    return {
        'mse': metrics_lib.mean_absolute_error(labels, predictions),
    }

  def _model_fn_with_eval_tensor_list(self, features, labels, mode, params):
    del params  # unused.
    predictions = layers.dense(
        features['x'], 1, kernel_initializer=init_ops.zeros_initializer())
    loss = losses.mean_squared_error(labels, predictions)

    return self._create_head(
        mode,
        loss,
        eval_metrics=(self._metric_fn_on_cpu, [labels, predictions]))

  def _model_fn_with_eval_dict(self, features, labels, mode, params):
    del params  # unused.
    predictions = layers.dense(
        features['x'], 1, kernel_initializer=init_ops.zeros_initializer())
    loss = losses.mean_squared_error(labels, predictions)

    return self._create_head(
        mode,
        loss,
        eval_metrics=(self._metric_fn_on_cpu, {
            'labels': labels,
            'predictions': predictions
        }))

  def _test_eval_steps(self, expected_eval_steps, iterations):

    run_config = create_run_config(
        iterations_per_loop=iterations, num_shards=1, num_cores_per_replica=2)
    est = tpu_estimator.TPUEstimator(
        model_fn=self._model_fn_with_eval_tensor_list,
        config=run_config,
        train_batch_size=16,
        eval_batch_size=16)

    est.train(self._create_input_fn(), steps=1)

    class _EvalStepCheckHook(session_run_hook.SessionRunHook):
      """Check eval step counter after one session.run.

      As the evaluation sets the eval iterations as the eval steps, the
      after_run should be invoked only once.
      """

      def __init__(self, iterations_per_loop, test_case):
        """Constructs the run hook."""
        self._iterations = iterations_per_loop
        self._invoked = False
        self._test_case = test_case

      def before_run(self, run_context):
        return session_run_hook.SessionRunArgs({
            'eval_steps': evaluation._get_or_create_eval_step()
        })

      def after_run(self, run_context, run_values):
        eval_steps = run_values.results['eval_steps']
        self._test_case.assertEqual(expected_eval_steps, eval_steps)
        self._test_case.assertFalse(self._invoked)
        self._invoked = True

    est.evaluate(
        self._create_input_fn(),
        steps=expected_eval_steps,
        hooks=[_EvalStepCheckHook(iterations, self)])

  def test_eval_metrics_with_tensor_list(self):
    run_config = create_run_config(
        iterations_per_loop=2, num_shards=1, num_cores_per_replica=2)
    est = tpu_estimator.TPUEstimator(
        model_fn=self._model_fn_with_eval_tensor_list,
        config=run_config,
        train_batch_size=16,
        eval_batch_size=16)

    est.train(self._create_input_fn(), steps=1)
    est.evaluate(self._create_input_fn(), steps=1)

  def test_eval_metrics_with_dict(self):
    run_config = create_run_config(
        iterations_per_loop=2, num_shards=1, num_cores_per_replica=2)
    est = tpu_estimator.TPUEstimator(
        model_fn=self._model_fn_with_eval_dict,
        config=run_config,
        train_batch_size=16,
        eval_batch_size=16)

    est.train(self._create_input_fn(), steps=1)
    est.evaluate(self._create_input_fn(), steps=1)

  def test_fail_with_wrong_num_shards(self):
    run_config = create_run_config(
        iterations_per_loop=2, num_shards=2, num_cores_per_replica=2)
    est = tpu_estimator.TPUEstimator(
        model_fn=self._model_fn_with_eval_tensor_list,
        config=run_config,
        train_batch_size=16,
        eval_batch_size=16)

    with self.assertRaisesRegex(ValueError, 'num_shards is not set correctly'):
      est.train(self._create_input_fn(), steps=1)


class TPUEstimatorModelParallelismInFeedTest(test.TestCase):

  def setUp(self):
    self._topology_2x2x2 = Topology(
        device_coordinates=np.array(
            [[[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 1],
              [1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0], [1, 1, 0, 1]]],
            dtype=np.int32),
        mesh_shape=np.array([2, 2, 1, 2], dtype=np.int32))

  def test_infeed_even_partition(self):
    """Tests even infeed tensors partition."""
    ds = device_assignment(
        self._topology_2x2x2, num_replicas=1, computation_shape=[1, 1, 1, 2])
    input_partition_dims = [[2, 1]]
    # pylint: disable=protected-access
    partitioned_infeed = tpu_feed._PartitionedInfeedQueue(
        number_of_tuple_elements=1,
        host_id=0,
        input_partition_dims=input_partition_dims,
        device_assignment=ds)
    x = array_ops.zeros((14, 5))
    tensors = partitioned_infeed._check_dims_and_partition_or_replicate_on_host(
        x, dims=input_partition_dims[0])
    self.assertEqual(2, len(tensors))
    self.assertEqual([(7, 5), (7, 5)], [t.shape for t in tensors])
    # pylint: enable=protected-access

  def test_infeed_uneven_partition(self):
    """Tests uneven infeed tensors partition."""
    ds = device_assignment(
        self._topology_2x2x2, num_replicas=1, computation_shape=[2, 2, 1, 2])
    input_partition_dims = [[4, 2]]
    # pylint: disable=protected-access
    partitioned_infeed = tpu_feed._PartitionedInfeedQueue(
        number_of_tuple_elements=1,
        host_id=0,
        input_partition_dims=input_partition_dims,
        device_assignment=ds)
    x = array_ops.zeros((14, 5))
    tensors = partitioned_infeed._check_dims_and_partition_or_replicate_on_host(
        x, dims=input_partition_dims[0])
    self.assertEqual(8, len(tensors))
    self.assertEqual((2, 2), tensors[-1].shape)
    # pylint: enable=protected-access

  def test_infeed_tailing_zero_partition(self):
    """Tests infeed tensors partition which causes zero-size tensors."""
    ds = device_assignment(
        self._topology_2x2x2, num_replicas=1, computation_shape=[1, 2, 1, 2])
    input_partition_dims = [[4, 1]]
    # pylint: disable=protected-access
    partitioned_infeed = tpu_feed._PartitionedInfeedQueue(
        number_of_tuple_elements=1,
        host_id=0,
        input_partition_dims=input_partition_dims,
        device_assignment=ds)
    x = array_ops.zeros((5, 5))
    tensors = partitioned_infeed._check_dims_and_partition_or_replicate_on_host(
        x, dims=input_partition_dims[0])
    self.assertEqual(4, len(tensors))
    self.assertEqual((1, 5), tensors[2].shape)
    self.assertEqual((0, 5), tensors[3].shape)
    # pylint: enable=protected-access

if __name__ == '__main__':
  test.main()
