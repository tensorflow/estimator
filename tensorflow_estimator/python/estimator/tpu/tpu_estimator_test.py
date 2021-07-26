# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TPUEstimator.

To improve the performance, the test has been splitted into multiple parts

1. Integration         tpu_estimator_integration_test
2. Model Parallellsim  tpu_estimator_model_parallelism_test
3. Evaluation          tpu_estimator_evaluation_test
4. Export              tpu_estimator_export_test
5. Input Host v2       tpu_estimator_input_v2_test
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import os
import re
import tempfile

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python import data as dataset_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.layers import layers
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.gen_array_ops import reshape
from tensorflow.python.ops.losses import losses
from tensorflow.python.ops.random_ops import random_uniform
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.summary import summary as summary_lib
from tensorflow.python.tpu import topology as tf_topology
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.training import moving_averages
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.export import export
from tensorflow_estimator.python.estimator.export import export_output as export_output_lib
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.tpu import tpu_config
from tensorflow_estimator.python.estimator.tpu import tpu_estimator
# pylint: enable=g-direct-tensorflow-import


flags.DEFINE_integer('test_num_shards', 8, 'number of replicas to test')

FLAGS = flags.FLAGS

_TRAIN = model_fn_lib.ModeKeys.TRAIN
_EVAL = model_fn_lib.ModeKeys.EVAL
_PREDICT = model_fn_lib.ModeKeys.PREDICT

_PER_HOST = 'per_host_sharding'
_PER_SHARD = 'per_shard_sharding'
_UNSHARDED = 'unsharded'
_INPUT_PIPELINE_WITH_QUEUE_RUNNER = (
    'Input pipeline contains one or more QueueRunners')


def events_from_file(filepath):
  """Returns all events in a single event file.

  Args:
    filepath: Path to the event file.

  Returns:
    A list of all tf.compat.v1.Event protos in the event file.
  """
  records = list(tf_record.tf_record_iterator(filepath))
  result = []
  for r in records:
    event = event_pb2.Event()
    event.ParseFromString(r)
    result.append(event)
  return result


def dense_computation(features):
  x = features['x']
  if len(x.get_shape().as_list()) == 4:
    x = math_ops.reduce_sum(x, axis=[1, 2])
  return layers.dense(x, 1, kernel_initializer=init_ops.zeros_initializer())


def get_model_fn(export_tpu_tensor=True,
                 export_cpu_tensor=False,
                 tpu_estimator_spec=True):

  def model_fn(features, labels, mode, params):
    del params
    loss = None
    train_op = None

    predictions = dense_computation(features)
    export_outputs = None
    if mode != _PREDICT:
      loss = losses.mean_squared_error(labels, predictions)
      optimizer = tf.compat.v1.tpu.CrossShardOptimizer(
          training.GradientDescentOptimizer(learning_rate=0.5))
      train_op = optimizer.minimize(loss, training.get_global_step())
    else:
      if export_tpu_tensor:
        key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        export_outputs = {
            key: export_output_lib.PredictOutput({
                'prediction': predictions
            })
        }
      else:
        export_outputs = {}

      if export_cpu_tensor:

        def host_call(predictions):
          classes = string_ops.as_string(predictions, name='classes')
          classification_output = export_output_lib.ClassificationOutput(
              classes=classes)
          export_outputs['classification'] = classification_output

        tf.compat.v1.tpu.outside_compilation(host_call, predictions)

    if tpu_estimator_spec:
      spec_type = tpu_estimator.TPUEstimatorSpec
    else:
      spec_type = model_fn_lib.EstimatorSpec

    return spec_type(
        mode,
        loss=loss,
        train_op=train_op,
        predictions={'predictions': predictions},
        export_outputs=export_outputs)

  return model_fn


def dummy_input_fn_with_dataset(dataset_size,
                                repeat=True,
                                x=None,
                                batch_size=None):
  if batch_size is None:
    batch_size = dataset_size
  if x is None:
    x = np.random.normal(size=[dataset_size, 1]).astype(np.float32)
  labels = [[2.0]] * dataset_size

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


def create_run_config(iterations_per_loop, num_shards=None, **kwargs):
  return tpu_config.RunConfig(
      master='',
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=iterations_per_loop,
          num_shards=num_shards if num_shards else FLAGS.test_num_shards,
          **kwargs),
  )


class TPUEstimatorConstructorTest(test.TestCase):

  def test_reserved_key(self):
    run_config = create_run_config(iterations_per_loop=4)
    params = {'batch_size': 128}
    with self.assertRaisesRegex(ValueError, 'are reserved keys'):
      tpu_estimator.TPUEstimator(
          model_fn=get_model_fn(), config=run_config, params=params)

  def test_missing_train_batch_size(self):
    run_config = create_run_config(iterations_per_loop=4)
    with self.assertRaisesRegex(ValueError,
                                '`train_batch_size` cannot be `None`'):
      tpu_estimator.TPUEstimator(
          model_fn=get_model_fn(), config=run_config, params={})

  def test_invalid_batch_size(self):
    run_config = create_run_config(iterations_per_loop=4)
    with self.assertRaisesRegex(TypeError, 'must be int'):
      tpu_estimator.TPUEstimator(
          model_fn=get_model_fn(), config=run_config, train_batch_size=1.0)

  def test_batch_size_with_num_shards_for_per_core_input(self):
    input_fn_call_count = [0]
    run_config = create_run_config(
        iterations_per_loop=4, per_host_input_for_training=False)
    num_shards = run_config.tpu_config.num_shards

    def _input_fn(params):
      input_fn_call_count[0] += 1
      self.assertEqual(128 // num_shards, params['batch_size'])
      return dummy_input_fn(params['batch_size'])

    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(), config=run_config, train_batch_size=128)
    self.assertEqual(0, input_fn_call_count[0])
    est.train(_input_fn, steps=1)
    self.assertEqual(num_shards, input_fn_call_count[0])

  def test_batch_size_with_num_shards_for_per_host_input(self):
    input_fn_call_count = [0]
    run_config = create_run_config(
        iterations_per_loop=4, per_host_input_for_training=True)

    def _input_fn(params):
      input_fn_call_count[0] += 1
      self.assertEqual(128, params['batch_size'])
      return dummy_input_fn(params['batch_size'])

    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(), config=run_config, train_batch_size=128)
    self.assertEqual(0, input_fn_call_count[0])
    est.train(_input_fn, steps=1)
    self.assertEqual(1, input_fn_call_count[0])

  def test_train_batch_size_with_non_divisible_num_shards(self):
    run_config = create_run_config(iterations_per_loop=4)
    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(), config=run_config, train_batch_size=127)
    with self.assertRaisesRegex(ValueError, 'train.*must be divisible'):
      est.train(dummy_input_fn_with_dataset, steps=1)

  def test_train_batch_size_with_non_divisible_num_shards_broadcast_mode(self):
    input_fn_call_count = [0]
    run_config = create_run_config(
        iterations_per_loop=4,
        per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST)

    def _input_fn(params):
      input_fn_call_count[0] += 1
      self.assertEqual(127, params['batch_size'])
      return dummy_input_fn(params['batch_size'])

    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(), config=run_config, train_batch_size=127)
    self.assertEqual(0, input_fn_call_count[0])
    est.train(_input_fn, steps=1)
    self.assertEqual(1, input_fn_call_count[0])

  def test_eval_batch_size_with_non_divisible_num_shards(self):
    run_config = create_run_config(iterations_per_loop=4)
    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(),
        config=run_config,
        train_batch_size=64,
        eval_batch_size=127)
    with self.assertRaisesRegex(ValueError, 'eval.*must be divisible'):
      est.evaluate(dummy_input_fn_with_dataset, steps=1)

  def test_predict_batch_size_with_non_divisible_num_shards_broadcast_mode(
      self):
    run_config = create_run_config(
        iterations_per_loop=4,
        per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST)

    def _input_fn(params):
      return dummy_input_fn_with_dataset(params['batch_size'])

    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(),
        config=run_config,
        train_batch_size=64,
        predict_batch_size=127)
    est.train(_input_fn, steps=1)
    est.predict(_input_fn)

  def test_predict_batch_size_with_non_divisible_num_shards(self):
    run_config = create_run_config(iterations_per_loop=4)

    def _input_fn(params):
      return dummy_input_fn_with_dataset(params['batch_size'])

    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(),
        config=run_config,
        train_batch_size=64,
        predict_batch_size=127)
    est.train(_input_fn, steps=1)
    with self.assertRaisesRegex(ValueError, 'predict.*must be divisible'):
      list(est.predict(_input_fn))

  def test_invalid_num_shards(self):
    run_config = tpu_config.RunConfig(
        master='',
        tpu_config=tpu_config.TPUConfig(iterations_per_loop=2, num_shards=16))
    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(), config=run_config, train_batch_size=128)
    with self.assertRaisesRegex(ValueError, 'num_shards is not set correctly'):
      est.train(dummy_input_fn_with_dataset, steps=1)


class TPUEstimatorTPUContextTest(test.TestCase):

  def test_context_replicas(self):

    def _input_fn(params):
      batch_size = params['batch_size']
      context = params['context']
      self.assertEqual(FLAGS.test_num_shards, context.num_replicas)
      self.assertEqual(1, context.num_hosts)
      self.assertEqual(0, context.current_host)
      self.assertEqual(FLAGS.test_num_shards, context.num_of_replicas_per_host)
      return dummy_input_fn(batch_size)

    run_config = create_run_config(
        iterations_per_loop=4, per_host_input_for_training=False)

    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(), config=run_config, train_batch_size=16)
    est.train(_input_fn, steps=4)

  def _query_system(self, master_address, cluster_def, query_topology):
    del master_address, cluster_def, query_topology
    # construct an ideal, not real, topology for 4x4.
    topology = tf_topology.Topology(
        mesh_shape=[4, 4, 1, 2],
        device_coordinates=[
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 1],
                [2, 0, 0, 0],
                [2, 0, 0, 1],
                [3, 0, 0, 0],
                [3, 0, 0, 1],
            ],
            [
                [0, 1, 0, 0],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
                [1, 1, 0, 1],
                [2, 1, 0, 0],
                [2, 1, 0, 1],
                [3, 1, 0, 0],
                [3, 1, 0, 1],
            ],
            [
                [0, 2, 0, 0],
                [0, 2, 0, 1],
                [1, 2, 0, 0],
                [1, 2, 0, 1],
                [2, 2, 0, 0],
                [2, 2, 0, 1],
                [3, 2, 0, 0],
                [3, 2, 0, 1],
            ],
            [
                [0, 3, 0, 0],
                [0, 3, 0, 1],
                [1, 3, 0, 0],
                [1, 3, 0, 1],
                [2, 3, 0, 0],
                [2, 3, 0, 1],
                [3, 3, 0, 0],
                [3, 3, 0, 1],
            ],
        ],
    )
    return tpu_system_metadata_lib.TPUSystemMetadata(
        num_cores=32,
        num_hosts=4,
        num_of_cores_per_host=8,
        topology=topology,
        devices=[])

  def test_num_cores_per_replica_is_not_greater_than_num_cores_per_host(self):

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    with test.mock.patch.object(
        tpu_system_metadata_lib,
        '_query_tpu_system_metadata',
        side_effect=self._query_system):

      FLAGS.test_num_shards = 2
      run_config = create_run_config(
          iterations_per_loop=1, num_cores_per_replica=16)

      with self.assertRaisesRegex(
          ValueError,
          'Except the PER_HOST_V2 mode, the num of cores required by '
          'model parallelism specified by TPUConfig.num_cores_per_replica '
          'should be less than or equal to the num_cores_per_host. '
          'num_cores_per_replica: 16, num_cores_per_host: 8'):
        est = tpu_estimator.TPUEstimator(
            model_fn=get_model_fn(), config=run_config, train_batch_size=64)
        est.train(_input_fn, steps=1)

  def test_device_for_replica_fn(self):

    def _input_fn(params):
      batch_size = params['batch_size']
      context = params['context']

      with self.assertRaisesRegex(
          RuntimeError, 'This TPUContext instance must not be '
          'called from input_fn.'):
        context.device_assignment()

      for replica_id in range(context.num_replicas):
        (host_device, ordinal_id) = context.device_for_replica(replica_id)
        self.assertEqual('/task:0/device:CPU:0', host_device)
        self.assertEqual(ordinal_id, replica_id)

      return dummy_input_fn(batch_size)

    run_config = create_run_config(
        iterations_per_loop=4, per_host_input_for_training=True)

    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(), config=run_config, train_batch_size=16)
    est.train(_input_fn, steps=4)

  def test_input_deployment_for_per_host(self):
    fake_num_cores = 32
    fake_num_hosts = 4
    fake_num_cores_per_host = fake_num_cores // fake_num_hosts
    invocation_count = [0]
    global_batch_size = 16 * fake_num_cores

    def _input_fn(params):
      batch_size = params['batch_size']
      self.assertEqual(global_batch_size // fake_num_hosts, batch_size)

      context = params['context']
      current_invocation_count = invocation_count[0]

      (current_input_device, invocation_index_in_context, total_invocations,
       replicas_consumed_by_current_invocation) = (
           context.current_input_fn_deployment())
      self.assertEqual('/replica:0/task:0/device:CPU:0', current_input_device)
      self.assertEqual(current_invocation_count, invocation_index_in_context)
      self.assertEqual(current_invocation_count, context.current_host)
      self.assertEqual(fake_num_hosts, total_invocations)
      self.assertEqual(fake_num_cores_per_host,
                       replicas_consumed_by_current_invocation)

      # Use the invocation_count to track the number of invocations.
      invocation_count[0] = current_invocation_count + 1

      return dummy_input_fn(batch_size)

    with test.mock.patch.object(
        tpu_system_metadata_lib,
        '_query_tpu_system_metadata',
        side_effect=self._query_system):

      run_config = create_run_config(
          iterations_per_loop=4,
          num_shards=fake_num_cores,
          per_host_input_for_training=True)

      est = tpu_estimator.TPUEstimator(
          model_fn=get_model_fn(),
          config=run_config,
          train_batch_size=global_batch_size)

      # This exception is ok as we do not have sufficient TPU cores to run the
      # model. as far as the assert after it is correct, input pipeline checking
      # is done and successful.
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  'there are only 2 cores in the TPU topology'):
        est.train(_input_fn, steps=4)

      self.assertEqual(fake_num_hosts, invocation_count[0])

  def test_input_deployment_for_per_host_v2(self):
    fake_num_cores = 32
    fake_num_hosts = 4
    fake_num_cores_per_host = fake_num_cores // fake_num_hosts
    invocation_count = [0]
    global_batch_size = 16 * fake_num_cores

    def _input_fn(params):
      batch_size = params['batch_size']
      self.assertEqual(global_batch_size // fake_num_cores, batch_size)

      context = params['context']
      current_invocation_count = invocation_count[0]

      (current_input_device, invocation_index_in_context, total_invocations,
       replicas_consumed_by_current_invocation) = (
           context.current_input_fn_deployment())

      self.assertEqual('/replica:0/task:0/device:CPU:0', current_input_device)
      self.assertEqual(current_invocation_count, invocation_index_in_context)
      self.assertEqual(fake_num_hosts, total_invocations)
      self.assertEqual(current_invocation_count, context.current_host)
      self.assertEqual(fake_num_cores_per_host,
                       replicas_consumed_by_current_invocation)

      # Use the invocation_count to track the number of invocations.
      invocation_count[0] = current_invocation_count + 1

      return dummy_input_fn_with_dataset(batch_size)

    with test.mock.patch.object(
        tpu_system_metadata_lib,
        '_query_tpu_system_metadata',
        side_effect=self._query_system):

      run_config = create_run_config(
          iterations_per_loop=4,
          num_shards=fake_num_cores,
          per_host_input_for_training=tpu_config.InputPipelineConfig.PER_HOST_V2
      )

      est = tpu_estimator.TPUEstimator(
          model_fn=get_model_fn(),
          config=run_config,
          train_batch_size=global_batch_size)
      # This exception is ok as we do not have sufficient TPU cores to run the
      # model. as far as the assert after it is correct, input pipeline checking
      # is done and successful.
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  'there are only 2 cores in the TPU topology'):
        est.train(_input_fn, steps=4)

    self.assertEqual(fake_num_hosts, invocation_count[0])

  def test_input_deployment_for_per_host_v2_with_model_parallelism(self):
    fake_num_cores = 32
    fake_num_hosts = 4
    fake_num_cores_per_host = fake_num_cores // fake_num_hosts
    num_cores_per_replica = 2
    fake_num_replicas = fake_num_cores // num_cores_per_replica
    fake_num_replicas_per_host = (
        fake_num_cores_per_host // num_cores_per_replica)

    invocation_count = [0]
    global_batch_size = 16 * fake_num_cores

    def _input_fn(params):
      batch_size = params['batch_size']
      self.assertEqual(global_batch_size // fake_num_replicas, batch_size)

      context = params['context']
      current_invocation_count = invocation_count[0]

      (current_input_device, invocation_index_in_context, total_invocations,
       replicas_consumed_by_current_invocation) = (
           context.current_input_fn_deployment())

      self.assertEqual('/replica:0/task:0/device:CPU:0', current_input_device)
      self.assertEqual(current_invocation_count, invocation_index_in_context)
      self.assertEqual(current_invocation_count, context.current_host)
      self.assertEqual(fake_num_hosts, total_invocations)
      self.assertEqual(fake_num_replicas_per_host,
                       replicas_consumed_by_current_invocation)

      # Use the invocation_count to track the number of invocations.
      invocation_count[0] = current_invocation_count + 1

      return dummy_input_fn_with_dataset(batch_size)

    with test.mock.patch.object(
        tpu_system_metadata_lib,
        '_query_tpu_system_metadata',
        side_effect=self._query_system):

      run_config = create_run_config(
          iterations_per_loop=4,
          num_shards=fake_num_replicas,
          per_host_input_for_training=tpu_config.InputPipelineConfig
          .PER_HOST_V2,
          num_cores_per_replica=num_cores_per_replica)
      est = tpu_estimator.TPUEstimator(
          model_fn=get_model_fn(),
          config=run_config,
          train_batch_size=global_batch_size)
      # This exception is ok as we do not have sufficient TPU cores to run the
      # model. as far as the assert after it is correct, input pipeline checking
      # is done and successful.
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  'there are only 2 cores in the TPU topology'):
        est.train(_input_fn, steps=4)

    self.assertEqual(fake_num_hosts, invocation_count[0])

  def test_input_deployment_model_parallelism_cross_host_replica(self):
    fake_num_cores = 32
    fake_num_hosts = 4
    fake_num_cores_per_host = fake_num_cores // fake_num_hosts
    num_cores_per_replica = 16
    self.assertGreater(num_cores_per_replica, fake_num_cores_per_host)

    fake_num_replicas = fake_num_cores // num_cores_per_replica

    host_ids = []
    invocation_count = [0]
    global_batch_size = 16 * fake_num_cores

    def _input_fn(params):
      batch_size = params['batch_size']
      self.assertEqual(global_batch_size // fake_num_replicas, batch_size)

      context = params['context']
      current_invocation_count = invocation_count[0]

      (current_input_device, invocation_index_in_context, total_invocations,
       replicas_consumed_by_current_invocation) = (
           context.current_input_fn_deployment())

      self.assertEqual('/replica:0/task:0/device:CPU:0', current_input_device)
      self.assertEqual(current_invocation_count, invocation_index_in_context)
      host_ids.append(context.current_host)
      self.assertEqual(fake_num_replicas, total_invocations)
      self.assertEqual(1, replicas_consumed_by_current_invocation)

      # Use the invocation_count to track the number of invocations.
      invocation_count[0] = current_invocation_count + 1

      return dummy_input_fn_with_dataset(batch_size)

    with test.mock.patch.object(
        tpu_system_metadata_lib,
        '_query_tpu_system_metadata',
        side_effect=self._query_system):

      run_config = create_run_config(
          iterations_per_loop=4,
          num_shards=fake_num_replicas,
          per_host_input_for_training=tpu_config.InputPipelineConfig
          .PER_HOST_V2,
          num_cores_per_replica=num_cores_per_replica)
      est = tpu_estimator.TPUEstimator(
          model_fn=get_model_fn(),
          config=run_config,
          train_batch_size=global_batch_size)
      # This exception is ok as we do not have sufficient TPU cores to run the
      # model. as far as the assert after it is correct, input pipeline checking
      # is done and successful.
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  'there are only 2 cores in the TPU topology'):
        est.train(_input_fn, steps=4)

    self.assertEqual(fake_num_replicas, invocation_count[0])
    self.assertEqual([0, 2], host_ids)

  def test_input_deployment_for_broadcast_mode(self):
    invocation_count = [0]
    global_batch_size = 16

    def _input_fn(params):
      batch_size = params['batch_size']
      self.assertEqual(global_batch_size, batch_size)

      context = params['context']
      current_invocation_count = invocation_count[0]

      (current_input_device, invocation_index_in_context, total_invocations,
       replicas_consumed_by_current_invocation) = (
           context.current_input_fn_deployment())

      self.assertEqual('/replica:0/task:0/device:CPU:0', current_input_device)
      self.assertEqual(current_invocation_count, invocation_index_in_context)
      self.assertEqual(1, total_invocations)
      self.assertEqual(FLAGS.test_num_shards,
                       replicas_consumed_by_current_invocation)

      # Use the invocation_count to track the number of invocations.
      invocation_count[0] = current_invocation_count + 1

      return dummy_input_fn_with_dataset(batch_size)

    run_config = create_run_config(
        iterations_per_loop=4,
        per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST)

    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(),
        config=run_config,
        train_batch_size=global_batch_size)
    est.train(_input_fn, steps=4)
    self.assertEqual(1, invocation_count[0])

  def test_input_deployment_for_eval_broadcast_mode(self):
    invocation_count = [0]
    global_batch_size = 16
    num_cores = FLAGS.test_num_shards

    def _input_fn(params, is_training=True):
      batch_size = params['batch_size']
      self.assertEqual(global_batch_size, batch_size)

      context = params['context']
      current_invocation_count = invocation_count[0]

      (current_input_device, invocation_index_in_context, total_invocations,
       replicas_consumed_by_current_invocation) = (
           context.current_input_fn_deployment())

      self.assertEqual('/replica:0/task:0/device:CPU:0', current_input_device)
      if is_training:
        self.assertEqual(current_invocation_count, invocation_index_in_context)
      else:
        self.assertEqual(current_invocation_count - 1,
                         invocation_index_in_context)
      self.assertEqual(1, total_invocations)
      self.assertEqual(num_cores, replicas_consumed_by_current_invocation)

      # Use the invocation_count to track the number of invocations.
      invocation_count[0] = current_invocation_count + 1

      return dummy_input_fn_with_dataset(batch_size)

    run_config = create_run_config(
        iterations_per_loop=4,
        per_host_input_for_training=True,
        eval_training_input_configuration=tpu_config.InputPipelineConfig.SLICED)

    def _assert_model_fn(features, labels, mode, params):
      actual_model_fn = get_model_fn()
      per_replica_batch_size = params['batch_size']
      self.assertEqual(per_replica_batch_size,
                       global_batch_size // num_cores)
      return actual_model_fn(features, labels, mode, params)

    est = tpu_estimator.TPUEstimator(
        model_fn=_assert_model_fn,
        config=run_config,
        train_batch_size=global_batch_size,
        eval_batch_size=global_batch_size)
    est.train(functools.partial(_input_fn, is_training=True), steps=1)
    self.assertEqual(1, invocation_count[0])
    est.evaluate(functools.partial(_input_fn, is_training=False), steps=1)
    self.assertEqual(2, invocation_count[0])

  def test_input_deployment_for_per_core(self):
    fake_num_cores = 32
    fake_num_hosts = 4
    fake_num_cores_per_host = fake_num_cores // fake_num_hosts
    invocation_count = [0]
    global_batch_size = 16 * fake_num_cores

    def _input_fn(params):
      batch_size = params['batch_size']
      self.assertEqual(global_batch_size // fake_num_cores, batch_size)

      context = params['context']
      current_invocation_count = invocation_count[0]

      (current_input_device, invocation_index_in_context, total_invocations,
       replicas_consumed_by_current_invocation) = (
           context.current_input_fn_deployment())

      self.assertEqual('/replica:0/task:0/device:CPU:0', current_input_device)
      self.assertEqual(current_invocation_count, invocation_index_in_context)
      self.assertEqual(current_invocation_count // fake_num_cores_per_host,
                       context.current_host)
      self.assertEqual(fake_num_cores, total_invocations)
      self.assertEqual(1, replicas_consumed_by_current_invocation)

      # Use the invocation_count to track the number of invocations.
      invocation_count[0] = current_invocation_count + 1

      return dummy_input_fn(batch_size)

    with test.mock.patch.object(
        tpu_system_metadata_lib,
        '_query_tpu_system_metadata',
        side_effect=self._query_system):

      run_config = create_run_config(
          iterations_per_loop=4,
          num_shards=fake_num_cores,
          per_host_input_for_training=False)

      est = tpu_estimator.TPUEstimator(
          model_fn=get_model_fn(),
          config=run_config,
          train_batch_size=global_batch_size)
      # This exception is ok as we do not have sufficient TPU cores to run the
      # model. as far as the assert after it is correct, input pipeline checking
      # is done and successful.
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  'there are only 2 cores in the TPU topology'):
        est.train(_input_fn, steps=4)

    self.assertEqual(fake_num_cores, invocation_count[0])

  def test_hparams_as_params(self):

    def _input_fn(params):
      batch_size = params['batch_size']
      context = params['context']
      self.assertEqual(FLAGS.test_num_shards, context.num_replicas)
      return dummy_input_fn(batch_size)

    run_config = create_run_config(
        iterations_per_loop=4, per_host_input_for_training=False)

    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(),
        params={},
        config=run_config,
        train_batch_size=16)
    est.train(_input_fn, steps=4)


class TPUEstimatorInputFnTest(parameterized.TestCase):

  def setUp(self):
    # TODO(b/65703635): Remove setting/restoring the constant here.
    # As we are transitioning from deprecated mode to new mode. We have to
    # test both cases to ensure we do not break clients.
    super(TPUEstimatorInputFnTest, self).setUp()
    self._old_value = tpu_estimator._WRAP_INPUT_FN_INTO_WHILE_LOOP

  def tearDown(self):
    super(TPUEstimatorInputFnTest, self).tearDown()
    tpu_estimator._WRAP_INPUT_FN_INTO_WHILE_LOOP = self._old_value

  # Use 10 to test TPUEstimator is correctly concatenating small tensors.
  @parameterized.parameters(1, 10)
  def test_succeed_with_dataset(self, num_features):
    tpu_estimator._WRAP_INPUT_FN_INTO_WHILE_LOOP = True

    def _input_fn(params):
      batch_size = params['batch_size']
      x = np.random.normal(size=[batch_size, 1]).astype(np.float32)
      x1 = np.random.normal(size=[batch_size, 1]).astype(np.int32)
      labels = [[2.0]] * batch_size

      dataset1 = dataset_lib.Dataset.from_tensor_slices(x)
      dataset2 = dataset_lib.Dataset.from_tensor_slices(x1)
      dataset3 = dataset_lib.Dataset.from_tensor_slices(labels)
      dataset = dataset_lib.Dataset.zip((dataset1, dataset2, dataset3))

      def _map_fn(x, x1, y):
        xs = {}
        for i in range(num_features):
          xs['x' * (i + 1)] = array_ops.identity(x)
          xs['x1' * (i + 1)] = array_ops.identity(x1)
        return xs, y

      dataset = dataset.map(_map_fn)
      dataset = dataset.repeat()
      dataset = dataset.batch(batch_size, drop_remainder=True)
      return dataset

    run_config = create_run_config(
        iterations_per_loop=4, per_host_input_for_training=True)
    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(), config=run_config, train_batch_size=16)
    est.train(_input_fn, steps=4)

  def test_succeed_with_input_return_features_and_labels_with_dataset(self):
    tpu_estimator._WRAP_INPUT_FN_INTO_WHILE_LOOP = True

    def _input_fn(params):
      batch_size = params['batch_size']
      return dummy_input_fn(batch_size)

    run_config = create_run_config(
        iterations_per_loop=4, per_host_input_for_training=False)
    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(), config=run_config, train_batch_size=16)
    est.train(_input_fn, steps=4)

  def test_fail_with_queue_based_input_fn_in_while_loop(self):
    tpu_estimator._WRAP_INPUT_FN_INTO_WHILE_LOOP = True

    data = np.arange(40, dtype=np.float32).reshape(40, 1)
    x = {'x': data}
    y = data * 2.0

    def input_fn(params):
      batch_size = params['batch_size']
      return numpy_io.numpy_input_fn(
          x, y, batch_size=batch_size, shuffle=False, num_epochs=None)()

    run_config = create_run_config(
        iterations_per_loop=4, per_host_input_for_training=False)
    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(), config=run_config, train_batch_size=16)

    with self.assertRaisesRegex(RuntimeError,
                                _INPUT_PIPELINE_WITH_QUEUE_RUNNER):
      est.train(input_fn, steps=4)

  def test_warning_with_queue_based_input_fn(self):
    tpu_estimator._WRAP_INPUT_FN_INTO_WHILE_LOOP = False

    data = np.arange(40, dtype=np.float32).reshape(40, 1)
    x = {'x': data}
    y = data * 2.0

    def input_fn(params):
      batch_size = params['batch_size']
      return numpy_io.numpy_input_fn(
          x, y, batch_size=batch_size, shuffle=False, num_epochs=None)()

    run_config = create_run_config(iterations_per_loop=4)
    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(), config=run_config, train_batch_size=16)

    with test.mock.patch.object(logging, 'warn') as mock_log:
      est.train(input_fn, steps=4)
      self.assertRegex(
          str(mock_log.call_args), _INPUT_PIPELINE_WITH_QUEUE_RUNNER)

  def test_nested_inputs_dict(self):
    self.help_test_nested_inputs(nest_type='dict')

  def test_nested_inputs_tuple(self):
    self.help_test_nested_inputs(nest_type='tuple')

  def test_nested_inputs_namedtuple(self):
    self.help_test_nested_inputs(nest_type='namedtuple')

  def help_test_nested_inputs(self, nest_type):
    self.assertIn(nest_type, ['dict', 'tuple', 'namedtuple'])
    tpu_estimator._WRAP_INPUT_FN_INTO_WHILE_LOOP = True

    class MyTuple(collections.namedtuple('MyTuple', ['a', 'b'])):
      pass

    def model_fn(features, labels, mode, params):
      del params
      if nest_type == 'dict':
        inputs = features['x']
      elif nest_type == 'tuple':
        inputs = features
      elif nest_type == 'namedtuple':
        inputs = tuple(features)
      else:
        inputs = features
      predictions = layers.dense(
          inputs[0], 1, kernel_initializer=init_ops.zeros_initializer())
      loss = losses.mean_squared_error(labels, predictions)
      export_outputs = None
      optimizer = tf.compat.v1.tpu.CrossShardOptimizer(
          training.GradientDescentOptimizer(learning_rate=0.5))
      train_op = optimizer.minimize(loss, training.get_global_step())

      return tpu_estimator.TPUEstimatorSpec(
          mode, loss=loss, train_op=train_op, export_outputs=export_outputs)

    def _input_fn(params):
      batch_size = params['batch_size']

      x = dataset_ops.Dataset.from_tensor_slices(
          (random_uniform([4, 1]),
           random_uniform([4, 1], maxval=100, dtype=dtypes.float32)))
      dataset_labels = dataset_ops.Dataset.from_tensor_slices(
          random_uniform([4, 1]))
      dataset = dataset_ops.Dataset.zip((x, dataset_labels))

      def _map_fn(x, y):
        if nest_type == 'dict':
          return {'x': x}, y
        elif nest_type == 'tuple':
          return tuple(x), y
        elif nest_type == 'namedtuple':
          return MyTuple(*x), y
        else:
          return x, y

      dataset = dataset.map(_map_fn)
      dataset = dataset.batch(batch_size, drop_remainder=True)
      dataset = dataset.repeat(-1)
      return dataset

    run_config = create_run_config(iterations_per_loop=4)
    est = tpu_estimator.TPUEstimator(
        model_fn=model_fn, config=run_config, train_batch_size=4)
    est.train(_input_fn, steps=4)


class _DummyHook(session_run_hook.SessionRunHook):
  """Check whether this hook is called or not."""

  def __init__(self):
    """Constructs the run hook."""
    self._called = False

  def after_create_session(self, sees, coord):
    del sees, coord
    self._called = True

  @property
  def called(self):
    return self._called


class TPUEstimatorModelFnTest(test.TestCase):

  def test_succeed_with_missing_labels(self):

    def _model_fn(features, mode, params):
      labels = features.pop('y')
      return get_model_fn()(features, labels, mode, params)

    def _input_fn_without_labels(params):
      batch_size = params['batch_size']
      features, labels = dummy_input_fn(batch_size)
      return {'x': features['x'], 'y': labels}

    run_config = create_run_config(iterations_per_loop=4)
    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn, config=run_config, train_batch_size=16)
    est.train(_input_fn_without_labels, steps=1)

  def test_succeed_with_log_step_count_steps_none(self):

    def _model_fn(features, mode, params):
      labels = features.pop('y')
      return get_model_fn()(features, labels, mode, params)

    def _input_fn_without_labels(params):
      batch_size = params['batch_size']
      features, labels = dummy_input_fn(batch_size)
      return {'x': features['x'], 'y': labels}

    run_config = create_run_config(iterations_per_loop=4)
    run_config = run_config.replace(log_step_count_steps=None)
    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn, config=run_config, train_batch_size=16)
    est.train(_input_fn_without_labels, steps=1)

  def test_missing_labels_in_model_fn_not_input_fn(self):

    def _model_fn(features, mode, params):
      del features, mode, params  # unused.
      return tpu_estimator.TPUEstimatorSpec()

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    run_config = create_run_config(iterations_per_loop=4)
    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn, config=run_config, train_batch_size=16)

    with self.assertRaisesRegex(
        ValueError,
        'model_fn does not take labels, but input_fn returns labels'):
      est.train(_input_fn, steps=1)

  def test_missing_params(self):

    def _model_fn(features, labels, mode):
      del features, labels, mode  # unused.
      return tpu_estimator.TPUEstimatorSpec()

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    run_config = create_run_config(iterations_per_loop=4)
    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn, config=run_config, train_batch_size=16)

    with self.assertRaisesRegex(ValueError,
                                'model_fn .* does not include params'):
      est.train(_input_fn, steps=1)

  def test_invalid_arg(self):

    def _model_fn(features, labels, invalid_arg):
      del features, labels, invalid_arg  # unused.
      return tpu_estimator.TPUEstimatorSpec()

    with self.assertRaisesRegex(ValueError,
                                'model_fn .* has following not expected args'):
      run_config = create_run_config(iterations_per_loop=4)
      tpu_estimator.TPUEstimator(
          model_fn=_model_fn, config=run_config, train_batch_size=16)

  def test_valid_training_hook(self):
    run_config = create_run_config(iterations_per_loop=4)
    dummy_hook = _DummyHook()

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    def _model_fn(features, labels, mode, params):
      spec = get_model_fn()(features, labels, mode, params)
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          train_op=spec.train_op,
          loss=spec.loss,
          training_hooks=[dummy_hook])

    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        config=run_config,
        train_batch_size=2 * FLAGS.test_num_shards)

    est.train(input_fn=_input_fn, steps=1)
    self.assertTrue(dummy_hook.called)

  def test_valid_eval_hook(self):
    run_config = create_run_config(iterations_per_loop=4)
    dummy_hook = _DummyHook()

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    def _model_fn(features, labels, mode, params):
      spec = get_model_fn()(features, labels, mode, params)
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          train_op=spec.train_op,
          loss=spec.loss,
          evaluation_hooks=[dummy_hook])

    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        config=run_config,
        train_batch_size=2 * FLAGS.test_num_shards,
        eval_batch_size=2 * FLAGS.test_num_shards)

    est.evaluate(input_fn=_input_fn, steps=1)
    self.assertTrue(dummy_hook.called)

  def test_valid_prediction_hook(self):
    run_config = create_run_config(iterations_per_loop=4)
    dummy_hook = _DummyHook()

    def _input_fn(params):
      return dummy_input_fn_with_dataset(params['batch_size'], repeat=False)

    def _model_fn(features, labels, mode, params):
      del labels, params
      predictions = dense_computation(features)
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          train_op=None,
          loss=None,
          predictions={'predictions': predictions},
          prediction_hooks=[dummy_hook])

    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        config=run_config,
        train_batch_size=2 * FLAGS.test_num_shards,
        predict_batch_size=2 * FLAGS.test_num_shards)

    list(est.predict(input_fn=_input_fn))
    self.assertTrue(dummy_hook.called)

  def test_invalid_training_chief_hook(self):
    run_config = create_run_config(iterations_per_loop=4)
    dummy_hook = session_run_hook.SessionRunHook()

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    def _model_fn(features, labels, mode, params):
      spec = get_model_fn()(features, labels, mode, params)
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          train_op=spec.train_op,
          loss=spec.loss,
          training_chief_hooks=[dummy_hook])

    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        config=run_config,
        train_batch_size=2 * FLAGS.test_num_shards)

    with self.assertRaisesRegex(
        ValueError, 'training_chief_hooks returned by '
        'EstimatorSpec is not supported in '
        'TPUEstimator'):
      est.train(input_fn=_input_fn, steps=1)

  def test_access_device_assignment_in_model_fn(self):

    def _model_fn(features, labels, mode, params):
      ctx = params['context']
      self.assertIsInstance(ctx.device_assignment,
                            tf.tpu.experimental.DeviceAssignment)
      return get_model_fn()(features, labels, mode, params)

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    FLAGS.test_num_shards //= 2
    run_config = create_run_config(
        iterations_per_loop=4, num_cores_per_replica=2)
    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn, config=run_config, train_batch_size=16)
    est.train(_input_fn, steps=4)
    FLAGS.test_num_shards *= 2

  def test_fail_to_call_deployment_in_model_fn(self):

    def _model_fn(features, labels, mode, params):
      ctx = params['context']
      with self.assertRaisesRegex(
          RuntimeError, 'This TPUContext instance must not be '
          'called from model_fn.'):
        ctx.current_input_fn_deployment()
      return get_model_fn()(features, labels, mode, params)

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    run_config = create_run_config(iterations_per_loop=4)
    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn, config=run_config, train_batch_size=16)
    est.train(_input_fn, steps=4)


class TPUEstimatorPredictionTest(test.TestCase):

  def _test_train_and_predict(self, run_config, dataset_size,
                              input_tensor=None):
    """Trains the model and returns the list of global steps after each loop."""

    def train_input_fn(params):
      return dummy_input_fn_with_dataset(
          dataset_size,
          repeat=True,
          x=input_tensor,
          batch_size=params['batch_size'])

    def predict_input_fn(params):
      return dummy_input_fn_with_dataset(
          dataset_size,
          repeat=False,
          x=input_tensor,
          batch_size=params['batch_size'])

    def _model_fn(features, labels, mode, params):
      return get_model_fn()(features, labels, mode, params)

    batch_size = 16

    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        config=run_config,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        predict_batch_size=batch_size)

    est.train(train_input_fn, steps=1)
    predictions = list(est.predict(predict_input_fn))
    if (run_config.tpu_config.per_host_input_for_training == tpu_config
        .InputPipelineConfig.BROADCAST):
      expected_size = batch_size
    elif (run_config.tpu_config.per_host_input_for_training == tpu_config
          .InputPipelineConfig.PER_HOST_V2):
      expected_size = dataset_size
    else:
      expected_size = batch_size

    self.assertEqual(expected_size, len(predictions))

  def _construct_run_config(self,
                            mode,
                            num_shards=2,
                            input_partition_dims=None,
                            num_cores_per_replica=None):

    return create_run_config(
        iterations_per_loop=4,
        num_shards=num_shards,
        per_host_input_for_training=mode,
        input_partition_dims=input_partition_dims,
        num_cores_per_replica=num_cores_per_replica)

  def test_train_and_predict_per_host_v1(self):
    self._test_train_and_predict(
        self._construct_run_config(tpu_config.InputPipelineConfig.PER_HOST_V1),
        16)

  def test_train_and_predict_per_host_v2_evenly_distributed(self):
    self._test_train_and_predict(
        self._construct_run_config(tpu_config.InputPipelineConfig.PER_HOST_V2),
        16)

  def test_train_and_predict_per_host_v2_not_evenly_distributed(self):
    self._test_train_and_predict(
        self._construct_run_config(tpu_config.InputPipelineConfig.PER_HOST_V2),
        24)

  def test_train_and_predict_with_input_partition(self):
    self._test_train_and_predict(
        self._construct_run_config(
            tpu_config.InputPipelineConfig.PER_HOST_V2,
            num_shards=1,
            input_partition_dims=[{
                'x': [1, 2, 1, 1]
            }, None],
            num_cores_per_replica=2), 16,
        np.zeros((16, 32, 32, 3), dtype=np.float32))

  def test_train_and_predict_broadcast(self):
    self._test_train_and_predict(
        self._construct_run_config(tpu_config.InputPipelineConfig.BROADCAST),
        16)

  def test_non_static_shape(self):

    def predict_input_fn(params):
      return dummy_input_fn_with_dataset(params['batch_size'], repeat=False)

    def _model_fn(features, labels, mode, params):
      spec = get_model_fn()(features, labels, mode, params)
      spec.predictions['dummy'] = array_ops.placeholder(
          dtypes.float32, shape=(None, 24))
      return spec

    batch_size = 16
    run_config = create_run_config(iterations_per_loop=4)
    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        config=run_config,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        predict_batch_size=batch_size)

    with self.assertRaisesRegex(ValueError, 'should be static'):
      list(est.predict(predict_input_fn))

  def test_predict_on_cpu(self):
    """Trains the model and returns the list of global steps after each loop."""

    def train_input_fn(params):
      return dummy_input_fn_with_dataset(params['batch_size'], repeat=True)

    def predict_input_fn(params):
      # A fixed input
      x = np.linspace(
          0.0, 100.0, num=batch_size).reshape(batch_size, 1).astype(np.float32)

      return dummy_input_fn_with_dataset(
          params['batch_size'], repeat=False, x=x)

    def _model_fn(features, labels, mode, params):
      return get_model_fn()(features, labels, mode, params)

    batch_size = 16
    run_config = create_run_config(iterations_per_loop=4)
    tpu_est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        config=run_config,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        predict_batch_size=batch_size,
        use_tpu=True)

    tpu_est.train(train_input_fn, steps=1)
    tpu_predictions = [
        x['predictions'] for x in tpu_est.predict(predict_input_fn)
    ]
    self.assertEqual(batch_size * 1, len(tpu_predictions))

    cpu_est = tpu_estimator.TPUEstimator(
        model_dir=tpu_est.model_dir,  # To load the ckpt.
        model_fn=_model_fn,
        config=run_config,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        predict_batch_size=batch_size,
        use_tpu=False)
    cpu_predictions = [
        x['predictions'] for x in cpu_est.predict(predict_input_fn)
    ]
    self.assertEqual(batch_size * 1, len(cpu_predictions))

    self.assertAllClose(tpu_predictions, cpu_predictions, atol=0.01)

  def test_train_and_export(self):

    def train_input_fn(params):
      return dummy_input_fn_with_dataset(params['batch_size'], repeat=True)

    def _model_fn(features, labels, mode, params):
      return get_model_fn()(features, labels, mode, params)

    batch_size = 16
    run_config = create_run_config(iterations_per_loop=4)
    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        config=run_config,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        predict_batch_size=1)

    est.train(train_input_fn, steps=1)

    # Even though the predict_batch_size is 1, not divisible by the num_shards
    # (2 in this case), export_savedmodel should not trigger the TPU validation.
    # This test ensures that the predict mode is handled correctly inside
    # TPUEstimator.
    feature_spec = {'x': parsing_ops.FixedLenFeature([1], dtypes.float32)}
    serving_input_receiver_fn = (
        export.build_parsing_serving_input_receiver_fn(feature_spec))
    est.export_saved_model(
        tempfile.mkdtemp(dir=self.get_temp_dir()), serving_input_receiver_fn)


class TPUEstimatorTrainingTest(test.TestCase):

  def _train_and_return_global_steps(self,
                                     iterations_per_loop,
                                     steps=None,
                                     max_steps=None,
                                     pre_train_steps=None):
    """Trains the model and returns the list of global steps after each loop."""

    def input_fn(params):
      return dummy_input_fn(params['batch_size'])

    def _model_fn(features, labels, mode, params):
      return get_model_fn()(features, labels, mode, params)

    run_config = create_run_config(iterations_per_loop=iterations_per_loop)
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

  def test_train_steps_not_divisible_by_iterations(self):
    # From scratch.
    global_steps_per_loop = self._train_and_return_global_steps(
        iterations_per_loop=4, steps=10)
    self.assertEqual([4, 8, 10], global_steps_per_loop)

    # From existing checkpoint.
    global_steps_per_loop = self._train_and_return_global_steps(
        iterations_per_loop=4, steps=10, pre_train_steps=3)
    self.assertEqual([7, 11, 13], global_steps_per_loop)

  def test_train_steps_divisible_by_iterations(self):
    # From scratch.
    global_steps_per_loop = self._train_and_return_global_steps(
        iterations_per_loop=4, steps=12)
    self.assertEqual([4, 8, 12], global_steps_per_loop)

    # From existing checkpoint.
    global_steps_per_loop = self._train_and_return_global_steps(
        iterations_per_loop=4, steps=12, pre_train_steps=3)
    self.assertEqual([7, 11, 15], global_steps_per_loop)

  def test_train_steps_with_large_iterations(self):
    # From scratch.
    global_steps_per_loop = self._train_and_return_global_steps(
        iterations_per_loop=40, steps=12)
    self.assertEqual([12], global_steps_per_loop)

    # From existing checkpoint.
    global_steps_per_loop = self._train_and_return_global_steps(
        iterations_per_loop=40, steps=12, pre_train_steps=3)
    self.assertEqual([15], global_steps_per_loop)

  def test_train_max_steps_not_divisible_by_iterations(self):
    # From scratch.
    global_steps_per_loop = self._train_and_return_global_steps(
        iterations_per_loop=4, max_steps=10)
    self.assertEqual([4, 8, 10], global_steps_per_loop)

    # From existing checkpoint.
    global_steps_per_loop = self._train_and_return_global_steps(
        iterations_per_loop=4, max_steps=10, pre_train_steps=3)
    self.assertEqual([7, 10], global_steps_per_loop)

  def test_train_max_steps_divisible_by_iterations(self):
    # From scratch.
    global_steps_per_loop = self._train_and_return_global_steps(
        iterations_per_loop=4, max_steps=12)
    self.assertEqual([4, 8, 12], global_steps_per_loop)

    # From existing checkpoint.
    global_steps_per_loop = self._train_and_return_global_steps(
        iterations_per_loop=4, max_steps=15, pre_train_steps=3)
    self.assertEqual([7, 11, 15], global_steps_per_loop)

  def test_train_max_steps_with_large_iterations(self):
    # From scratch.
    global_steps_per_loop = self._train_and_return_global_steps(
        iterations_per_loop=40, max_steps=12)
    self.assertEqual([12], global_steps_per_loop)

    # From existing checkpoint.
    global_steps_per_loop = self._train_and_return_global_steps(
        iterations_per_loop=40, max_steps=12, pre_train_steps=3)
    self.assertEqual([12], global_steps_per_loop)

  def test_error_out_if_train_steps_is_float(self):
    with self.assertRaisesRegex(TypeError, 'must be int'):
      self._train_and_return_global_steps(iterations_per_loop=40, steps=12.3)

  def test_error_out_if_train_steps_is_invalid(self):
    with self.assertRaisesRegex(ValueError, 'Must specify.*> 0'):
      self._train_and_return_global_steps(iterations_per_loop=40, steps=-32)

  def test_error_out_if_train_max_steps_is_float(self):
    with self.assertRaisesRegex(TypeError, 'must be int'):
      self._train_and_return_global_steps(
          iterations_per_loop=40, max_steps=12.3)

  def test_error_out_if_train_max_steps_is_invalid(self):
    with self.assertRaisesRegex(ValueError, 'Must specify.*> 0'):
      self._train_and_return_global_steps(iterations_per_loop=40, max_steps=-32)

  def test_warm_starts(self):

    def _make_model_fn(x, use_tpu):

      def _variable_creating_model_fn(features, labels, mode, params):
        del params
        loss = None
        train_op = None
        variable_scope.get_variable('x', initializer=x)
        predictions = dense_computation(features)
        loss = losses.mean_squared_error(labels, predictions)
        optimizer = training.GradientDescentOptimizer(learning_rate=0.5)
        if use_tpu:
          optimizer = tf.compat.v1.tpu.CrossShardOptimizer(optimizer)
        train_op = optimizer.minimize(loss, training.get_global_step())
        if use_tpu:
          return tpu_estimator.TPUEstimatorSpec(
              mode, loss=loss, train_op=train_op)
        else:
          return model_fn_lib.EstimatorSpec(
              mode,
              loss=constant_op.constant(1.),
              train_op=state_ops.assign_add(training.get_global_step(), 1))

      return _variable_creating_model_fn

    def input_fn(params):
      return dummy_input_fn(params.get('batch_size', 16))

    run_config = create_run_config(iterations_per_loop=1)
    tpu_est = tpu_estimator.TPUEstimator(
        model_fn=_make_model_fn(42., use_tpu=True),
        config=run_config,
        train_batch_size=16,
        eval_batch_size=16)
    tpu_est.train(input_fn, steps=10)

    warm_started_est = estimator_lib.Estimator(
        model_fn=_make_model_fn(36., use_tpu=False),
        warm_start_from=tpu_est.model_dir)
    warm_started_est.train(input_fn, steps=5)
    # warm_start is called after the model_fn, so x should have the value
    # from the checkpoint.
    self.assertEqual(42., warm_started_est.get_variable_value('x'))


class TPUEstimatorValidationTest(parameterized.TestCase, test.TestCase):

  def _query_system(self, master_address, cluster_def, query_topology):
    del master_address, cluster_def, query_topology
    return tpu_system_metadata_lib.TPUSystemMetadata(
        num_cores=16,
        num_hosts=2,
        num_of_cores_per_host=8,
        topology=None,
        devices=[])

  def test_error_if_cross_replica_sum_missing(self):

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    def _model_fn(features, labels, params):
      del params
      predictions = layers.dense(
          features['x'], 1, kernel_initializer=init_ops.zeros_initializer())
      loss = losses.mean_squared_error(labels, predictions)
      optimizer = training.GradientDescentOptimizer(learning_rate=0.5)
      train_op = optimizer.minimize(loss, training.get_global_step())

      return tpu_estimator.TPUEstimatorSpec(
          mode=None, loss=loss, train_op=train_op)

    run_config = create_run_config(iterations_per_loop=4)
    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn, train_batch_size=8, config=run_config, params={})

    with self.assertRaisesRegex(ValueError, 'model training on TPUs'):
      est.train(input_fn=_input_fn, steps=1)

  def test_no_error_if_cross_replica_sum_present(self):

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    run_config = create_run_config(iterations_per_loop=4)
    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(),
        train_batch_size=8,
        config=run_config,
        params={})
    est.train(input_fn=_input_fn, steps=1)

  def test_error_dynamic_shape_tensor_features_for_model(self):
    """Asserting that features Tensor to TPUEstimator model has static shape.

    """

    def _input_fn(params):
      features = reshape(
          math_ops.range(params['batch_size'] * 64, dtype=dtypes.float32),
          (params['batch_size'], 64))
      # Make features with dynamic shape by the help of random padding.
      padding = random_uniform([], minval=0, maxval=10, dtype=dtypes.int32)
      features = array_ops.pad(features, [(0, 0), (0, padding)])
      return dataset_lib.Dataset.from_tensor_slices(
          (features, math_ops.range(params['batch_size']) % 10)).repeat().batch(
              16, drop_remainder=True)

    def _model_fn(features, labels, mode, params):
      del labels
      del params
      if mode == _PREDICT:
        return tpu_estimator.TPUEstimatorSpec(
            mode=mode, predictions={'value': features})

    run_config = create_run_config(iterations_per_loop=4)
    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        train_batch_size=8,
        config=run_config,
        predict_batch_size=16,
        params={})
    with self.assertRaisesRegex(ValueError, 'features.*must have static'):
      list(est.predict(_input_fn))

  def test_error_dynamic_shape_dict_tensor_features_for_model(self):
    """Asserting that features dict to TPUEstimator model has static shape.

    """

    def _input_fn_dict(params):
      features = reshape(
          math_ops.range(params['batch_size'] * 64, dtype=dtypes.float32),
          (params['batch_size'], 64))
      # Make features with dynamic shape by the help of random padding.
      padding = random_uniform([], minval=0, maxval=10, dtype=dtypes.int32)
      features = array_ops.pad(features, [(0, 0), (0, padding)])
      dataset = dataset_lib.Dataset.from_tensor_slices(features)
      dataset = dataset.map(lambda v: {'key': v})
      return dataset.repeat().batch(16, drop_remainder=True)

    def _model_fn(features, labels, mode, params):
      del labels
      del params
      if mode == _PREDICT:
        return tpu_estimator.TPUEstimatorSpec(
            mode=mode, predictions={'value': features['key']})

    run_config = create_run_config(iterations_per_loop=4)
    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        train_batch_size=8,
        config=run_config,
        predict_batch_size=16,
        params={})
    with self.assertRaisesRegex(ValueError, 'features.*must have static.*'):
      list(est.predict(_input_fn_dict))

  def test_error_dynamic_shape_tensor_labels_for_model(self):
    """Asserting that labels to TPUEstimator model has static shape.

    """

    def _input_fn(params):
      features = reshape(
          math_ops.range(params['batch_size'] * 64, dtype=dtypes.float32),
          (params['batch_size'], 64))
      labels = reshape(
          math_ops.range(params['batch_size'] * 64, dtype=dtypes.float32),
          (params['batch_size'], 64))
      # Make labels with dynamic shape by the help of random padding.
      padding = random_uniform([], minval=0, maxval=10, dtype=dtypes.int32)
      labels = array_ops.pad(labels, [(0, 0), (0, padding)])
      dataset = dataset_lib.Dataset.from_tensor_slices((features, labels))
      return dataset.repeat().batch(16, drop_remainder=True)

    def _model_fn(features, labels, mode, params):
      del labels
      del params
      if mode == _PREDICT:
        return tpu_estimator.TPUEstimatorSpec(
            mode=mode, predictions={'value': features})

    run_config = create_run_config(iterations_per_loop=4)
    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        train_batch_size=8,
        config=run_config,
        predict_batch_size=16,
        params={})
    with self.assertRaisesRegex(ValueError, 'labels.*must have static'):
      list(est.predict(_input_fn))

  def test_error_dynamic_shape_dict_tensor_labels_for_model(self):
    """Asserting that labels dict to TPUEstimator model has static shape.

    """

    def _input_fn_dict(params):
      features = reshape(
          math_ops.range(params['batch_size'] * 64, dtype=dtypes.float32),
          (params['batch_size'], 64))
      labels = reshape(
          math_ops.range(params['batch_size'] * 64, dtype=dtypes.float32),
          (params['batch_size'], 64))
      # Make labels with dynamic shape by the help of random padding.
      padding = random_uniform([], minval=0, maxval=10, dtype=dtypes.int32)
      labels = array_ops.pad(labels, [(0, 0), (0, padding)])
      dataset = dataset_lib.Dataset.from_tensor_slices((features, labels))
      dataset = dataset.map(lambda f, l: ({'fkey': f}, {'lkey': l}))
      return dataset.repeat().batch(16, drop_remainder=True)

    def _model_fn(features, labels, mode, params):
      del labels
      del params
      if mode == _PREDICT:
        return tpu_estimator.TPUEstimatorSpec(
            mode=mode, predictions={'value': features['fkey']})

    run_config = create_run_config(iterations_per_loop=4)
    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        train_batch_size=8,
        config=run_config,
        predict_batch_size=16,
        params={})
    with self.assertRaisesRegex(ValueError, 'labels.*must have static*. shape'):
      list(est.predict(_input_fn_dict))

  def test_error_none_eval_batch_size_for_evaluation_mode(self):

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    with self.assertRaisesRegex(
        ValueError,
        'eval_batch_size in TPUEstimator constructor cannot be `None`'):
      est = tpu_estimator.TPUEstimator(
          model_fn=get_model_fn(),
          config=create_run_config(iterations_per_loop=4),
          train_batch_size=64,
          use_tpu=True)
      est.evaluate(_input_fn, steps=1)

  def test_error_none_predict_batch_size_for_prediction_mode(self):

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    with self.assertRaisesRegex(
        ValueError,
        'predict_batch_size in TPUEstimator constructor cannot be `None`'):
      est = tpu_estimator.TPUEstimator(
          model_fn=get_model_fn(),
          config=create_run_config(iterations_per_loop=4),
          train_batch_size=64,
          use_tpu=True)
      list(est.predict(_input_fn))

  @parameterized.parameters(
      (tpu_config.InputPipelineConfig.PER_HOST_V1, 'evaluate'),
      (tpu_config.InputPipelineConfig.PER_HOST_V1, 'predict'),
      (tpu_config.InputPipelineConfig.PER_HOST_V2, 'predict'))
  def test_error_num_hosts_and_replicas_larger_than_1_in_eval_and_predict_mode(
      self, input_pipeline_mode, predict_or_evaluate):

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    with test.mock.patch.object(
        tpu_system_metadata_lib,
        '_query_tpu_system_metadata',
        side_effect=self._query_system):

      run_config = create_run_config(
          iterations_per_loop=1, num_cores_per_replica=8,
          per_host_input_for_training=input_pipeline_mode)

      if predict_or_evaluate == 'evaluate':
        expected_error_re = ('TPUEstimator.evaluate is only supported '
                             'under three conditions')
      else:
        expected_error_re = ('TPUEstimator.predict is only supported '
                             'under three conditions')

      with self.assertRaisesRegex(ValueError, expected_error_re):
        est = tpu_estimator.TPUEstimator(
            model_fn=get_model_fn(),
            config=run_config,
            train_batch_size=32,
            eval_batch_size=32,
            predict_batch_size=32,
            use_tpu=True)
        if predict_or_evaluate == 'evaluate':
          est.evaluate(_input_fn, steps=1)
        else:
          list(est.predict(_input_fn))

  def test_evaluate_1host_and_replicas_larger_than_1_with_PER_HOST_V2(
      self):
    fake_num_cores = 32

    def _input_fn(params):
      batch_size = params['batch_size']
      x = np.random.normal(size=[batch_size, 20]).astype(np.float32)
      return dummy_input_fn_with_dataset(batch_size, repeat=False, x=x)

    with test.mock.patch.object(
        tpu_system_metadata_lib,
        '_query_tpu_system_metadata',
        side_effect=self._query_system):

      run_config = create_run_config(
          iterations_per_loop=4,
          num_shards=fake_num_cores // 2,
          per_host_input_for_training=tpu_config.InputPipelineConfig.PER_HOST_V2
      )

      est = tpu_estimator.TPUEstimator(
          model_fn=get_model_fn(),
          config=run_config,
          train_batch_size=32,
          eval_batch_size=32,
          predict_batch_size=32,
          use_tpu=True)

      # This exception is ok as we do not have sufficient TPU cores to run the
      # model.
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  'there are only 2 cores in the TPU topology'):
        est.evaluate(_input_fn, steps=1)

  @parameterized.parameters(
      (tpu_config.InputPipelineConfig.BROADCAST, 'evaluate'),
      (tpu_config.InputPipelineConfig.PER_HOST_V1, 'evaluate'),
      (tpu_config.InputPipelineConfig.PER_HOST_V2, 'evaluate'),
      (tpu_config.InputPipelineConfig.BROADCAST, 'predict'),
      (tpu_config.InputPipelineConfig.PER_HOST_V1, 'predict'),
      (tpu_config.InputPipelineConfig.PER_HOST_V2, 'predict'))
  def test_no_error_1host_1replica_in_eval_and_predict_mode(
      self, input_pipeline_mode, predict_or_evaluate):

    def _input_fn(params):
      return dummy_input_fn_with_dataset(
          dataset_size=params['batch_size'], repeat=False)

    FLAGS.test_num_shards = None
    run_config = create_run_config(
        iterations_per_loop=1, num_cores_per_replica=2,
        per_host_input_for_training=input_pipeline_mode)

    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(),
        config=run_config,
        train_batch_size=32,
        eval_batch_size=32,
        predict_batch_size=32,
        use_tpu=True)
    if predict_or_evaluate == 'evaluate':
      est.evaluate(_input_fn, steps=1)
    else:
      list(est.predict(_input_fn))


class TPUConfigTest(test.TestCase):

  def _create_ctx(self, run_config, mode=_TRAIN):
    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(), config=run_config, train_batch_size=16)

    with est._ctx.with_mode(mode) as ctx:
      return ctx

  def test_no_cluster_spec(self):
    run_config = tpu_config.RunConfig()
    ctx = self._create_ctx(run_config)
    self.assertIsNone(ctx.master_job)
    ctx = self._create_ctx(run_config, mode=_EVAL)
    self.assertIsNone(ctx.master_job)

    run_config = tpu_config.RunConfig(master='grpc://10.4.5.7:8470')
    ctx = self._create_ctx(run_config)
    self.assertEqual('tpu_worker', ctx.master_job)
    ctx = self._create_ctx(run_config, mode=_EVAL)
    self.assertEqual('tpu_worker', ctx.master_job)

    run_config = tpu_config.RunConfig(
        master='grpc://10.4.5.7:8470', evaluation_master='grpc://10.5.6.7:8470')
    ctx = self._create_ctx(run_config)
    self.assertEqual('tpu_worker', ctx.master_job)
    ctx = self._create_ctx(run_config, mode=_EVAL)
    self.assertEqual('tpu_worker', ctx.master_job)

  def test_cluster_spec_prop(self):
    cluster_def = cluster_pb2.ClusterDef()
    worker_job = cluster_def.job.add()
    worker_job.name = 'worker'
    worker_job.tasks[0] = 'grpc://10.2.3.4:8470'
    session_config = config_pb2.ConfigProto(cluster_def=cluster_def)
    run_config = tpu_config.RunConfig(
        session_config=session_config, master='grpc://10.2.3.4:8470')

    ctx = self._create_ctx(run_config)
    self.assertEqual('worker', ctx.master_job)

  def test_cluster_spec_prop_multi_jobs(self):
    cluster_def = cluster_pb2.ClusterDef()
    worker_job = cluster_def.job.add()
    worker_job.name = 'worker'
    worker_job.tasks[0] = 'grpc://10.2.3.4:8470'
    coordinator_job = cluster_def.job.add()
    coordinator_job.name = 'coordinator'
    coordinator_job.tasks[0] = 'grpc://10.2.3.4:8470'
    session_config = config_pb2.ConfigProto(cluster_def=cluster_def)
    run_config = tpu_config.RunConfig(
        session_config=session_config, master='grpc://10.2.3.4:8470')

    ctx = self._create_ctx(run_config)
    self.assertEqual('worker', ctx.master_job)

  def test_cluster_spec_prop_cannot_infer(self):
    # No coordinator.
    cluster_def = cluster_pb2.ClusterDef()
    worker_job = cluster_def.job.add()
    worker_job.name = 'worker'
    worker_job.tasks[0] = 'grpc://10.2.3.4:8470'
    coordinator_job = cluster_def.job.add()
    coordinator_job.name = 'other_worker'
    coordinator_job.tasks[0] = 'grpc://10.2.3.4:8470'
    session_config = config_pb2.ConfigProto(cluster_def=cluster_def)
    run_config = tpu_config.RunConfig(
        session_config=session_config, master='grpc://10.2.3.4:8470')
    with self.assertRaises(ValueError):
      ctx = self._create_ctx(run_config)
      ctx.master_job  # pylint:disable=pointless-statement

    # 2 non-coordinator jobs.
    cluster_def = cluster_pb2.ClusterDef()
    worker_job = cluster_def.job.add()
    worker_job.name = 'worker'
    worker_job.tasks[0] = 'grpc://10.2.3.4:8470'
    other_worker_job = cluster_def.job.add()
    other_worker_job.name = 'other_worker'
    other_worker_job.tasks[0] = 'grpc://10.2.3.5:8470'
    coordinator_job = cluster_def.job.add()
    coordinator_job.name = 'coordinator'
    coordinator_job.tasks[0] = 'grpc://10.2.3.4:8470'
    session_config = config_pb2.ConfigProto(cluster_def=cluster_def)
    run_config = tpu_config.RunConfig(
        session_config=session_config, master='grpc://10.2.3.4:8470')
    with self.assertRaises(ValueError):
      ctx = self._create_ctx(run_config)
      ctx.master_job  # pylint:disable=pointless-statement

  def test_session_config_none(self):
    run_config = tpu_config.RunConfig()
    self.assertIsNone(run_config.session_config)
    ctx = self._create_ctx(run_config)
    self.assertIsNone(ctx.master_job)

    run_config = tpu_config.RunConfig(master='grpc://10.2.3.4:8470')
    self.assertIsNone(run_config.session_config)
    ctx = self._create_ctx(run_config)
    self.assertEqual('tpu_worker', ctx.master_job)

  def test_override_name(self):
    tpu_cfg = tpu_config.TPUConfig(tpu_job_name='my_custom_job')
    run_config = tpu_config.RunConfig(tpu_config=tpu_cfg)
    ctx = self._create_ctx(run_config)
    self.assertEqual('my_custom_job', ctx.master_job)

  def test_evaluation_master(self):
    run_config = tpu_config.RunConfig(master='grpc://10.2.3.4:8470')
    self.assertEqual(run_config.master, run_config.evaluation_master)

    run_config = tpu_config.RunConfig(
        master='grpc://10.2.3.4:8470', evaluation_master='grpc://1.1.1.1:8470')
    self.assertEqual('grpc://1.1.1.1:8470', run_config.evaluation_master)

  def test_input_partition_config(self):
    with self.assertRaisesRegex(ValueError,
                                'input_partition_dims is.* PER_HOST_V2 mode.'):
      tpu_config.TPUConfig(
          num_shards=1, input_partition_dims=[[1, 2, 1, 1], None])

    with self.assertRaisesRegex(ValueError,
                                '.*requires setting num_cores_per_replica.'):
      tpu_config.TPUConfig(
          num_shards=1,
          per_host_input_for_training=tpu_config.InputPipelineConfig
          .PER_HOST_V2,
          input_partition_dims=[[1, 2, 1, 1], None])

    with self.assertRaisesRegex(ValueError, '.*with one or two elements.'):
      tpu_config.TPUConfig(
          num_shards=1,
          per_host_input_for_training=tpu_config.InputPipelineConfig
          .PER_HOST_V2,
          input_partition_dims=[[1, 2, 1, 1], None, None])

    tpu_config.TPUConfig(
        num_shards=1,
        num_cores_per_replica=2,
        per_host_input_for_training=tpu_config.InputPipelineConfig.PER_HOST_V2,
        input_partition_dims=[[1, 2, 1, 1], None])


class TPUEstimatorInputPartitionValidationTest(test.TestCase):

  def _train(self,
             iterations_per_loop,
             image_height=224,
             image_width=224,
             steps=None,
             num_shards=None,
             num_cores_per_replica=None,
             input_partition_dims=None):
    """Trains the model with InputPartition config."""

    def input_fn(params):
      batch_size = params['batch_size']
      x = np.random.normal(
          size=[batch_size, image_height, image_width, 3]).astype(np.float32)
      return dummy_input_fn_with_dataset(batch_size, repeat=True, x=x)

    run_config = create_run_config(
        iterations_per_loop=iterations_per_loop,
        num_shards=num_shards,
        num_cores_per_replica=num_cores_per_replica,
        input_partition_dims=input_partition_dims,
        per_host_input_for_training=tpu_config.InputPipelineConfig.PER_HOST_V2)
    est = tpu_estimator.TPUEstimator(
        model_fn=get_model_fn(),
        config=run_config,
        train_batch_size=128 * num_shards,
        eval_batch_size=128 * num_shards)

    est.train(input_fn, steps=steps, max_steps=None)

  def test_train_with_non_positive_dims(self):
    with self.assertRaisesRegex(ValueError,
                                'All input partition dims must be >= 1.'):
      self._train(
          iterations_per_loop=2,
          image_height=321,
          image_width=224,
          steps=2,
          num_shards=1,
          num_cores_per_replica=2,
          input_partition_dims=[{
              'x': [1, 2, 0, 1]
          }, None])

  def test_train_with_unmatched_partition_dims(self):
    with self.assertRaisesRegex(
        ValueError, 'The product of each input partition dim should '
        'equal to num_cores_per_replica.*'):
      self._train(
          iterations_per_loop=2,
          image_height=320,
          image_width=224,
          steps=2,
          num_shards=1,
          num_cores_per_replica=2,
          input_partition_dims=[{
              'x': [1, 2, 2, 1]
          }, None])

  def test_train_with_shape_unmatched_partition_dims(self):
    with self.assertRaisesRegex(ValueError,
                                'Input partition dims must have the same .*'):
      self._train(
          iterations_per_loop=2,
          image_height=320,
          image_width=224,
          steps=2,
          num_shards=1,
          num_cores_per_replica=2,
          input_partition_dims=[{
              'x': [1, 2, 1]
          }, None])

  def test_train_with_unmatched_feature_keys(self):
    with self.assertRaisesRegex(
        ValueError, r'TPUConfig.input_partition_dims\[0\]'
        ' mismatched feature .*'):
      self._train(
          iterations_per_loop=2,
          image_height=320,
          image_width=224,
          steps=2,
          num_shards=1,
          num_cores_per_replica=2,
          input_partition_dims=[{
              'wrong_key': [1, 2, 1]
          }, None])

  def test_train_with_unmatched_label_keys(self):
    with self.assertRaisesRegex(
        ValueError, r'TPUConfig.input_partition_dims\[1\]'
        ' mismatched label .*'):
      self._train(
          iterations_per_loop=2,
          image_height=320,
          image_width=224,
          steps=2,
          num_shards=1,
          num_cores_per_replica=2,
          input_partition_dims=[{
              'x': [1, 2, 1, 1]
          }, {
              'wrong_key': None
          }])

  def test_train_uneven_partitions_successful(self):
    # image_height=321, partitioned to 2 tensors with heights 161 and 160.
    self._train(
        iterations_per_loop=2,
        image_height=321,
        image_width=224,
        steps=2,
        num_shards=1,
        num_cores_per_replica=2,
        input_partition_dims=[{
            'x': [1, 2, 1, 1]
        }, None])

  def test_uneven_partitions_computation(self):
    image_height, image_width = 321, 224

    def _predict_input_fn(params):
      batch_size = params['batch_size']
      x = np.random.normal(
          size=[batch_size, image_height, image_width, 3]).astype(np.float32)
      return dummy_input_fn_with_dataset(batch_size, repeat=False, x=x)

    def _model_fn(features, labels, mode, params):
      del params, labels
      if mode == _PREDICT:
        conv_output = layers.conv2d(features['x'], filters=1, kernel_size=3)
        return tpu_estimator.TPUEstimatorSpec(
            mode=mode, predictions={'predictions': conv_output})

    run_config = create_run_config(
        iterations_per_loop=2,
        num_shards=1,
        num_cores_per_replica=2,
        input_partition_dims=[{
            'x': [1, 2, 1, 1]
        }],
        per_host_input_for_training=tpu_config.InputPipelineConfig.PER_HOST_V2)
    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        config=run_config,
        train_batch_size=128,
        predict_batch_size=1)
    res = list(est.predict(_predict_input_fn))
    self.assertEqual(len(res), 1)
    self.assertEqual(res[0]['predictions'].shape, (319, 222, 1))

  def _test_input_partitions_with_nested_label(self, input_partition_dims):
    image_height, image_width = 224, 224

    def _dummy_input_fn_with_dataset(dataset_size,
                                     repeat=True,
                                     x=None,
                                     batch_size=None):
      if batch_size is None:
        batch_size = dataset_size
      if x is None:
        x = np.random.normal(size=[dataset_size, 1]).astype(np.float32)
      labels = [[2.0]] * dataset_size

      dataset1 = dataset_lib.Dataset.from_tensor_slices(x)
      dataset2 = dataset_lib.Dataset.from_tensor_slices(labels)
      dataset = dataset_lib.Dataset.zip((dataset1, dataset2))
      if repeat:
        dataset = dataset.repeat()
      dataset = dataset.batch(batch_size, drop_remainder=True)

      def _map(x, y):
        return {'x': x}, {'label_1': {'label_2': y, 'label_3': y}, 'label_4': y}

      return dataset.map(_map)

    def _input_fn(params):
      batch_size = params['batch_size']
      x = np.random.normal(
          size=[batch_size, image_height, image_width, 3]).astype(np.float32)
      return _dummy_input_fn_with_dataset(batch_size, repeat=True, x=x)

    def _model_fn(features, labels, mode, params):
      del params
      predictions = dense_computation(features)
      loss = losses.mean_squared_error(labels['label_1']['label_3'],
                                       predictions)
      optimizer = tf.compat.v1.tpu.CrossShardOptimizer(
          training.GradientDescentOptimizer(learning_rate=0.5))
      train_op = optimizer.minimize(loss, training.get_global_step())

      return tpu_estimator.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)

    run_config = create_run_config(
        iterations_per_loop=2,
        num_shards=1,
        num_cores_per_replica=2,
        input_partition_dims=input_partition_dims,
        per_host_input_for_training=tpu_config.InputPipelineConfig.PER_HOST_V2)
    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        config=run_config,
        train_batch_size=128,
        predict_batch_size=1)
    est.train(_input_fn, steps=4, max_steps=None)

  def test_fully_specified_input_partitions_with_with_nested_label(self):
    self._test_input_partitions_with_nested_label([{'x': [1, 2, 1, 1]}, None])

  def test_partial_specified_input_partitions_with_nested_label(self):
    self._test_input_partitions_with_nested_label([{
        'x': [1, 2, 1, 1]
    }, {
        'label_1': {
            'label_2': None,
            'label_3': None
        },
        'label_4': None
    }])

  def test_incorrect_input_partitions_with_nested_label(self):
    with self.assertRaisesRegex(
        ValueError, r'TPUConfig.input_partition_dims\[1\]'
        ' mismatched the structure of labels. .*'):
      self._test_input_partitions_with_nested_label([{
          'x': [1, 2, 1, 1]
      }, {
          'label_1': None,
          'label_4': None
      }])


class TPUEstimatorInputPipelinePlacementTest(test.TestCase):

  def _test_placement(self, per_host):

    num_cores = 32
    batch_sizes = []
    global_batch_size = 1024
    host_id_matcher = re.compile(r'^input_pipeline_task(\d+)/(.*)$')
    host_to_device = collections.defaultdict(list)

    def _input_fn(params):
      batch_sizes.append(params['batch_size'])
      return dummy_input_fn(params['batch_size'])

    def _model_fn(features, labels, mode, params):
      # Examine the input pipeline placement.
      operations = ops.get_default_graph().get_operations()
      for op in operations:
        result = host_id_matcher.match(op.name)
        if result is None:
          continue
        # There is one op to read iterations_per_loop var (the Send node of
        # tf.identity). It is colocated with global step. So, ignore here.
        if result.group(2) == 'Identity/ReadVariableOp':
          continue
        host_id = int(result.group(1))
        host_to_device[host_id].append(op.device)
      return get_model_fn()(features, labels, mode, params)

    run_config = tpu_config.RunConfig(
        master='fake://123',
        tpu_config=tpu_config.TPUConfig(
            num_shards=num_cores, per_host_input_for_training=per_host))
    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        train_batch_size=global_batch_size,
        config=run_config)

    old_value = tpu_estimator._WRAP_INPUT_FN_INTO_WHILE_LOOP
    tpu_estimator._WRAP_INPUT_FN_INTO_WHILE_LOOP = True

    try:
      est.train(input_fn=_input_fn, steps=1)
      self.fail('The train should not finish.')
    except errors.NotFoundError:
      # Expected. The TF sesion master is not valid.
      pass

    tpu_estimator._WRAP_INPUT_FN_INTO_WHILE_LOOP = old_value

    expected_num_hosts = num_cores // 8
    if per_host:
      self.assertEqual(len(batch_sizes), expected_num_hosts)
      self.assertEqual(batch_sizes[0], global_batch_size // expected_num_hosts)
    else:
      self.assertEqual(len(batch_sizes), num_cores)
      self.assertEqual(batch_sizes[0], global_batch_size // num_cores)
    self.assertEqual(expected_num_hosts, len(list(host_to_device.keys())))
    for host_id in range(expected_num_hosts):
      # On each host, all ops should be placed on the same device
      device_set = set(host_to_device[host_id])
      self.assertEqual(1, len(device_set))
      self.assertEqual('/job:tpu_worker/task:{}/device:CPU:0'.format(host_id),
                       host_to_device[host_id][0])

  def _query_system(self, master_address, cluster_def, query_topology):
    del master_address, cluster_def, query_topology
    return tpu_system_metadata_lib.TPUSystemMetadata(
        num_cores=32,
        num_hosts=4,
        num_of_cores_per_host=8,
        topology=None,
        devices=[])

  def test_per_host_placement(self):
    with test.mock.patch.object(
        tpu_system_metadata_lib,
        '_query_tpu_system_metadata',
        side_effect=self._query_system):
      self._test_placement(True)

  def test_per_core_placement(self):
    with test.mock.patch.object(
        tpu_system_metadata_lib,
        '_query_tpu_system_metadata',
        side_effect=self._query_system):
      self._test_placement(False)


class TPUEstimatorScaffoldTest(test.TestCase):

  def _get_scaffold_fn(self, mode):

    def _scaffold_fn_on_cpu():
      scaffold = training.Scaffold()
      finalize_fn = scaffold.finalize

      def _finalize():
        self.assertNotIn(mode, self.is_finalize_fn_called)
        self.is_finalize_fn_called[mode] = True
        return finalize_fn()

      scaffold.finalize = _finalize
      return scaffold

    return _scaffold_fn_on_cpu

  def _input_fn(self, params):
    return dummy_input_fn(params['batch_size'])

  def _predict_input_fn(self, params):
    return dummy_input_fn_with_dataset(
        dataset_size=params['batch_size'], repeat=False)

  def _model_fn(self, features, labels, mode, config, params):
    """Creates a head returning `TPUEstimatorSpec` based on mode."""
    predictions = layers.dense(
        features['x'], 1, kernel_initializer=init_ops.zeros_initializer())

    eval_metrics = None
    train_op = None
    loss = None

    if mode != _PREDICT:
      loss = losses.mean_squared_error(labels, predictions)
      if mode == _TRAIN:
        optimizer = training.GradientDescentOptimizer(learning_rate=0.5)
        if params['use_tpu']:
          optimizer = tf.compat.v1.tpu.CrossShardOptimizer(optimizer)
        train_op = optimizer.minimize(
            loss, global_step=training.get_global_step())
      elif mode == _EVAL:

        def _metric_fn_on_cpu(labels, predictions):
          return {
              'mse': metrics_lib.mean_absolute_error(labels, predictions),
          }

        eval_metrics = (_metric_fn_on_cpu, [labels, predictions])

    return tpu_estimator.TPUEstimatorSpec(
        mode=mode,
        train_op=train_op,
        loss=loss,
        predictions={'x': predictions},
        scaffold_fn=self._get_scaffold_fn(mode),
        eval_metrics=eval_metrics)

  def test_train(self):
    for use_tpu in [True, False]:
      self.is_finalize_fn_called = {}
      est = tpu_estimator.TPUEstimator(
          model_fn=self._model_fn,
          train_batch_size=8,
          config=create_run_config(iterations_per_loop=4),
          use_tpu=use_tpu)
      est.train(input_fn=self._input_fn, steps=1)
      self.assertTrue(self.is_finalize_fn_called[_TRAIN])

  def test_eval(self):
    for use_tpu in [True, False]:
      self.is_finalize_fn_called = {}
      est = tpu_estimator.TPUEstimator(
          model_fn=self._model_fn,
          train_batch_size=8,
          eval_batch_size=8,
          config=create_run_config(iterations_per_loop=4),
          use_tpu=use_tpu)

      # Generate checkpoint.
      est.train(input_fn=self._input_fn, steps=1)

      est.evaluate(input_fn=self._input_fn, steps=1)
      self.assertTrue(self.is_finalize_fn_called[_EVAL])

  def test_predict(self):
    for use_tpu in [True, False]:
      self.is_finalize_fn_called = {}
      est = tpu_estimator.TPUEstimator(
          model_fn=self._model_fn,
          train_batch_size=8,
          predict_batch_size=8,
          config=create_run_config(iterations_per_loop=4),
          use_tpu=use_tpu)

      # Generate checkpoint.
      est.train(input_fn=self._input_fn, steps=1)
      list(est.predict(input_fn=self._predict_input_fn))
      self.assertTrue(self.is_finalize_fn_called[_PREDICT])

  def test_scaffold_fn_capture_tpu_tensor(self):

    def _model_fn(features, labels, mode, config, params):
      """Creates a head returning `TPUEstimatorSpec` based on mode."""
      del config, params
      predictions = layers.dense(
          features['x'], 1, kernel_initializer=init_ops.zeros_initializer())
      loss = losses.mean_squared_error(labels, predictions)
      optimizer = training.GradientDescentOptimizer(learning_rate=0.5)
      optimizer = tf.compat.v1.tpu.CrossShardOptimizer(optimizer)
      train_op = optimizer.minimize(
          loss, global_step=training.get_global_step())

      def scaffold_fn():
        summary_lib.scalar('loss_', loss)

        return training.Scaffold()

      return tpu_estimator.TPUEstimatorSpec(
          mode=mode, train_op=train_op, loss=loss, scaffold_fn=scaffold_fn)

    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        train_batch_size=8,
        config=create_run_config(iterations_per_loop=4))

    with self.assertRaises(ValueError):
      est.train(input_fn=self._input_fn, steps=1)

  def test_scaffold_capture_tpu_tensor(self):

    def _model_fn(features, labels, mode, config, params):
      """Creates a head returning `TPUEstimatorSpec` based on mode."""
      del config, params
      predictions = layers.dense(
          features['x'], 1, kernel_initializer=init_ops.zeros_initializer())
      loss = losses.mean_squared_error(labels, predictions)
      optimizer = training.GradientDescentOptimizer(learning_rate=0.5)
      optimizer = tf.compat.v1.tpu.CrossShardOptimizer(optimizer)
      train_op = optimizer.minimize(
          loss, global_step=training.get_global_step())

      # Scaffold.finalize will "merge" all summaries, so we will be able
      # to detect invalid TPU tensor capture.
      summary_lib.scalar('loss_', loss)

      def scaffold_fn():
        return training.Scaffold()

      return tpu_estimator.TPUEstimatorSpec(
          mode=mode, train_op=train_op, loss=loss, scaffold_fn=scaffold_fn)

    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        train_batch_size=8,
        config=create_run_config(iterations_per_loop=4))

    with self.assertRaises(ValueError):
      est.train(input_fn=self._input_fn, steps=1)


class TPUEstimatorScaffoldWithEMATest(test.TestCase):

  def _get_scaffold(self, ema):
    var_dict = ema.variables_to_restore()
    return training.Scaffold(saver=training.Saver(var_dict))

  def _input_fn(self, params):
    return dummy_input_fn(params['batch_size'])

  def _model_fn(self, features, labels, mode, config, params):
    """Creates a head returning `TPUEstimatorSpec` based on mode."""
    with variable_scope.variable_scope('foo'):
      predictions = layers.dense(
          features['x'], 1, kernel_initializer=init_ops.zeros_initializer())

    eval_metrics = None
    train_op = None

    loss = losses.mean_squared_error(labels, predictions)
    ema = moving_averages.ExponentialMovingAverage(decay=0.999)

    if mode == _TRAIN:
      optimizer = training.GradientDescentOptimizer(learning_rate=0.5)
      optimizer = tf.compat.v1.tpu.CrossShardOptimizer(optimizer)

      opt_op = optimizer.minimize(loss, global_step=training.get_global_step())

      with ops.control_dependencies([opt_op]):
        train_op = ema.apply()

    elif mode == _EVAL:

      def _metric_fn_on_cpu(labels, predictions):
        return {
            'mse': metrics_lib.mean_absolute_error(labels, predictions),
        }

      eval_metrics = (_metric_fn_on_cpu, [labels, predictions])

    # Change the saver for non-training mode.
    scaffold_fn = None if mode == _TRAIN else (lambda: self._get_scaffold(ema))

    return tpu_estimator.TPUEstimatorSpec(
        mode=mode,
        train_op=train_op,
        loss=loss,
        predictions=predictions,
        scaffold_fn=scaffold_fn,
        eval_metrics=eval_metrics)

  def test_ema_with_train_and_evaluate(self):
    use_tpu = True
    est = tpu_estimator.TPUEstimator(
        model_fn=self._model_fn,
        train_batch_size=8,
        eval_batch_size=8,
        config=create_run_config(iterations_per_loop=1),
        use_tpu=use_tpu)

    # With iterations_per_loop=1 and train steps = 2, the after_run in the hook
    # will be invoked once to change the bias value. Make the bias variable
    # super large here to avoid flaky.
    rewrite_var_hook = _RewriteVarHook(
        scope_name='foo', variable_name='dense/bias', value=[100])
    est.train(input_fn=self._input_fn, steps=2, hooks=[rewrite_var_hook])

    bias_value = est.get_variable_value('foo/dense/bias')
    bias_ma_value = est.get_variable_value(
        'foo/dense/bias/ExponentialMovingAverage')

    self.assertNotAllClose(bias_value, bias_ma_value)

    model_variable_value_hook = (
        _ModelVariableValueHook(scope_name='foo', variable_name='dense/bias'))

    est.evaluate(
        input_fn=self._input_fn, steps=1, hooks=[model_variable_value_hook])

    bias_value_during_eval = model_variable_value_hook.got_value
    self.assertAlmostEqual(bias_ma_value, bias_value_during_eval)


class _ModelVariableValueHook(session_run_hook.SessionRunHook):
  """Capture the value of given variable after initialization."""

  def __init__(self, scope_name, variable_name):
    """Constructs the run hook."""
    self.scope_name = scope_name
    self.variable_name = variable_name
    self.got_value = None

  def after_create_session(self, sess, coord):
    del coord

    with variable_scope.variable_scope(self.scope_name, reuse=True):
      self.got_value = sess.run(variable_scope.get_variable(self.variable_name))


class _RewriteVarHook(session_run_hook.SessionRunHook):
  """Rwrite the variable value hook."""

  def __init__(self, scope_name, variable_name, value):
    """Constructs the run hook."""
    self.scope_name = scope_name
    self.variable_name = variable_name
    self.value = value

  def begin(self):
    with variable_scope.variable_scope(self.scope_name, reuse=True):
      self._var = variable_scope.get_variable(self.variable_name)

  def after_run(self, run_context, run_values):
    self._var.load(self.value, session=run_context.session)


class TPUEstimatorHostCallTest(test.TestCase):

  def _input_fn(self, params):
    return dummy_input_fn(params['batch_size'])

  def _host_call(self, model_dir, mode):

    def fn(global_step, labels, predictions):
      global_step = math_ops.cast(global_step[0], dtypes.int64)
      # We add a filename suffix here to avoid clashing with existing summary
      # creation in Estimator. Otherwise both may attempt to open the same
      # filename.
      #
      # The name of the op is set to model_dir to avoid ResourceManager caching
      # the same summary writer instance across tests.
      #
      # In addition, we give different suffixes for train and eval to avoid
      # FileWriter in evaluate() overwrites the events dumped by training.
      # This is because the event file path has timestamps at second accuracy
      # but the CPU training could be super fast.
      with tf.summary.create_file_writer(
          model_dir,
          filename_suffix='.TPUEstimator-{}'.format(1 if mode == model_fn_lib
                                                    .ModeKeys.TRAIN else 2),
          name=os.path.basename(model_dir)).as_default():
        with summary_ops_v2.record_summaries_every_n_global_steps(
            5 if mode == model_fn_lib.ModeKeys.TRAIN else 1,
            global_step=global_step):
          loss = losses.mean_squared_error(labels, predictions)
          summary_ops_v2.scalar('host_call_test', loss, step=global_step)
          summary_ops_v2.scalar(
              'host_call_global_step', global_step, step=global_step)
          return tf.compat.v1.summary.all_v2_summary_ops()

    return fn

  def _metric_fn_on_cpu(self, labels, predictions):
    return {
        'mse': metrics_lib.mean_absolute_error(labels, predictions),
    }

  def _model_fn(self, model_dir):

    def fn(features, labels, mode, params):
      del params
      train_op = None
      predictions = dense_computation(features)
      loss = losses.mean_squared_error(labels, predictions)
      if mode == _TRAIN:
        optimizer = tf.compat.v1.tpu.CrossShardOptimizer(
            training.GradientDescentOptimizer(learning_rate=0.5))
        train_op = optimizer.minimize(loss, training.get_global_step())
      return tpu_estimator.TPUEstimatorSpec(
          mode,
          loss=loss,
          train_op=train_op,
          predictions=predictions,
          eval_metrics=(self._metric_fn_on_cpu, [labels, predictions]),
          host_call=(self._host_call(model_dir, mode), [
              array_ops.reshape(
                  math_ops.cast(training.get_global_step(), dtypes.int32), [1]),
              labels, predictions
          ]))

    return fn

  def _events_from_logdir(self, logdir):
    files = gfile.ListDirectory(logdir)
    events = []
    found = False
    for f in sorted(files):
      # Note that we need to distinguish between the TPUEstimator events file
      # and the SummarySaverHook one.
      if '.tfevents.' in f and '.TPUEstimator' in f:
        found = True
        f = os.path.join(logdir, f)
        events.extend(events_from_file(f))
    self.assertEqual(True, found)
    return events

  def _test_summaries(self, use_tpu, output_every_n_steps=False):
    if not use_tpu and FLAGS.tpu_use_tfrt:
      self.skipTest('This test calls TPU ops without initializing TPU system. '
                    'See b/194549789')
    outfeed_every_n_steps = 2 if output_every_n_steps else 1
    model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    run_config = tpu_config.RunConfig(
        master='',
        model_dir=model_dir,
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=21,
            num_shards=FLAGS.test_num_shards,
            experimental_host_call_every_n_steps=outfeed_every_n_steps,
        ))
    est = tpu_estimator.TPUEstimator(
        model_fn=self._model_fn(model_dir),
        train_batch_size=8,
        eval_batch_size=8,
        config=run_config,
        use_tpu=use_tpu)

    est.train(input_fn=self._input_fn, steps=42)
    events = self._events_from_logdir(model_dir)
    events = [e for e in events if e.WhichOneof('what') != 'file_version']
    if not output_every_n_steps or not use_tpu:
      self.assertEqual(18, len(events))
      self.assertEqual(
          9,
          len([e for e in events
               if e.summary.value[0].tag == 'host_call_test']))
      self.assertEqual([value*5 for value in range(9)], [
          e.summary.value[0].simple_value
          for e in events
          if e.summary.value[0].tag == 'host_call_global_step'])
    else:
      self.assertEqual(10, len(events))
      self.assertEqual(
          5,
          len([e for e in events
               if e.summary.value[0].tag == 'host_call_test']))
      self.assertEqual([0, 10, 20, 25, 35], [
          e.summary.value[0].simple_value
          for e in events
          if e.summary.value[0].tag == 'host_call_global_step'])

    est.evaluate(input_fn=self._input_fn, steps=7)
    events = self._events_from_logdir(model_dir)
    events = [e for e in events if e.WhichOneof('what') != 'file_version']
    if not output_every_n_steps or not use_tpu:
      self.assertEqual(32, len(events))  # 18 from train + 14 from eval
      self.assertEqual(
          16,  # 9 from train + 7 from eval
          len([e for e in events
               if e.summary.value[0].tag == 'host_call_test']))
      self.assertEqual(
          [value*5 for value in range(9)] + [42] * 7,
          [e.summary.value[0].simple_value
           for e in events
           if e.summary.value[0].tag == 'host_call_global_step'])
    else:
      self.assertEqual(24, len(events))
      self.assertEqual(
          12,
          len([e for e in events
               if e.summary.value[0].tag == 'host_call_test']))
      self.assertEqual(
          [0, 10, 20, 25, 35] + [42] * 7,
          [e.summary.value[0].simple_value
           for e in events
           if e.summary.value[0].tag == 'host_call_global_step'])

  def test_summaries(self):
    self._test_summaries(True)

  def test_summaries_on_cpu(self):
    self._test_summaries(False)

  def test_summaries_every_n_steps(self):
    self._test_summaries(True, True)

  def test_summaries_on_cpu_every_n_steps(self):
    self._test_summaries(False, True)

  def test_keras_tensorflow_op_layer(self):
    def model_fn(features, labels, mode, params):
      del features, labels, params
      i1 = tf.keras.Input(10)
      i2 = tf.keras.Input(10)
      out = tf.concat([i1, i2], axis=1)
      out = tf.keras.layers.Dense(1)(out)
      model = tf.keras.Model([i1, i2], out)
      x = [tf.ones((5, 10)), tf.ones((5, 10))]
      y = model(x)
      loss = tf.reduce_mean(y)
      if mode == _TRAIN:
        optimizer = tf.compat.v1.tpu.CrossShardOptimizer(
            training.GradientDescentOptimizer(learning_rate=0.5))
        train_op = optimizer.minimize(loss, training.get_global_step())
      return tpu_estimator.TPUEstimatorSpec(
          mode,
          loss=loss,
          train_op=train_op)

    model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    run_config = tpu_config.RunConfig(
        master='',
        model_dir=model_dir,
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=1,
            num_shards=FLAGS.test_num_shards,
        ))
    est = tpu_estimator.TPUEstimator(
        model_fn=model_fn,
        train_batch_size=8,
        eval_batch_size=8,
        config=run_config)
    est.train(input_fn=self._input_fn, steps=42)


if __name__ == '__main__':
  tf.compat.v1.disable_v2_behavior()
  test.main()
