# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""TPU RunConfig tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow_estimator.python.estimator import run_config as run_config_lib
from tensorflow_estimator.python.estimator.tpu import tpu_config as tpu_config_lib
from tensorflow_estimator.python.estimator.tpu import util as util_lib


def _set_tf_config_env_variable(tf_config):
  return tf.compat.v1.test.mock.patch.dict('os.environ',
                                           {'TF_CONFIG': json.dumps(tf_config)})


class TPURunConfigTest(tf.test.TestCase):

  def test_no_session_config_set_in_local_case(self):
    run_config = tpu_config_lib.RunConfig()
    self.assertIsNone(run_config.session_config)

  def test_no_session_config_overwrite_in_local_case(self):
    session_config = config_pb2.ConfigProto(allow_soft_placement=True)
    run_config = tpu_config_lib.RunConfig(session_config=session_config)
    self.assertEqual(session_config, run_config.session_config)

  def test_no_session_config_set_with_cluster_spec(self):
    tf_config = {
        'cluster': {
            run_config_lib.TaskType.CHIEF: ['host3:3'],
            run_config_lib.TaskType.WORKER: ['host3:4']
        },
        'task': {
            'type': run_config_lib.TaskType.CHIEF,
            'index': 0
        }
    }
    with _set_tf_config_env_variable(tf_config):
      run_config = tpu_config_lib.RunConfig()
      self.assertIsNone(run_config.session_config)

  def test_no_session_config_overwrite_with_cluster_spec(self):
    tf_config = {
        'cluster': {
            run_config_lib.TaskType.CHIEF: ['host3:3'],
            run_config_lib.TaskType.WORKER: ['host3:4']
        },
        'task': {
            'type': run_config_lib.TaskType.CHIEF,
            'index': 0
        }
    }
    with _set_tf_config_env_variable(tf_config):
      session_config = config_pb2.ConfigProto(allow_soft_placement=True)
      run_config = tpu_config_lib.RunConfig(session_config=session_config)
      self.assertEqual(session_config, run_config.session_config)

  def test_fail_with_invalid_num_shards(self):
    with self.assertRaisesRegexp(ValueError, 'must be positive'):
      tpu_config_lib.RunConfig(
          tpu_config=tpu_config_lib.TPUConfig(num_shards=0))

  def _validate_invalid_iterations_per_loop(self, iterations_per_loop):
    with self.assertRaisesRegexp(ValueError, 'must be positive'):
      tpu_config_lib.RunConfig(
          tpu_config=tpu_config_lib.TPUConfig(
              iterations_per_loop=iterations_per_loop))

  def test_fail_with_iterations_per_loop(self):
    self._validate_invalid_iterations_per_loop(0)
    self._validate_invalid_iterations_per_loop(-1)
    self._validate_invalid_iterations_per_loop('-1h')
    self._validate_invalid_iterations_per_loop('-1m')
    self._validate_invalid_iterations_per_loop('-1s')

  def test_fail_with_invalid_num_cores_per_replica(self):
    with self.assertRaisesRegexp(
        ValueError, 'num_cores_per_replica must be 1, 2, 4, 8, 16, 32, 64, 128;'
        ' got 7'):
      tpu_config_lib.TPUConfig(num_cores_per_replica=7)

  def _evaluate_iterations_per_loop_in_seconds(self, value, expected_value,
                                               expected_unit):
    config = tpu_config_lib.RunConfig(
        tpu_config=tpu_config_lib.TPUConfig(iterations_per_loop=value))
    self.assertEqual(config.tpu_config.iterations_per_loop, value)
    d = util_lib.parse_iterations_per_loop(
        config.tpu_config.iterations_per_loop)
    self.assertEqual(expected_value, d.value)
    self.assertEqual(expected_unit, d.unit)

  def test_valid_iterations_per_loop(self):
    self._evaluate_iterations_per_loop_in_seconds(1, 1, 'count')
    self._evaluate_iterations_per_loop_in_seconds(100, 100, 'count')
    self._evaluate_iterations_per_loop_in_seconds('300s', 300, 'seconds')
    self._evaluate_iterations_per_loop_in_seconds('1m', 60, 'seconds')
    self._evaluate_iterations_per_loop_in_seconds('1h', 3600, 'seconds')


class TPURunConfigMasterTest(tf.test.TestCase):

  def test_default_values(self):
    run_config = tpu_config_lib.RunConfig()
    self.assertEqual('', run_config.master)
    self.assertEqual('', run_config.evaluation_master)

  def test_user_provided_master_and_evaluation_master(self):
    run_config = tpu_config_lib.RunConfig(
        master='_master_123', evaluation_master='_eval_master_123')
    self.assertEqual('_master_123', run_config.master)
    self.assertEqual('_eval_master_123', run_config.evaluation_master)

  def test_evaluation_master_defaults_to_master(self):
    run_config = tpu_config_lib.RunConfig(master='_master_123')
    self.assertEqual('_master_123', run_config.master)
    self.assertEqual('_master_123', run_config.evaluation_master)

  def test_tf_config(self):
    tf_config = {
        'session_master': '_master_123',
        'eval_session_master': '_eval_master_123'
    }
    with _set_tf_config_env_variable(tf_config):
      run_config = tpu_config_lib.RunConfig()
      self.assertEqual('_master_123', run_config.master)
      self.assertEqual('_eval_master_123', run_config.evaluation_master)

  def test_evaluation_master_defaults_to_master_in_tf_config(self):
    tf_config = {
        'session_master': '_master_123',
    }
    with _set_tf_config_env_variable(tf_config):
      run_config = tpu_config_lib.RunConfig()
      self.assertEqual('_master_123', run_config.master)
      self.assertEqual('_master_123', run_config.evaluation_master)

  def test_respect_evaluation_master_in_tf_config(self):
    tf_config = {
        'cluster': {
            run_config_lib.TaskType.CHIEF: ['host0:0'],
        },
        'task': {
            'type': run_config_lib.TaskType.EVALUATOR,
            'index': 0
        },
    }
    with _set_tf_config_env_variable(tf_config):
      run_config = tpu_config_lib.RunConfig(master='_something')
      self.assertEqual('', run_config.evaluation_master)

  def test_user_overwrites_tf_config(self):
    tf_config = {
        'session_master': '_master_123',
        'eval_session_master': '_eval_master_123'
    }
    with _set_tf_config_env_variable(tf_config):
      run_config = tpu_config_lib.RunConfig(
          master='_new_master_123', evaluation_master='_new_eval_master_123')
      self.assertEqual('_new_master_123', run_config.master)
      self.assertEqual('_new_eval_master_123', run_config.evaluation_master)

  def test_user_overwrites_master_in_tf_config(self):
    tf_config = {
        'session_master': '_master_123',
        'eval_session_master': '_eval_master_123'
    }
    with _set_tf_config_env_variable(tf_config):
      run_config = tpu_config_lib.RunConfig(master='_new_master_123')
      self.assertEqual('_new_master_123', run_config.master)
      self.assertEqual('_eval_master_123', run_config.evaluation_master)


class TPUJobNameTest(tf.test.TestCase):

  def test_default_name(self):
    config = tpu_config_lib.RunConfig()
    self.assertIsNone(config.tpu_config.tpu_job_name)

  def test_with_tf_config(self):
    tf_config = {'service': {'tpu_worker_job_name': '_my_new_name',}}
    with _set_tf_config_env_variable(tf_config):
      config = tpu_config_lib.RunConfig()
      self.assertEqual('_my_new_name', config.tpu_config.tpu_job_name)


if __name__ == '__main__':
  tf.test.main()
