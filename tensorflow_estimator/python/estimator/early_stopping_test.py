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
"""Tests for early_stopping."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import test
from tensorflow.python.training import monitored_session
from tensorflow.python.training import training_util
from tensorflow_estimator.python.estimator import early_stopping
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import run_config


class _FakeRunConfig(run_config.RunConfig):

  def __init__(self, is_chief):
    super(_FakeRunConfig, self).__init__()
    self._is_chief = is_chief

  @property
  def is_chief(self):
    return self._is_chief


def _dummy_model_fn(features, labels, params):
  _, _, _ = features, labels, params


class _FakeEstimator(estimator.Estimator):
  """Fake estimator for testing."""

  def __init__(self, config):
    super(_FakeEstimator, self).__init__(
        model_fn=_dummy_model_fn, config=config)


def _write_events(eval_dir, params):
  """Test helper to write events to summary files."""
  for steps, loss, accuracy in params:
    estimator._write_dict_to_summary(eval_dir, {
        'loss': loss,
        'accuracy': accuracy,
    }, steps)


class EarlyStoppingHooksTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    config = _FakeRunConfig(is_chief=True)
    self._estimator = _FakeEstimator(config=config)
    eval_dir = self._estimator.eval_dir()
    os.makedirs(eval_dir)
    _write_events(
        eval_dir,
        [
            # steps, loss, accuracy
            (1000, 0.8, 0.5),
            (2000, 0.7, 0.6),
            (3000, 0.4, 0.7),
            (3500, 0.41, 0.68),
        ])

  def run_session(self, hooks, should_stop):
    hooks = hooks if isinstance(hooks, list) else [hooks]
    with ops.Graph().as_default():
      training_util.create_global_step()
      no_op = control_flow_ops.no_op()
      with monitored_session.SingularMonitoredSession(hooks=hooks) as mon_sess:
        mon_sess.run(no_op)
        self.assertEqual(mon_sess.should_stop(), should_stop)

  @parameterized.parameters(False, True)
  def test_make_early_stopping_hook(self, should_stop):
    self.run_session([
        early_stopping.make_early_stopping_hook(
            self._estimator, should_stop_fn=lambda: should_stop)
    ], should_stop)

  def test_make_early_stopping_hook_typeerror(self):
    with self.assertRaises(TypeError):
      early_stopping.make_early_stopping_hook(
          estimator=object(), should_stop_fn=lambda: True)

  def test_make_early_stopping_hook_valueerror(self):
    with self.assertRaises(ValueError):
      early_stopping.make_early_stopping_hook(
          self._estimator,
          should_stop_fn=lambda: True,
          run_every_secs=60,
          run_every_steps=100)


if __name__ == '__main__':
  test.main()
