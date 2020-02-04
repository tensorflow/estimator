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
"""Tests for optimizers.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras.optimizer_v2 import adagrad
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras.optimizer_v2 import ftrl
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow_estimator.python.estimator.canned import optimizers


class _TestOptimizerV2(optimizer_v2.OptimizerV2):

  def __init__(self):
    super(_TestOptimizerV2, self).__init__(name='TestOptimizer')

  def get_config(self):
    pass


class GetOptimizerInstanceV2(tf.test.TestCase):
  """Tests for Optimizer V2."""

  def test_unsupported_name(self):
    with self.assertRaisesRegexp(
        ValueError, 'Unsupported optimizer name: unsupported_name'):
      optimizers.get_optimizer_instance_v2(
          'unsupported_name', learning_rate=0.1)

  def test_adagrad_but_no_learning_rate(self):
    with self.cached_session():
      opt = optimizers.get_optimizer_instance_v2('Adagrad')
      # The creation of variables in optimizer_v2 is deferred to when it's
      # called, so we need to manually create it here. Same for all other tests.
      self.assertIsInstance(opt.learning_rate, tf.Variable)
      self.evaluate(tf.compat.v1.initializers.global_variables())
      self.assertIsInstance(opt, adagrad.Adagrad)
      self.assertAlmostEqual(0.001, self.evaluate(opt.learning_rate))

  def test_adam_but_no_learning_rate(self):
    with self.cached_session():
      opt = optimizers.get_optimizer_instance_v2('Adam')
      self.assertIsInstance(opt.learning_rate, tf.Variable)
      self.evaluate(tf.compat.v1.initializers.global_variables())
      self.assertIsInstance(opt, adam.Adam)
      self.assertAlmostEqual(0.001, self.evaluate(opt.learning_rate))

  def test_adagrad(self):
    with self.cached_session():
      opt = optimizers.get_optimizer_instance_v2('Adagrad', learning_rate=0.1)
      self.assertIsInstance(opt.learning_rate, tf.Variable)
      self.evaluate(tf.compat.v1.initializers.global_variables())
      self.assertIsInstance(opt, adagrad.Adagrad)
      self.assertAlmostEqual(0.1, self.evaluate(opt.learning_rate))

  def test_adam(self):
    with self.cached_session():
      opt = optimizers.get_optimizer_instance_v2('Adam', learning_rate=0.1)
      self.assertIsInstance(opt.learning_rate, tf.Variable)
      self.evaluate(tf.compat.v1.initializers.global_variables())
      self.assertIsInstance(opt, adam.Adam)
      self.assertAlmostEqual(0.1, self.evaluate(opt.learning_rate))

  def test_ftrl(self):
    with self.cached_session():
      opt = optimizers.get_optimizer_instance_v2('Ftrl', learning_rate=0.1)
      self.assertIsInstance(opt.learning_rate, tf.Variable)
      self.evaluate(tf.compat.v1.initializers.global_variables())
      self.assertIsInstance(opt, ftrl.Ftrl)
      self.assertAlmostEqual(0.1, self.evaluate(opt.learning_rate))

  def test_rmsprop(self):
    with self.cached_session():
      opt = optimizers.get_optimizer_instance_v2('RMSProp', learning_rate=0.1)
      self.assertIsInstance(opt.learning_rate, tf.Variable)
      self.evaluate(tf.compat.v1.initializers.global_variables())
      self.assertIsInstance(opt, rmsprop.RMSProp)
      self.assertAlmostEqual(0.1, self.evaluate(opt.learning_rate))

  def test_sgd(self):
    with self.cached_session():
      opt = optimizers.get_optimizer_instance_v2('SGD', learning_rate=0.1)
      self.assertIsInstance(opt.learning_rate, tf.Variable)
      self.evaluate(tf.compat.v1.initializers.global_variables())
      self.assertIsInstance(opt, gradient_descent.SGD)
      self.assertAlmostEqual(0.1, self.evaluate(opt.learning_rate))

  def test_object(self):
    opt = optimizers.get_optimizer_instance_v2(_TestOptimizerV2())
    self.assertIsInstance(opt, _TestOptimizerV2)

  def test_object_invalid(self):
    with self.assertRaisesRegexp(
        ValueError,
        'The given object is not a tf.keras.optimizers.Optimizer instance'):
      optimizers.get_optimizer_instance_v2((1, 2, 3))

  def test_callable(self):

    def _optimizer_fn():
      return _TestOptimizerV2()

    opt = optimizers.get_optimizer_instance_v2(_optimizer_fn)
    self.assertIsInstance(opt, _TestOptimizerV2)

  def test_lambda(self):
    opt = optimizers.get_optimizer_instance_v2(lambda: _TestOptimizerV2())  # pylint: disable=unnecessary-lambda
    self.assertIsInstance(opt, _TestOptimizerV2)

  def test_callable_returns_invalid(self):

    def _optimizer_fn():
      return (1, 2, 3)

    with self.assertRaisesRegexp(
        ValueError,
        'The given object is not a tf.keras.optimizers.Optimizer instance'):
      optimizers.get_optimizer_instance_v2(_optimizer_fn)


if __name__ == '__main__':
  tf.test.main()
