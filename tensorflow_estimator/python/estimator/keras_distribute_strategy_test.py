# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras model-to-estimator using tf.distribute.Strategy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import test
from tensorflow.python.ops.parsing_ops import gen_parsing_ops
from tensorflow_estimator.python.estimator import keras as keras_lib
from tensorflow_estimator.python.estimator import run_config as run_config_lib

_RANDOM_SEED = 1337
_TRAIN_SIZE = 200
_INPUT_SIZE = (10,)
_NUM_CLASS = 2


def simple_sequential_model():
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=_INPUT_SIZE))
  model.add(tf.keras.layers.Dropout(0.1))
  model.add(tf.keras.layers.Dense(_NUM_CLASS, activation='softmax'))
  return model


def simple_functional_model():
  a = tf.keras.layers.Input(shape=_INPUT_SIZE)
  b = tf.keras.layers.Dense(16, activation='relu')(a)
  b = tf.keras.layers.Dropout(0.1)(b)
  b = tf.keras.layers.Dense(_NUM_CLASS, activation='softmax')(b)
  model = tf.keras.models.Model(inputs=[a], outputs=[b])
  return model


def multi_inputs_multi_outputs_model():
  input_a = tf.keras.layers.Input(shape=(16,), name='input_a')
  input_b = tf.keras.layers.Input(shape=(16,), name='input_b')
  input_m = tf.keras.layers.Input(shape=(8,), dtype='string', name='input_m')
  dense = tf.keras.layers.Dense(8, name='dense_1')

  interm_a = dense(input_a)
  # Read m
  interm_m = tf.keras.layers.Lambda(gen_parsing_ops.string_to_number)(input_m)
  interm_s = tf.keras.layers.Lambda(lambda k: k[0] * k[1])([interm_m, interm_a])
  interm_b = dense(input_b)
  merged = tf.keras.layers.concatenate([interm_s, interm_b], name='merge')
  output_c = tf.keras.layers.Dense(3, activation='softmax', name='dense_2')(
      merged)
  output_d = tf.keras.layers.Dense(2, activation='softmax', name='dense_3')(
      merged)
  model = tf.keras.models.Model(
      inputs=[input_a, input_b, input_m], outputs=[output_c, output_d])
  model.compile(
      loss='categorical_crossentropy',
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics={
          'dense_2': 'categorical_accuracy',
          'dense_3': 'categorical_accuracy'
      })
  return model


def get_ds_train_input_fn():
  np.random.seed(_RANDOM_SEED)
  (x_train, y_train), _ = get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=_INPUT_SIZE,
      num_classes=_NUM_CLASS)
  y_train = tf.keras.utils.to_categorical(y_train)

  dataset = tf.compat.v1.data.Dataset.from_tensor_slices((x_train, y_train))
  dataset = dataset.batch(32)
  return dataset


def get_ds_test_input_fn():
  np.random.seed(_RANDOM_SEED)
  _, (x_test, y_test) = get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=_INPUT_SIZE,
      num_classes=_NUM_CLASS)
  y_test = tf.keras.utils.to_categorical(y_test)

  dataset = tf.compat.v1.data.Dataset.from_tensor_slices((x_test, y_test))
  dataset = dataset.batch(32)
  return dataset


def get_multi_inputs_multi_outputs_data():
  (a_train, c_train), (a_test, c_test) = get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=(16,),
      num_classes=3,
      random_seed=_RANDOM_SEED)
  (b_train, d_train), (b_test, d_test) = get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=(16,),
      num_classes=2,
      random_seed=_RANDOM_SEED)
  (m_train, _), (m_test, _) = get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=(8,),
      num_classes=2,
      random_seed=_RANDOM_SEED)

  c_train = tf.keras.utils.to_categorical(c_train)
  c_test = tf.keras.utils.to_categorical(c_test)
  d_train = tf.keras.utils.to_categorical(d_train)
  d_test = tf.keras.utils.to_categorical(d_test)

  train_data = {
      'input_a': a_train,
      'input_b': b_train,
      'input_m': m_train,
      'output_c': c_train,
      'output_d': d_train
  }
  test_data = {
      'input_a': a_test,
      'input_b': b_test,
      'input_m': m_test,
      'output_c': c_test,
      'output_d': d_test
  }

  return (train_data, test_data)


class TestEstimatorDistributionStrategy(tf.test.TestCase,
                                        parameterized.TestCase):

  def setUp(self):
    super(TestEstimatorDistributionStrategy, self).setUp()
    strategy_combinations.set_virtual_cpus_to_at_least(3)
    self._base_dir = os.path.join(self.get_temp_dir(),
                                  'keras_to_estimator_strategy_test')
    tf.compat.v1.gfile.MakeDirs(self._base_dir)
    self._config = run_config_lib.RunConfig(
        tf_random_seed=_RANDOM_SEED, model_dir=self._base_dir)

  def tearDown(self):
    super(TestEstimatorDistributionStrategy, self).tearDown()
    tf.compat.v1.summary.FileWriterCache.clear()
    if os.path.isdir(self._base_dir):
      tf.compat.v1.gfile.DeleteRecursively(self._base_dir)

  @tf.compat.v2.__internal__.distribute.combinations.generate(
      tf.compat.v2.__internal__.test.combinations.combine(
          distribution=[
              tf.compat.v2.__internal__.distribute.combinations.mirrored_strategy_with_cpu_1_and_2,
          ],
          mode=['graph'],
          cloning=[True, False]))
  def test_train_functional_with_distribution_strategy(self, distribution,
                                                       cloning):
    keras_model = simple_functional_model()
    keras_model.compile(
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
        cloning=cloning)
    config = run_config_lib.RunConfig(
        tf_random_seed=_RANDOM_SEED,
        model_dir=self._base_dir,
        train_distribute=distribution,
        eval_distribute=distribution)
    with self.cached_session():
      est_keras = keras_lib.model_to_estimator(
          keras_model=keras_model, config=config)
      before_eval_results = est_keras.evaluate(
          input_fn=get_ds_test_input_fn, steps=1)
      est_keras.train(input_fn=get_ds_train_input_fn, steps=_TRAIN_SIZE / 16)
      after_eval_results = est_keras.evaluate(
          input_fn=get_ds_test_input_fn, steps=1)
      self.assertLess(after_eval_results['loss'], before_eval_results['loss'])

    tf.compat.v1.summary.FileWriterCache.clear()
    tf.compat.v1.gfile.DeleteRecursively(self._config.model_dir)

  @tf.compat.v2.__internal__.distribute.combinations.generate(
      tf.compat.v2.__internal__.test.combinations.combine(
          distribution=[
              tf.compat.v2.__internal__.distribute.combinations.mirrored_strategy_with_cpu_1_and_2,
          ],
          mode=['graph'],
          cloning=[True, False]))
  def test_train_sequential_with_distribution_strategy(self, distribution,
                                                       cloning):
    keras_model = simple_sequential_model()
    keras_model.compile(
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
        cloning=cloning)
    config = run_config_lib.RunConfig(
        tf_random_seed=_RANDOM_SEED,
        model_dir=self._base_dir,
        train_distribute=distribution)
    with self.cached_session():
      est_keras = keras_lib.model_to_estimator(
          keras_model=keras_model, config=config)
      before_eval_results = est_keras.evaluate(
          input_fn=get_ds_test_input_fn, steps=1)
      est_keras.train(input_fn=get_ds_train_input_fn, steps=_TRAIN_SIZE / 16)
      after_eval_results = est_keras.evaluate(
          input_fn=get_ds_test_input_fn, steps=1)
      self.assertLess(after_eval_results['loss'], before_eval_results['loss'])

    tf.compat.v1.summary.FileWriterCache.clear()
    tf.compat.v1.gfile.DeleteRecursively(self._config.model_dir)

  @tf.compat.v2.__internal__.distribute.combinations.generate(
      tf.compat.v2.__internal__.test.combinations.combine(
          distribution=[
              tf.compat.v2.__internal__.distribute.combinations.mirrored_strategy_with_cpu_1_and_2,
          ],
          mode=['graph']))
  def test_multi_inputs_multi_outputs_with_input_fn_as_dict(self, distribution):
    train_data, test_data = get_multi_inputs_multi_outputs_data()

    def train_input_fn():
      input_dict = {
          'input_a': train_data['input_a'],
          'input_b': train_data['input_b'],
          'input_m': train_data['input_m'].astype(np.str)
      }
      output_dict = {
          'dense_2': train_data['output_c'],
          'dense_3': train_data['output_d']
      }
      return tf.compat.v1.data.Dataset.from_tensor_slices(
          (input_dict, output_dict)).batch(16)

    def eval_input_fn():
      input_dict = {
          'input_a': test_data['input_a'],
          'input_b': test_data['input_b'],
          'input_m': test_data['input_m'].astype(np.str)
      }
      output_dict = {
          'dense_2': test_data['output_c'],
          'dense_3': test_data['output_d']
      }
      return tf.compat.v1.data.Dataset.from_tensor_slices(
          (input_dict, output_dict)).batch(16)

    self.do_test_multi_inputs_multi_outputs_with_input_fn(
        distribution, train_input_fn, eval_input_fn)

  def do_test_multi_inputs_multi_outputs_with_input_fn(self, distribution,
                                                       train_input_fn,
                                                       eval_input_fn):
    config = run_config_lib.RunConfig(
        tf_random_seed=_RANDOM_SEED,
        model_dir=self._base_dir,
        train_distribute=distribution)
    with self.cached_session():
      model = multi_inputs_multi_outputs_model()
      est_keras = keras_lib.model_to_estimator(keras_model=model, config=config)
      baseline_eval_results = est_keras.evaluate(
          input_fn=eval_input_fn, steps=1)
      est_keras.train(input_fn=train_input_fn, steps=_TRAIN_SIZE / 16)
      eval_results = est_keras.evaluate(input_fn=eval_input_fn, steps=1)
      self.assertLess(eval_results['loss'], baseline_eval_results['loss'])


def get_test_data(train_samples,
                  test_samples,
                  input_shape,
                  num_classes,
                  random_seed=None):
  if random_seed is not None:
    np.random.seed(random_seed)
  num_sample = train_samples + test_samples
  templates = 2 * num_classes * np.random.random((num_classes,) + input_shape)
  y = np.random.randint(0, num_classes, size=(num_sample,))
  x = np.zeros((num_sample,) + input_shape, dtype=np.float32)
  for i in range(num_sample):
    x[i] = templates[y[i]] + np.random.normal(loc=0, scale=1., size=input_shape)
  return ((x[:train_samples], y[:train_samples]),
          (x[train_samples:], y[train_samples:]))


if __name__ == '__main__':
  test.main()
