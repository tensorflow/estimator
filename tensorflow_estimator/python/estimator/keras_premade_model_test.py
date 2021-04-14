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
"""Tests for keras premade model in model_to_estimator routines."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from tensorflow_estimator.python.estimator import keras as keras_lib
from tensorflow_estimator.python.estimator import run_config as run_config_lib
from tensorflow_estimator.python.estimator.inputs import numpy_io

_RANDOM_SEED = 1337


def gen_input_fn(x, y=None, batch_size=32, num_epochs=10, shuffle=False):

  def input_fn():
    ds = tf.compat.v1.data.Dataset.from_tensor_slices((
        x, y) if y is not None else x)
    if shuffle:
      ds = ds.shuffle(1000)
    return ds.repeat(num_epochs).batch(batch_size)

  return input_fn


def get_resource_for_simple_model():

  input_name = 'input_1'
  output_name = 'output_1'

  np.random.seed(_RANDOM_SEED)
  x_train = np.random.uniform(low=-5, high=5, size=(64, 2)).astype('f')
  y_train = .3 * x_train[:, 0] + .2 * x_train[:, 1]
  x_test = np.random.uniform(low=-5, high=5, size=(64, 2)).astype('f')
  y_test = .3 * x_test[:, 0] + .2 * x_test[:, 1]

  train_input_fn = gen_input_fn(
      x=x_train, y=y_train, num_epochs=None, shuffle=False)

  evaluate_input_fn = gen_input_fn(
      x=randomize_io_type(x_test, input_name),
      y=randomize_io_type(y_test, output_name),
      num_epochs=1,
      shuffle=False)

  return (x_train, y_train), (x_test, y_test), train_input_fn, evaluate_input_fn


def randomize_io_type(array, name):
  switch = np.random.random()
  if switch > 0.5:
    return array
  else:
    return {name: array}


class KerasPremadeModelTest(tf.test.TestCase):

  def setUp(self):
    self._base_dir = os.path.join(self.get_temp_dir(), 'keras_estimator_test')
    tf.compat.v1.gfile.MakeDirs(self._base_dir)
    self._config = run_config_lib.RunConfig(
        tf_random_seed=_RANDOM_SEED, model_dir=self._base_dir)
    super(KerasPremadeModelTest, self).setUp()

  def tearDown(self):
    # Make sure nothing is stuck in limbo.
    tf.compat.v1.summary.FileWriterCache.clear()
    if os.path.isdir(self._base_dir):
      tf.compat.v1.gfile.DeleteRecursively(self._base_dir)
    tf.keras.backend.clear_session()
    super(KerasPremadeModelTest, self).tearDown()

  def test_train_premade_linear_model_with_dense_features(self):
    vocab_list = ['alpha', 'beta', 'gamma']
    vocab_val = [0.4, 0.6, 0.9]
    data = np.random.choice(vocab_list, size=256)
    y = np.zeros_like(data, dtype=np.float32)
    for vocab, val in zip(vocab_list, vocab_val):
      indices = np.where(data == vocab)
      y[indices] = val + np.random.uniform(
          low=-0.01, high=0.01, size=indices[0].shape)
    cat_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='symbol', vocabulary_list=vocab_list)
    ind_column = tf.feature_column.indicator_column(cat_column)
    keras_input = tf.keras.layers.Input(
        name='symbol', shape=3, dtype=tf.dtypes.string)
    feature_layer = tf.compat.v1.keras.layers.DenseFeatures([ind_column])
    h = feature_layer({'symbol': keras_input})
    linear_model = tf.keras.experimental.LinearModel(units=1)
    h = linear_model(h)

    model = tf.keras.models.Model(inputs=keras_input, outputs=h)
    opt = tf.keras.optimizers.SGD(0.1)
    model.compile(opt, 'mse', ['mse'])
    train_input_fn = numpy_io.numpy_input_fn(
        x={'symbol': data}, y=y, num_epochs=20, shuffle=False)
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'symbol': data}, y=y, num_epochs=20, shuffle=False)
    est = keras_lib.model_to_estimator(
        keras_model=model, config=self._config, checkpoint_format='saver')
    before_eval_results = est.evaluate(input_fn=eval_input_fn, steps=1)
    est.train(input_fn=train_input_fn, steps=30)
    after_eval_results = est.evaluate(input_fn=eval_input_fn, steps=1)
    self.assertLess(after_eval_results['loss'], before_eval_results['loss'])
    self.assertLess(after_eval_results['loss'], 0.05)

  def test_train_premade_linear_model(self):
    (x_train,
     y_train), _, train_inp_fn, eval_inp_fn = get_resource_for_simple_model()

    linear_model = tf.keras.experimental.LinearModel(units=1)
    opt = tf.keras.optimizers.SGD(0.1)
    linear_model.compile(opt, 'mse', ['mse'])
    linear_model.fit(x_train, y_train, epochs=10)

    est = keras_lib.model_to_estimator(
        keras_model=linear_model,
        config=self._config,
        checkpoint_format='saver')
    before_eval_results = est.evaluate(input_fn=eval_inp_fn, steps=1)
    est.train(input_fn=train_inp_fn, steps=500)
    after_eval_results = est.evaluate(input_fn=eval_inp_fn, steps=1)
    self.assertLess(after_eval_results['loss'], before_eval_results['loss'])
    self.assertLess(after_eval_results['loss'], 0.1)

  def test_train_premade_widedeep_model_with_feature_layers(self):
    vocab_list = ['alpha', 'beta', 'gamma']
    vocab_val = [0.4, 0.6, 0.9]
    data = np.random.choice(vocab_list, size=256)
    y = np.zeros_like(data, dtype=np.float32)
    for vocab, val in zip(vocab_list, vocab_val):
      indices = np.where(data == vocab)
      y[indices] = val + np.random.uniform(
          low=-0.01, high=0.01, size=indices[0].shape)
    cat_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='symbol', vocabulary_list=vocab_list)
    ind_column = tf.feature_column.indicator_column(cat_column)
    # TODO(tanzheny): use emb column for dense part once b/139667019 is fixed.
    # emb_column = feature_column.embedding_column(cat_column, dimension=5)
    keras_input = tf.keras.layers.Input(
        name='symbol', shape=3, dtype=tf.dtypes.string)

    # build linear part with feature layer.
    linear_feature_layer = tf.compat.v1.keras.layers.DenseFeatures([ind_column])
    linear_model = tf.keras.experimental.LinearModel(
        units=1, name='Linear', kernel_initializer='zeros')
    combined_linear = tf.keras.models.Sequential([linear_feature_layer, linear_model])

    # build dnn part with feature layer.
    dnn_feature_layer = tf.compat.v1.keras.layers.DenseFeatures([ind_column])
    dense_layer = tf.keras.layers.Dense(
        units=1, name='DNNDense', kernel_initializer='zeros')
    combined_dnn = tf.keras.models.Sequential([dnn_feature_layer, dense_layer])

    # build and compile wide deep.
    wide_deep_model = tf.keras.experimental.WideDeepModel(combined_linear, combined_dnn)
    wide_deep_model._set_inputs({'symbol': keras_input})
    sgd_opt = tf.keras.optimizers.SGD(0.1)
    adam_opt = tf.keras.optimizers.Adam(0.1)
    wide_deep_model.compile([sgd_opt, adam_opt], 'mse', ['mse'])

    # build estimator.
    train_input_fn = numpy_io.numpy_input_fn(
        x={'symbol': data}, y=y, num_epochs=20, shuffle=False)
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'symbol': data}, y=y, num_epochs=20, shuffle=False)
    est = keras_lib.model_to_estimator(
        keras_model=wide_deep_model,
        config=self._config,
        checkpoint_format='saver')

    before_eval_results = est.evaluate(input_fn=eval_input_fn, steps=1)
    est.train(input_fn=train_input_fn, steps=20)
    after_eval_results = est.evaluate(input_fn=eval_input_fn, steps=1)
    self.assertLess(after_eval_results['loss'], before_eval_results['loss'])
    self.assertLess(after_eval_results['loss'], 0.1)


if __name__ == '__main__':
  tf.test.main()
