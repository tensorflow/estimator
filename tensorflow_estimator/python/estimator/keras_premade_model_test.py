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

from tensorflow.python import keras
from tensorflow.python.feature_column import dense_features
from tensorflow.python.feature_column import feature_column_v2 as feature_column
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.premade import linear
from tensorflow.python.keras.premade import wide_deep
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer_cache
from tensorflow_estimator.python.estimator import keras as keras_lib
from tensorflow_estimator.python.estimator import run_config as run_config_lib
from tensorflow_estimator.python.estimator.inputs import numpy_io

_RANDOM_SEED = 1337


class KerasPremadeModelTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._base_dir = os.path.join(self.get_temp_dir(), 'keras_estimator_test')
    gfile.MakeDirs(self._base_dir)
    self._config = run_config_lib.RunConfig(
        tf_random_seed=_RANDOM_SEED, model_dir=self._base_dir)
    super(KerasPremadeModelTest, self).setUp()

  def tearDown(self):
    # Make sure nothing is stuck in limbo.
    writer_cache.FileWriterCache.clear()
    if os.path.isdir(self._base_dir):
      gfile.DeleteRecursively(self._base_dir)
    keras.backend.clear_session()
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
    cat_column = feature_column.categorical_column_with_vocabulary_list(
        key='symbol', vocabulary_list=vocab_list)
    ind_column = feature_column.indicator_column(cat_column)
    keras_input = keras.layers.Input(
        name='symbol', shape=3, dtype=dtypes.string)
    feature_layer = dense_features.DenseFeatures([ind_column])
    h = feature_layer({'symbol': keras_input})
    linear_model = linear.LinearModel(units=1)
    h = linear_model(h)

    model = keras.Model(inputs=keras_input, outputs=h)
    opt = gradient_descent.SGD(0.1)
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

  def test_train_premade_widedeep_model_with_feature_layers(self):
    vocab_list = ['alpha', 'beta', 'gamma']
    vocab_val = [0.4, 0.6, 0.9]
    data = np.random.choice(vocab_list, size=256)
    y = np.zeros_like(data, dtype=np.float32)
    for vocab, val in zip(vocab_list, vocab_val):
      indices = np.where(data == vocab)
      y[indices] = val + np.random.uniform(
          low=-0.01, high=0.01, size=indices[0].shape)
    cat_column = feature_column.categorical_column_with_vocabulary_list(
        key='symbol', vocabulary_list=vocab_list)
    ind_column = feature_column.indicator_column(cat_column)
    # TODO(tanzheny): use emb column for dense part once b/139667019 is fixed.
    # emb_column = feature_column.embedding_column(cat_column, dimension=5)
    keras_input = keras.layers.Input(
        name='symbol', shape=3, dtype=dtypes.string)

    # build linear part with feature layer.
    linear_feature_layer = dense_features.DenseFeatures([ind_column])
    linear_model = linear.LinearModel(
        units=1, name='Linear', kernel_initializer='zeros')
    combined_linear = keras.Sequential([linear_feature_layer, linear_model])

    # build dnn part with feature layer.
    dnn_feature_layer = dense_features.DenseFeatures([ind_column])
    dense_layer = keras.layers.Dense(
        units=1, name='DNNDense', kernel_initializer='zeros')
    combined_dnn = keras.Sequential([dnn_feature_layer, dense_layer])

    # build and compile wide deep.
    wide_deep_model = wide_deep.WideDeepModel(combined_linear, combined_dnn)
    wide_deep_model._set_inputs({'symbol': keras_input})
    sgd_opt = gradient_descent.SGD(0.1)
    adam_opt = adam.Adam(0.1)
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
  test.main()
