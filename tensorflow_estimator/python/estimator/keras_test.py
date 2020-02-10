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
"""Tests for training routines."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import tempfile

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import keras
from tensorflow.python.feature_column import dense_features
from tensorflow.python.feature_column import dense_features_v2
from tensorflow.python.framework import test_util
from tensorflow.python.keras import optimizers as optimizer_v1
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers import recurrent_v2 as rnn_v2
from tensorflow.python.keras.optimizer_v2 import gradient_descent as optimizer_v2
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.ops.parsing_ops import gen_parsing_ops
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.training import saver as saver_lib
from tensorflow_estimator.python.estimator import keras as keras_lib
from tensorflow_estimator.python.estimator import run_config as run_config_lib
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None

_RANDOM_SEED = 1337
_TRAIN_SIZE = 200
_INPUT_SIZE = (10,)
_NUM_CLASS = 2

_TMP_DIR = '/tmp'


def simple_sequential_model():
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(16, activation='relu', input_shape=_INPUT_SIZE))
  model.add(keras.layers.Dropout(0.1))
  model.add(keras.layers.Dense(_NUM_CLASS, activation='softmax'))
  return model


def simple_functional_model(activation='relu'):
  a = keras.layers.Input(shape=_INPUT_SIZE, name='input_layer')
  b = keras.layers.Dense(16, activation=activation)(a)
  b = keras.layers.Dropout(0.1)(b)
  b = keras.layers.Dense(_NUM_CLASS, activation='softmax')(b)
  model = keras.models.Model(inputs=[a], outputs=[b])
  return model


def simple_subclassed_model():

  class SimpleModel(keras.Model):

    def __init__(self):
      super(SimpleModel, self).__init__()
      self.dense1 = keras.layers.Dense(16, activation='relu')
      self.dp = keras.layers.Dropout(0.1)
      self.dense2 = keras.layers.Dense(_NUM_CLASS, activation='softmax')

    def call(self, inputs):
      x = self.dense1(inputs)
      x = self.dp(x)
      return self.dense2(x)

  return SimpleModel()


def gen_input_fn(x, y=None, batch_size=128, num_epochs=1, shuffle=False):

  def input_fn():
    ds = tf.compat.v1.data.Dataset.from_tensor_slices((
        x, y) if y is not None else x)
    if shuffle:
      ds = ds.shuffle(1000)
    return ds.repeat(num_epochs).batch(batch_size)

  return input_fn


def get_multi_inputs_multi_outputs_data():
  (a_train, c_train), (a_test, c_test) = testing_utils.get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=(16,),
      num_classes=3,
      random_seed=_RANDOM_SEED)
  (b_train, d_train), (b_test, d_test) = testing_utils.get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=(16,),
      num_classes=2,
      random_seed=_RANDOM_SEED)
  (m_train, _), (m_test, _) = testing_utils.get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=(8,),
      num_classes=2,
      random_seed=_RANDOM_SEED)

  c_train = np_utils.to_categorical(c_train)
  c_test = np_utils.to_categorical(c_test)
  d_train = np_utils.to_categorical(d_train)
  d_test = np_utils.to_categorical(d_test)

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


def get_resource_for_simple_model(
    model_type='sequential',
    is_evaluate=False,
):
  if model_type == 'sequential':
    model = simple_sequential_model()
    model.build()
  elif model_type == 'subclass':
    model = simple_subclassed_model()
  else:
    assert model_type == 'functional'
    model = simple_functional_model()

  if model_type == 'subclass':
    input_name = 'input_1'
    output_name = 'output_1'
  else:
    input_name = model.input_names[0]
    output_name = model.output_names[0]

  np.random.seed(_RANDOM_SEED)
  (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=_INPUT_SIZE,
      num_classes=_NUM_CLASS)
  y_train = np_utils.to_categorical(y_train)
  y_test = np_utils.to_categorical(y_test)

  train_input_fn = gen_input_fn(
      x=randomize_io_type(x_train, input_name),
      y=randomize_io_type(y_train, output_name),
      shuffle=False,
      num_epochs=None,
      batch_size=16)

  evaluate_input_fn = gen_input_fn(
      x=randomize_io_type(x_test, input_name),
      y=randomize_io_type(y_test, output_name),
      num_epochs=1,
      shuffle=False)

  predict_input_fn = gen_input_fn(
      x=randomize_io_type(x_test, input_name), num_epochs=1, shuffle=False)

  inference_input_fn = evaluate_input_fn if is_evaluate else predict_input_fn

  return model, (x_train, y_train), (x_test,
                                     y_test), train_input_fn, inference_input_fn


def randomize_io_type(array, name):
  switch = np.random.random()
  if switch > 0.5:
    return array
  else:
    return {name: array}


def multi_inputs_multi_outputs_model():
  input_a = keras.layers.Input(shape=(16,), name='input_a')
  input_b = keras.layers.Input(shape=(16,), name='input_b')
  input_m = keras.layers.Input(shape=(8,), dtype='string', name='input_m')
  dense = keras.layers.Dense(8, name='dense_1')

  interm_a = dense(input_a)
  # Read m
  interm_m = keras.layers.Lambda(gen_parsing_ops.string_to_number)(input_m)
  interm_s = keras.layers.Lambda(lambda k: k[0] * k[1])([interm_m, interm_a])
  interm_b = dense(input_b)
  merged = keras.layers.concatenate([interm_s, interm_b], name='merge')
  output_c = keras.layers.Dense(3, activation='softmax', name='dense_2')(merged)
  output_d = keras.layers.Dense(2, activation='softmax', name='dense_3')(merged)
  model = keras.models.Model(
      inputs=[input_a, input_b, input_m], outputs=[output_c, output_d])
  model.compile(
      loss='categorical_crossentropy',
      optimizer='rmsprop',
      metrics={
          'dense_2': 'categorical_accuracy',
          'dense_3': 'categorical_accuracy'
      })
  return model


class MyHook(tf.compat.v1.train.SessionRunHook):

  def begin(self):
    _ = tf.compat.v1.get_variable('temp', [1])


class TestKerasEstimator(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    self._base_dir = os.path.join(self.get_temp_dir(), 'keras_estimator_test')
    tf.compat.v1.gfile.MakeDirs(self._base_dir)
    self._config = run_config_lib.RunConfig(
        tf_random_seed=_RANDOM_SEED, model_dir=self._base_dir)
    super(TestKerasEstimator, self).setUp()

  def tearDown(self):
    # Make sure nothing is stuck in limbo.
    tf.compat.v1.summary.FileWriterCache.clear()
    if os.path.isdir(self._base_dir):
      tf.compat.v1.gfile.DeleteRecursively(self._base_dir)
    keras.backend.clear_session()
    super(TestKerasEstimator, self).tearDown()

  @parameterized.named_parameters(
      dict(
          testcase_name='functional',
          model_type='functional',
          checkpoint_format='saver'),
      dict(
          testcase_name='sequential',
          model_type='sequential',
          checkpoint_format='saver'),
      dict(
          testcase_name='subclass',
          model_type='subclass',
          optimizer='tf_rmsprop',
          checkpoint_format='saver'),
      dict(
          testcase_name='functional_object_ckpt',
          model_type='functional',
          checkpoint_format='checkpoint'),
      dict(
          testcase_name='sequential_object_ckpt_w_fit',
          model_type='sequential',
          checkpoint_format='checkpoint',
          fit_before_export=True,
          optimizer='tf_rmsprop'),
      dict(
          testcase_name='functional_w_fit',
          model_type='functional',
          fit_before_export=True,
          optimizer='tf_rmsprop',
          checkpoint_format='saver'),
      dict(
          testcase_name='subclass_w_fit',
          model_type='subclass',
          fit_before_export=True,
          optimizer='tf_rmsprop',
          checkpoint_format='saver'),
      # b/109935364
      dict(
          testcase_name='hooks',
          model_type='subclass',
          hook=MyHook,
          optimizer='tf_rmsprop',
          checkpoint_format='saver'),
      dict(
          testcase_name='hooks_and_fit',
          model_type='subclass',
          hook=MyHook,
          fit_before_export=True,
          optimizer='tf_rmsprop',
          checkpoint_format='saver'),
      dict(
          testcase_name='tf_optimizer',
          model_type='subclass',
          hook=MyHook,
          optimizer='tf_rmsprop',
          fit_before_export=True,
          checkpoint_format='saver'))
  def test_train_keras_estimator(self,
                                 model_type,
                                 checkpoint_format=None,
                                 fit_before_export=False,
                                 optimizer='rmsprop',
                                 hook=None):
    hooks = [hook()] if hook else None
    tf_optimizer = False
    if optimizer == 'tf_rmsprop':
      tf_optimizer = True
      optimizer = tf.compat.v1.train.RMSPropOptimizer(1e-3)

    keras_model, (x_train, y_train), (_, _), train_input_fn, eval_input_fn = (
        get_resource_for_simple_model(model_type=model_type, is_evaluate=True))
    keras_model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    if fit_before_export:
      keras_model.fit(x_train, y_train, epochs=1)

    est_keras = keras_lib.model_to_estimator(
        keras_model=keras_model,
        config=self._config,
        checkpoint_format=checkpoint_format)

    est_keras.train(
        input_fn=train_input_fn, steps=_TRAIN_SIZE / 16, hooks=hooks)
    before_eval_results = est_keras.evaluate(input_fn=eval_input_fn, steps=1)
    est_keras.train(
        input_fn=train_input_fn, steps=_TRAIN_SIZE / 16, hooks=hooks)
    after_eval_results = est_keras.evaluate(input_fn=eval_input_fn, steps=1)
    self.assertLess(after_eval_results['loss'], before_eval_results['loss'])

    if checkpoint_format == 'object' and tf_optimizer:
      latest_checkpoint = tf.train.latest_checkpoint(est_keras.model_dir)
      keras_model.load_weights(latest_checkpoint)

  def test_train_with_dense_features(self):
    feature_dict = {
        'sex': np.int64([1, 1, 1, 1, 0]),
        'cp': np.int64([0, 3, 3, 2, 1]),
        'slope': np.int64([3, 2, 0, 3, 1]),
    }
    label = np.int64([0, 1, 0, 0, 0])
    train_input_fn = numpy_io.numpy_input_fn(
        x=feature_dict, y=label, num_epochs=1, shuffle=False)
    feature_columns = list()
    input_features = dict()
    for feature_name, data_array in feature_dict.items():
      feature_columns.append(
          tf.feature_column.indicator_column(
              tf.feature_column.categorical_column_with_identity(
                  key=feature_name,
                  num_buckets=np.size(np.unique(data_array)))))
      input_features[feature_name] = keras.layers.Input(
          name=feature_name,
          shape=(np.size(np.unique(data_array)),),
          dtype=tf.dtypes.int64)

    x = dense_features.DenseFeatures(feature_columns)(input_features)
    x = keras.layers.Dense(16, activation='relu')(x)
    logits = keras.layers.Dense(1, activation='linear')(x)
    model = keras.Model(inputs=input_features, outputs=logits)

    model.compile(
        optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    estimator_model = keras_lib.model_to_estimator(keras_model=model)
    estimator_model.train(input_fn=train_input_fn, steps=5)

  # TODO(b/139845232): Enable after TF2 nightly's start.
  def DISABLED_test_train_with_dense_features_embedding(self):
    feature_dict = {
        'sex': np.int64([1, 1, 1, 1, 0]),
        'cp': np.int64([0, 3, 3, 2, 1]),
        'slope': np.int64([3, 2, 0, 3, 1]),
    }
    label = np.int64([0, 1, 0, 0, 0])
    train_input_fn = numpy_io.numpy_input_fn(
        x=feature_dict, y=label, num_epochs=1, shuffle=False)
    feature_columns = list()
    input_features = dict()
    for feature_name, data_array in feature_dict.items():
      feature_columns.append(
          tf.feature_column.embedding_column(
              tf.feature_column.categorical_column_with_identity(
                  key=feature_name, num_buckets=np.size(np.unique(data_array))),
              dimension=3))
      input_features[feature_name] = keras.layers.Input(
          name=feature_name,
          shape=(np.size(np.unique(data_array)),),
          dtype=tf.dtypes.int64)

    df = dense_features.DenseFeatures(feature_columns)
    x = df(input_features)
    x = keras.layers.Dense(16, activation='relu')(x)
    logits = keras.layers.Dense(1, activation='linear')(x)
    model = keras.Model(inputs=input_features, outputs=logits)

    model.compile(
        optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    estimator_model = keras_lib.model_to_estimator(keras_model=model)
    estimator_model.train(input_fn=train_input_fn, steps=5)
    # We assert that we find the embedding_weights variables in the dependencies
    # for the DenseFeatures layer.
    dependency_names = [x.name for x in df._checkpoint_dependencies]
    self.assertNotIn('embedding_weights', dependency_names)
    self.assertIn('cp_embedding/embedding_weights', dependency_names)
    self.assertIn('sex_embedding/embedding_weights', dependency_names)
    self.assertIn('slope_embedding/embedding_weights', dependency_names)

  # TODO(b/139845232): Enable after TF2 nightly's start.
  def DISABLED_test_train_with_dense_features_v2(self):
    feature_dict = {
        'sex': np.int64([1, 1, 1, 1, 0]),
        'cp': np.int64([0, 3, 3, 2, 1]),
        'slope': np.int64([3, 2, 0, 3, 1]),
    }
    label = np.int64([0, 1, 0, 0, 0])
    train_input_fn = numpy_io.numpy_input_fn(
        x=feature_dict, y=label, num_epochs=1, shuffle=False)
    feature_columns = list()
    input_features = dict()
    for feature_name, data_array in feature_dict.items():
      feature_columns.append(
          tf.feature_column.embedding_column(
              tf.feature_column.categorical_column_with_identity(
                  key=feature_name, num_buckets=np.size(np.unique(data_array))),
              dimension=3))
      input_features[feature_name] = keras.layers.Input(
          name=feature_name,
          shape=(np.size(np.unique(data_array)),),
          dtype=tf.dtypes.int64)

    df = dense_features_v2.DenseFeatures(feature_columns)
    x = df(input_features)
    x = keras.layers.Dense(16, activation='relu')(x)
    logits = keras.layers.Dense(1, activation='linear')(x)
    model = keras.Model(inputs=input_features, outputs=logits)

    model.compile(
        optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    estimator_model = keras_lib.model_to_estimator(keras_model=model)
    estimator_model.train(input_fn=train_input_fn, steps=5)
    # We assert that we find the embedding_weights variables in the dependencies
    # for the DenseFeatures layer.
    dependency_names = [x.name for x in df._checkpoint_dependencies]
    self.assertNotIn('embedding_weights', dependency_names)
    self.assertIn('cp_embedding/embedding_weights', dependency_names)
    self.assertIn('sex_embedding/embedding_weights', dependency_names)
    self.assertIn('slope_embedding/embedding_weights', dependency_names)

  def test_evaluate(self):
    keras_model, (x_train, y_train), (
        x_test, y_test), _, eval_input_fn = get_resource_for_simple_model(
            model_type='functional', is_evaluate=True)

    metrics = [
        'binary_accuracy', 'binary_crossentropy', 'categorical_accuracy',
        'categorical_crossentropy', 'cosine_proximity', 'hinge',
        'kullback_leibler_divergence', 'mean_absolute_error',
        'mean_absolute_percentage_error', 'mean_squared_error',
        'mean_squared_logarithmic_error', 'poisson', 'squared_hinge',
        'top_k_categorical_accuracy'
    ]
    keras_model.compile(
        loss='categorical_crossentropy', optimizer='adam', metrics=metrics)
    keras_model.fit(x_train, y_train, epochs=1)
    keras_eval = keras_model.evaluate(x_test, y_test, batch_size=32)

    keras_est = keras_lib.model_to_estimator(
        keras_model=keras_model, config=self._config)
    est_eval = keras_est.evaluate(input_fn=eval_input_fn)

    metrics = ['loss'] + metrics

    # Check loss and all metrics match between keras and estimator.
    def shift(val):
      if val == 0:
        return 0
      else:
        return val / 10**int(math.log10(abs(val)))

    for i, metric_name in enumerate(metrics):
      self.assertAlmostEqual(
          shift(keras_eval[i]),
          shift(est_eval[metric_name]),
          places=4,
          msg='%s mismatch, keras model: %s, estimator: %s' %
          (metric_name, keras_eval[i], est_eval[metric_name]))

  def test_predict(self):
    # Check that predict on a pretrained model yield the same result.
    keras_model, (x_train, y_train), (
        x_test, _), _, pred_input_fn = get_resource_for_simple_model(
            model_type='sequential', is_evaluate=False)

    keras_model.compile(
        loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  def test_multi_inputs_multi_outputs_with_input_fn_as_dict(self):
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
      return input_dict, output_dict

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
      return input_dict, output_dict

    def pred_input_fn():
      input_dict = {
          'input_a': test_data['input_a'],
          'input_b': test_data['input_b'],
          'input_m': test_data['input_m'].astype(np.str)
      }
      return input_dict

    self.do_test_multi_inputs_multi_outputs_with_input_fn(
        train_input_fn, eval_input_fn, pred_input_fn)

  def test_multi_inputs_multi_outputs_with_input_fn_as_list(self):
    train_data, test_data = get_multi_inputs_multi_outputs_data()

    def train_input_fn():
      input_list = [
          train_data['input_a'], train_data['input_b'],
          train_data['input_m'].astype(np.str)
      ]
      output_list = [train_data['output_c'], train_data['output_d']]
      return input_list, output_list

    def eval_input_fn():
      input_list = [
          test_data['input_a'], test_data['input_b'],
          test_data['input_m'].astype(np.str)
      ]
      output_list = [test_data['output_c'], test_data['output_d']]
      return input_list, output_list

    def pred_input_fn():
      input_list = [
          test_data['input_a'], test_data['input_b'],
          test_data['input_m'].astype(np.str)
      ]
      return input_list

    self.do_test_multi_inputs_multi_outputs_with_input_fn(
        train_input_fn, eval_input_fn, pred_input_fn)

  def do_test_multi_inputs_multi_outputs_with_input_fn(self, train_input_fn,
                                                       eval_input_fn,
                                                       pred_input_fn):
    model = multi_inputs_multi_outputs_model()
    est_keras = keras_lib.model_to_estimator(
        keras_model=model, config=self._config)
    baseline_eval_results = est_keras.evaluate(input_fn=eval_input_fn, steps=1)
    est_keras.train(input_fn=train_input_fn, steps=_TRAIN_SIZE / 16)
    eval_results = est_keras.evaluate(input_fn=eval_input_fn, steps=1)
    self.assertLess(eval_results['loss'], baseline_eval_results['loss'])
    est_keras.predict(input_fn=pred_input_fn)

  def test_init_from_file(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    keras_model, (x_train, y_train), (
        x_test, _), _, pred_input_fn = get_resource_for_simple_model(
            model_type='functional', is_evaluate=False)

    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['categorical_accuracy'])
    keras_model.fit(x_train, y_train, epochs=1)
    keras_pred = [np.argmax(y) for y in keras_model.predict(x_test)]
    fname = os.path.join(self._base_dir, 'keras_model.h5')
    keras.models.save_model(keras_model, fname)

    keras_est = keras_lib.model_to_estimator(
        keras_model_path=fname, config=self._config)
    est_pred = [
        np.argmax(y[keras_model.output_names[0]])
        for y in keras_est.predict(input_fn=pred_input_fn)
    ]
    self.assertAllEqual(est_pred, keras_pred)

  def test_keras_model_init_error(self):
    with self.assertRaisesRegexp(ValueError, 'Either'):
      keras_lib.model_to_estimator()

    keras_model = simple_sequential_model()
    with self.assertRaisesRegexp(ValueError, 'not both'):
      keras_lib.model_to_estimator(
          keras_model=keras_model,
          keras_model_path=tempfile.mkdtemp(dir=self._base_dir))

    keras_model = simple_sequential_model()
    with self.assertRaisesRegexp(ValueError, 'compiled'):
      keras_lib.model_to_estimator(keras_model=keras_model)

  def test_invalid_ionames_error(self):
    (x_train, y_train), (_, _) = testing_utils.get_test_data(
        train_samples=_TRAIN_SIZE,
        test_samples=100,
        input_shape=(10,),
        num_classes=2)
    y_train = np_utils.to_categorical(y_train)

    def invald_input_name_input_fn():
      input_dict = {'invalid_input_name': x_train}
      return input_dict, y_train

    def invald_output_name_input_fn():
      input_dict = {'input_layer': x_train}
      output_dict = {'invalid_output_name': y_train}
      return input_dict, output_dict

    model = simple_functional_model()
    model.compile(
        loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    est_keras = keras_lib.model_to_estimator(
        keras_model=model, config=self._config)
    with self.assertRaisesRegexp(
        KeyError,
        'features keys: .*invalid_input_name.*Missed keys: .*input_layer'):
      est_keras.train(input_fn=invald_input_name_input_fn, steps=100)

    with self.assertRaisesRegexp(
        KeyError, 'labels keys: .*invalid_output_name.*Missed keys: .*dense_1'):
      est_keras.train(input_fn=invald_output_name_input_fn, steps=100)

  def test_custom_objects(self):

    def relu6(x):
      return keras.backend.relu(x, max_value=6)

    keras_model = simple_functional_model(activation=relu6)
    keras_model.compile(loss='categorical_crossentropy', optimizer='adam')
    custom_objects = {'relu6': relu6}

    (x_train, y_train), _ = testing_utils.get_test_data(
        train_samples=_TRAIN_SIZE,
        test_samples=50,
        input_shape=(10,),
        num_classes=2)
    y_train = np_utils.to_categorical(y_train, 2)
    input_name = keras_model.input_names[0]
    output_name = keras_model.output_names[0]
    train_input_fn = gen_input_fn(
        x=randomize_io_type(x_train, input_name),
        y=randomize_io_type(y_train, output_name),
        shuffle=False,
        num_epochs=None,
        batch_size=16)
    with self.assertRaisesRegexp(ValueError, 'relu6'):
      est = keras_lib.model_to_estimator(
          keras_model=keras_model,
          model_dir=tempfile.mkdtemp(dir=self._base_dir))
      est.train(input_fn=train_input_fn, steps=1)

    est = keras_lib.model_to_estimator(
        keras_model=keras_model,
        model_dir=tempfile.mkdtemp(dir=self._base_dir),
        custom_objects=custom_objects)
    est.train(input_fn=train_input_fn, steps=1)

  def test_tf_config(self):
    keras_model, (_, _), (_, _), _, _ = get_resource_for_simple_model()
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['mse', keras.metrics.CategoricalAccuracy()])

    tf_config = json.dumps({
        'cluster': {
            run_config_lib.TaskType.PS: ['localhost:1234'],
            run_config_lib.TaskType.WORKER: ['localhost:1236'],
            run_config_lib.TaskType.MASTER: ['localhost:1238']
        },
        'task': {
            'type': run_config_lib.TaskType.MASTER,
            'index': 0
        }
    })
    with tf.compat.v1.test.mock.patch.dict('os.environ',
                                           {'TF_CONFIG': tf_config}):
      keras_lib.model_to_estimator(
          keras_model=keras_model,
          model_dir=tempfile.mkdtemp(dir=self._base_dir))

  def test_gpu_config(self):
    with tf.Graph().as_default():
      keras_model, (_, _), (_, _), _, _ = get_resource_for_simple_model()
      keras_model.compile(
          loss='categorical_crossentropy',
          optimizer='rmsprop',
          metrics=['mse', keras.metrics.CategoricalAccuracy()])

      gpu_options = config_pb2.GPUOptions(per_process_gpu_memory_fraction=0.3)
      sess_config = config_pb2.ConfigProto(gpu_options=gpu_options)
      self._config._session_config = sess_config
      keras_lib.model_to_estimator(keras_model=keras_model, config=self._config)
      self.assertEqual(
          keras.backend.get_session(
          )._config.gpu_options.per_process_gpu_memory_fraction,
          gpu_options.per_process_gpu_memory_fraction)

  def test_with_empty_config(self):
    keras_model, _, _, _, _ = get_resource_for_simple_model(
        model_type='sequential', is_evaluate=True)
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['mse', keras.metrics.CategoricalAccuracy()])

    est_keras = keras_lib.model_to_estimator(
        keras_model=keras_model,
        model_dir=self._base_dir,
        config=run_config_lib.RunConfig())
    self.assertEqual(run_config_lib.get_default_session_config(),
                     est_keras._session_config)
    self.assertEqual(est_keras._session_config,
                     est_keras._config.session_config)
    self.assertEqual(self._base_dir, est_keras._config.model_dir)
    self.assertEqual(self._base_dir, est_keras._model_dir)

    est_keras = keras_lib.model_to_estimator(
        keras_model=keras_model, model_dir=self._base_dir, config=None)
    self.assertEqual(run_config_lib.get_default_session_config(),
                     est_keras._session_config)
    self.assertEqual(est_keras._session_config,
                     est_keras._config.session_config)
    self.assertEqual(self._base_dir, est_keras._config.model_dir)
    self.assertEqual(self._base_dir, est_keras._model_dir)

  def test_with_empty_config_and_empty_model_dir(self):
    keras_model, _, _, _, _ = get_resource_for_simple_model(
        model_type='sequential', is_evaluate=True)
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['mse', keras.metrics.CategoricalAccuracy()])

    with tf.compat.v1.test.mock.patch.object(
        tempfile, 'mkdtemp', return_value=_TMP_DIR):
      est_keras = keras_lib.model_to_estimator(
          keras_model=keras_model, config=run_config_lib.RunConfig())
      self.assertEqual(est_keras._model_dir, _TMP_DIR)

  def test_with_conflicting_model_dir_and_config(self):
    keras_model, _, _, _, _ = get_resource_for_simple_model(
        model_type='sequential', is_evaluate=True)
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['mse', keras.metrics.CategoricalAccuracy()])

    with self.assertRaisesRegexp(
        ValueError, '`model_dir` are set both in '
        'constructor and `RunConfig`'):
      keras_lib.model_to_estimator(
          keras_model=keras_model,
          model_dir=self._base_dir,
          config=run_config_lib.RunConfig(model_dir=_TMP_DIR))

  def test_pretrained_weights(self):
    keras_model, (_, _), (_, _), _, _ = get_resource_for_simple_model()
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.compat.v1.train.RMSPropOptimizer(1e-3),
        metrics=['mse', keras.metrics.CategoricalAccuracy()])
    keras_model.train_on_batch(
        np.random.random((10,) + _INPUT_SIZE), np.random.random(
            (10, _NUM_CLASS)))
    weights = keras_model.get_weights()
    keras_model, (_, _), (_, _), _, _ = get_resource_for_simple_model()
    keras_model.set_weights(weights)

    if tf.executing_eagerly():
      sgd_optimizer = optimizer_v2.SGD(lr=0.0001, momentum=0.9)
    else:
      sgd_optimizer = optimizer_v1.SGD(lr=0.0001, momentum=0.9)
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd_optimizer,
        metrics=['mse', keras.metrics.CategoricalAccuracy()])
    keras_lib.model_to_estimator(keras_model=keras_model, config=self._config)

  def assert_increasing_global_step(self, optimizer):
    keras_model, _, _, train_input_fn, _ = get_resource_for_simple_model(
        model_type='sequential', is_evaluate=True)
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['mse', keras.metrics.CategoricalAccuracy()])
    with self.cached_session() as sess:
      keras_model_fn = keras_lib._create_keras_model_fn(keras_model)
      global_step = tf.compat.v1.train.create_global_step()
      features, labels = train_input_fn().make_one_shot_iterator().get_next()
      spec = keras_model_fn(features, labels, mode=ModeKeys.TRAIN)

      sess.run(tf.compat.v1.initializers.global_variables())
      sess.run(tf.compat.v1.initializers.local_variables())

      self.assertEqual(global_step.eval(), 0)  # Sanity check
      sess.run(spec.train_op)
      self.assertEqual(global_step.eval(), 1)

  @test_util.run_v1_only('training_util.create_global_step is v1 only.')
  def test_model_fn_increments_global_step_tf_optimizer(self):
    self.assert_increasing_global_step(
        tf.compat.v1.train.RMSPropOptimizer(1e-3))

  @test_util.run_v1_only('training_util.create_global_step is v1 only.')
  def test_model_fn_increments_global_step_keras_optimizer(self):
    self.assert_increasing_global_step('rmsprop')

  @parameterized.named_parameters(
      dict(testcase_name='object_ckpt', checkpoint_format='checkpoint'),
      dict(testcase_name='name_ckpt', checkpoint_format='saver'))
  def test_export_keras_estimator(self, checkpoint_format):
    keras_model, (x_train, y_train), (
        _, _), train_input_fn, _ = get_resource_for_simple_model(
            model_type='sequential', is_evaluate=False)

    keras_model.compile(
        loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    keras_model.fit(x_train, y_train, epochs=1)
    bias_value = keras.backend.get_value(keras_model.layers[0].bias)

    est_keras = keras_lib.model_to_estimator(
        keras_model=keras_model,
        model_dir=tempfile.mkdtemp(dir=self._base_dir),
        checkpoint_format=checkpoint_format)

    def serving_input_receiver_fn():
      feature_spec = {
          'dense_input': tf.io.FixedLenFeature([1], dtype=tf.dtypes.float32)
      }
      return export_lib.build_parsing_serving_input_receiver_fn(feature_spec)

    # Try immediately exporting, testing that (1) exported values are the same,
    # and (2) estimator can be exported without saving a checkpoint into the
    # model directory.
    saved_model_dir = est_keras.export_saved_model(
        tempfile.mkdtemp(dir=self._base_dir), serving_input_receiver_fn())
    variables_path = saved_model_utils.get_variables_path(saved_model_dir)

    variable_name = 'dense/bias'
    if checkpoint_format == 'checkpoint':
      names_to_keys = saver_lib.object_graph_key_mapping(variables_path)
      variable_name = names_to_keys[variable_name]

    self.assertAllClose(bias_value,
                        tf.train.load_variable(variables_path, variable_name))

    # Export the estimator after training a bit.
    est_keras.train(input_fn=train_input_fn, steps=_TRAIN_SIZE / 16)
    saved_model_dir = est_keras.export_saved_model(
        tempfile.mkdtemp(dir=self._base_dir), serving_input_receiver_fn())
    variables_path = saved_model_utils.get_variables_path(saved_model_dir)
    self.assertNotAllClose(
        bias_value, tf.train.load_variable(variables_path, variable_name))

  def test_export_subclassed_model_retains_model_state(self):
    keras_model, (x_train, y_train), (
        _, _), train_input_fn, eval_input_fn = get_resource_for_simple_model(
            model_type='subclass', is_evaluate=True)
    keras_model.compile(
        optimizer=tf.compat.v1.train.RMSPropOptimizer(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    keras_model.fit(x_train, y_train, epochs=1)
    iterations = keras.backend.get_value(keras_model.optimizer.iterations)
    optimizer = keras_model.optimizer
    est_keras = keras_lib.model_to_estimator(
        keras_model=keras_model, config=self._config, checkpoint_format='saver')
    est_keras.train(input_fn=train_input_fn, steps=_TRAIN_SIZE / 16)

    # Subclassed models resets the model object. Assert that attributes are
    # properly restored.
    iterations_after = keras.backend.get_value(keras_model.optimizer.iterations)
    self.assertEqual(optimizer, keras_model.optimizer)
    self.assertEqual(iterations, iterations_after)
    # TODO(b/132839451): model.fit results in an error after model_to_estimator.
    # keras_model.fit(x_train, y_train, epochs=1)

  def test_warm_start_from_keras_ckpt(self):
    keras_model, (x_train, y_train), (
        _, _), train_input_fn, eval_input_fn = get_resource_for_simple_model(
            model_type='functional', is_evaluate=True)
    keras_model.compile(
        optimizer=tf.compat.v1.train.RMSPropOptimizer(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    keras_model.fit(x_train, y_train, epochs=1)

    warm_start_path = os.path.join(self._config.model_dir, 'keras',
                                   'warm_start.ckpt')
    keras_model.save_weights(warm_start_path)

    est_keras = keras_lib.model_to_estimator(
        keras_model=keras_model, config=self._config, checkpoint_format='saver')

    self.assertEqual(warm_start_path,
                     est_keras._warm_start_settings.ckpt_to_initialize_from)
    before_eval_results = est_keras.evaluate(input_fn=eval_input_fn, steps=1)
    est_keras.train(input_fn=train_input_fn, steps=_TRAIN_SIZE / 16)
    after_eval_results = est_keras.evaluate(input_fn=eval_input_fn, steps=1)
    self.assertLess(after_eval_results['loss'], before_eval_results['loss'])

  def test_sample_weights(self):
    # Create simple pass-through model
    input_layer = keras.layers.Input(shape=1, name='input_layer')
    keras_model = keras.Model(inputs=input_layer, outputs=input_layer)

    keras_model.compile(loss='mean_absolute_error', optimizer='adam')

    features = [[0.], [0], [1], [1]]
    sample_weights = [0, .4, 1, 1]
    targets = [[0], [1], [0], [1]]

    expected_loss = keras_model.test_on_batch(
        tf.constant(features), tf.constant(targets),
        tf.constant(sample_weights))

    def input_fn():
      dataset = tf.compat.v1.data.Dataset.from_tensors(({
          'features': features,
          'sample_weights': sample_weights
      }, targets))
      return dataset

    est_keras = keras_lib.model_to_estimator(
        keras_model=keras_model, model_dir=tempfile.mkdtemp(dir=self._base_dir))
    eval_results = est_keras.evaluate(input_fn, steps=1)
    self.assertAllClose(expected_loss, eval_results['loss'])

    # Test multiple with outputs and sample weights.
    keras_model = keras.Model(
        inputs=input_layer, outputs=[input_layer, input_layer])
    keras_model.compile(loss='mean_absolute_error', optimizer='adam')
    expected_loss = keras_model.test_on_batch(
        tf.constant(features),
        [tf.constant(targets), tf.constant(targets)],
        [tf.constant(sample_weights),
         tf.constant(sample_weights)])[0]

    def input_fn_multiple_targets():
      dataset = tf.compat.v1.data.Dataset.from_tensors(
          (features, sample_weights, targets))
      dataset = dataset.map(lambda x, y, z: ({
          'features': x,
          'sample_weights': (y, y)
      }, (z, z)))
      return dataset

    est_keras = keras_lib.model_to_estimator(
        keras_model=keras_model, model_dir=tempfile.mkdtemp(dir=self._base_dir))
    eval_results = est_keras.evaluate(input_fn_multiple_targets, steps=1)
    self.assertAllClose(expected_loss, eval_results['loss'])

  @parameterized.parameters([rnn_v2.LSTM, rnn_v2.GRU])
  def test_model_to_estimator_with_rnn(self, layer):
    # See https://github.com/tensorflow/tensorflow/issues/27750 for details.
    timestep = 10
    rnn_cell_size = 8

    layers = [
        keras.layers.Reshape([timestep, 1], input_shape=[
            timestep,
        ]),
        layer(rnn_cell_size, return_sequences=True),
        layer(rnn_cell_size),
        keras.layers.Dense(1)
    ]

    model = keras.Sequential(layers)
    model.compile(loss='mse', optimizer='sgd')
    keras_lib.model_to_estimator(
        keras_model=model,
        checkpoint_format='checkpoint',
        model_dir=tempfile.mkdtemp(dir=self._base_dir))


if __name__ == '__main__':
  tf.test.main()
