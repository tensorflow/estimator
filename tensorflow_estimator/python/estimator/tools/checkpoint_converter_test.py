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
"""Tests for checkpoint_converter.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import tensorflow as tf
import numpy as np

from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer_cache
from tensorflow_estimator.python.estimator.canned import dnn
from tensorflow_estimator.python.estimator.canned import dnn_linear_combined
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import linear
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.tools import checkpoint_converter


class DNNCheckpointConverterTest(tf.test.TestCase):

  def setUp(self):
    self._old_ckpt_dir = os.path.join(self.get_temp_dir(), 'source_ckpt')
    self._new_ckpt_dir = os.path.join(self.get_temp_dir(), 'target_ckpt')

  def tearDown(self):
    if os.path.exists(self._old_ckpt_dir):
      tf.compat.v1.summary.FileWriterCache.clear()
      shutil.rmtree(self._old_ckpt_dir)
    if os.path.exists(self._new_ckpt_dir):
      tf.compat.v1.summary.FileWriterCache.clear()
      shutil.rmtree(self._new_ckpt_dir)

  def _test_ckpt_converter(self, train_input_fn, eval_input_fn,
                           predict_input_fn, input_dimension, label_dimension,
                           batch_size, optimizer):

    # Create checkpoint in CannedEstimator v1.
    feature_columns_v1 = [
        feature_column._numeric_column('x', shape=(input_dimension,))
    ]

    est_v1 = dnn.DNNEstimator(
        head=head_lib._regression_head(label_dimension=label_dimension),
        hidden_units=(2, 2),
        feature_columns=feature_columns_v1,
        model_dir=self._old_ckpt_dir,
        optimizer=optimizer)
    # Train
    num_steps = 10
    est_v1.train(train_input_fn, steps=num_steps)
    self.assertIsNotNone(est_v1.latest_checkpoint())
    self.assertTrue(est_v1.latest_checkpoint().startswith(self._old_ckpt_dir))

    # Convert checkpoint from v1 to v2.
    source_checkpoint = os.path.join(self._old_ckpt_dir, 'model.ckpt-10')
    source_graph = os.path.join(self._old_ckpt_dir, 'graph.pbtxt')
    target_checkpoint = os.path.join(self._new_ckpt_dir, 'model.ckpt-10')
    checkpoint_converter.convert_checkpoint('dnn', source_checkpoint,
                                            source_graph, target_checkpoint)

    # Create CannedEstimator V2 and restore from the converted checkpoint.
    feature_columns_v2 = [
        tf.feature_column.numeric_column('x', shape=(input_dimension,))
    ]
    est_v2 = dnn.DNNEstimatorV2(
        head=regression_head.RegressionHead(label_dimension=label_dimension),
        hidden_units=(2, 2),
        feature_columns=feature_columns_v2,
        model_dir=self._new_ckpt_dir,
        optimizer=optimizer)
    # Train
    extra_steps = 10
    est_v2.train(train_input_fn, steps=extra_steps)
    self.assertIsNotNone(est_v2.latest_checkpoint())
    self.assertTrue(est_v2.latest_checkpoint().startswith(self._new_ckpt_dir))
    # Make sure estimator v2 restores from the converted checkpoint, and
    # continues training extra steps.
    self.assertEqual(num_steps + extra_steps,
                     est_v2.get_variable_value(tf.compat.v1.GraphKeys.GLOBAL_STEP))

  def _create_input_fn(self, label_dimension, batch_size):
    """Creates input_fn for integration test."""
    data = np.linspace(0., 2., batch_size * label_dimension, dtype=np.float32)
    data = data.reshape(batch_size, label_dimension)
    # learn y = x
    train_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, y=data, batch_size=batch_size, shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, batch_size=batch_size, shuffle=False)

    return train_input_fn, eval_input_fn, predict_input_fn

  def _test_ckpt_converter_with_an_optimizer(self, opt):
    """Tests checkpoint converter with an optimizer."""
    label_dimension = 2
    batch_size = 10
    train_input_fn, eval_input_fn, predict_input_fn = self._create_input_fn(
        label_dimension, batch_size)

    self._test_ckpt_converter(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        input_dimension=label_dimension,
        label_dimension=label_dimension,
        batch_size=batch_size,
        optimizer=opt)

  def test_ckpt_converter_with_adagrad(self):
    """Tests checkpoint converter with Adagrad."""
    self._test_ckpt_converter_with_an_optimizer('Adagrad')

  def test_ckpt_converter_with_rmsprop(self):
    """Tests checkpoint converter with RMSProp."""
    self._test_ckpt_converter_with_an_optimizer('RMSProp')

  def test_ckpt_converter_with_ftrl(self):
    """Tests checkpoint converter with Ftrl."""
    self._test_ckpt_converter_with_an_optimizer('Ftrl')

  def test_ckpt_converter_with_adam(self):
    """Tests checkpoint converter with Adam."""
    self._test_ckpt_converter_with_an_optimizer('Adam')

  def test_ckpt_converter_with_sgd(self):
    """Tests checkpoint converter with SGD."""
    self._test_ckpt_converter_with_an_optimizer('SGD')


class LinearCheckpointConverterTest(tf.test.TestCase):

  def setUp(self):
    self._old_ckpt_dir = os.path.join(self.get_temp_dir(), 'source_ckpt')
    self._new_ckpt_dir = os.path.join(self.get_temp_dir(), 'target_ckpt')

  def tearDown(self):
    if os.path.exists(self._old_ckpt_dir):
      tf.compat.v1.summary.FileWriterCache.clear()
      shutil.rmtree(self._old_ckpt_dir)
    if os.path.exists(self._new_ckpt_dir):
      tf.compat.v1.summary.FileWriterCache.clear()
      shutil.rmtree(self._new_ckpt_dir)

  def _test_ckpt_converter(self, train_input_fn, eval_input_fn,
                           predict_input_fn, input_dimension, label_dimension,
                           batch_size, optimizer):

    # Create checkpoint in CannedEstimator v1.
    feature_columns_v1 = [
        feature_column._numeric_column('x', shape=(input_dimension,))
    ]

    est_v1 = linear.LinearEstimator(
        head=head_lib._regression_head(label_dimension=label_dimension),
        feature_columns=feature_columns_v1,
        model_dir=self._old_ckpt_dir,
        optimizer=optimizer)
    # Train
    num_steps = 10
    est_v1.train(train_input_fn, steps=num_steps)
    self.assertIsNotNone(est_v1.latest_checkpoint())
    self.assertTrue(est_v1.latest_checkpoint().startswith(self._old_ckpt_dir))

    # Convert checkpoint from v1 to v2.
    source_checkpoint = os.path.join(self._old_ckpt_dir, 'model.ckpt-10')
    source_graph = os.path.join(self._old_ckpt_dir, 'graph.pbtxt')
    target_checkpoint = os.path.join(self._new_ckpt_dir, 'model.ckpt-10')
    checkpoint_converter.convert_checkpoint('linear', source_checkpoint,
                                            source_graph, target_checkpoint)

    # Create CannedEstimator V2 and restore from the converted checkpoint.
    feature_columns_v2 = [
        tf.feature_column.numeric_column('x', shape=(input_dimension,))
    ]
    est_v2 = linear.LinearEstimatorV2(
        head=regression_head.RegressionHead(label_dimension=label_dimension),
        feature_columns=feature_columns_v2,
        model_dir=self._new_ckpt_dir,
        optimizer=optimizer)
    # Train
    extra_steps = 10
    est_v2.train(train_input_fn, steps=extra_steps)
    self.assertIsNotNone(est_v2.latest_checkpoint())
    self.assertTrue(est_v2.latest_checkpoint().startswith(self._new_ckpt_dir))
    # Make sure estimator v2 restores from the converted checkpoint, and
    # continues training extra steps.
    self.assertEqual(num_steps + extra_steps,
                     est_v2.get_variable_value(tf.compat.v1.GraphKeys.GLOBAL_STEP))

  def _create_input_fn(self, label_dimension, batch_size):
    """Creates input_fn for integration test."""
    data = np.linspace(0., 2., batch_size * label_dimension, dtype=np.float32)
    data = data.reshape(batch_size, label_dimension)
    # learn y = x
    train_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, y=data, batch_size=batch_size, shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, batch_size=batch_size, shuffle=False)

    return train_input_fn, eval_input_fn, predict_input_fn

  def _test_ckpt_converter_with_an_optimizer(self, opt):
    """Tests checkpoint converter with an optimizer."""
    label_dimension = 2
    batch_size = 10
    train_input_fn, eval_input_fn, predict_input_fn = self._create_input_fn(
        label_dimension, batch_size)

    self._test_ckpt_converter(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        input_dimension=label_dimension,
        label_dimension=label_dimension,
        batch_size=batch_size,
        optimizer=opt)

  def test_ckpt_converter_with_adagrad(self):
    """Tests checkpoint converter with Adagrad."""
    self._test_ckpt_converter_with_an_optimizer('Adagrad')

  def test_ckpt_converter_with_rmsprop(self):
    """Tests checkpoint converter with RMSProp."""
    self._test_ckpt_converter_with_an_optimizer('RMSProp')

  def test_ckpt_converter_with_ftrl(self):
    """Tests checkpoint converter with Ftrl."""
    self._test_ckpt_converter_with_an_optimizer('Ftrl')

  def test_ckpt_converter_with_adam(self):
    """Tests checkpoint converter with Adam."""
    self._test_ckpt_converter_with_an_optimizer('Adam')

  def test_ckpt_converter_with_sgd(self):
    """Tests checkpoint converter with SGD."""
    self._test_ckpt_converter_with_an_optimizer('SGD')


class DNNLinearCombinedCheckpointConverterTest(tf.test.TestCase):

  def setUp(self):
    self._old_ckpt_dir = os.path.join(self.get_temp_dir(), 'source_ckpt')
    self._new_ckpt_dir = os.path.join(self.get_temp_dir(), 'target_ckpt')

  def tearDown(self):
    if os.path.exists(self._old_ckpt_dir):
      tf.compat.v1.summary.FileWriterCache.clear()
      shutil.rmtree(self._old_ckpt_dir)
    if os.path.exists(self._new_ckpt_dir):
      tf.compat.v1.summary.FileWriterCache.clear()
      shutil.rmtree(self._new_ckpt_dir)

  def _test_ckpt_converter(self, train_input_fn, eval_input_fn,
                           predict_input_fn, input_dimension, label_dimension,
                           batch_size, dnn_optimizer, linear_optimizer):

    # Create checkpoint in CannedEstimator v1.
    linear_feature_columns_v1 = [
        feature_column._numeric_column('x', shape=(input_dimension,))
    ]
    dnn_feature_columns_v1 = [
        feature_column._numeric_column('x', shape=(input_dimension,))
    ]
    est_v1 = dnn_linear_combined.DNNLinearCombinedEstimator(
        head=head_lib._regression_head(label_dimension=label_dimension),
        linear_feature_columns=linear_feature_columns_v1,
        dnn_feature_columns=dnn_feature_columns_v1,
        dnn_hidden_units=(2, 2),
        model_dir=self._old_ckpt_dir,
        dnn_optimizer=dnn_optimizer,
        linear_optimizer=linear_optimizer)
    # Train
    num_steps = 10
    est_v1.train(train_input_fn, steps=num_steps)
    self.assertIsNotNone(est_v1.latest_checkpoint())
    self.assertTrue(est_v1.latest_checkpoint().startswith(self._old_ckpt_dir))

    # Convert checkpoint from v1 to v2.
    source_checkpoint = os.path.join(self._old_ckpt_dir, 'model.ckpt-10')
    source_graph = os.path.join(self._old_ckpt_dir, 'graph.pbtxt')
    target_checkpoint = os.path.join(self._new_ckpt_dir, 'model.ckpt-10')
    checkpoint_converter.convert_checkpoint('combined', source_checkpoint,
                                            source_graph, target_checkpoint)

    # Create CannedEstimator V2 and restore from the converted checkpoint.
    linear_feature_columns_v2 = [
        tf.feature_column.numeric_column('x', shape=(input_dimension,))
    ]
    dnn_feature_columns_v2 = [
        tf.feature_column.numeric_column('x', shape=(input_dimension,))
    ]
    est_v2 = dnn_linear_combined.DNNLinearCombinedEstimatorV2(
        head=regression_head.RegressionHead(label_dimension=label_dimension),
        linear_feature_columns=linear_feature_columns_v2,
        dnn_feature_columns=dnn_feature_columns_v2,
        dnn_hidden_units=(2, 2),
        model_dir=self._new_ckpt_dir,
        dnn_optimizer=dnn_optimizer,
        linear_optimizer=linear_optimizer)
    # Train
    extra_steps = 10
    est_v2.train(train_input_fn, steps=extra_steps)
    self.assertIsNotNone(est_v2.latest_checkpoint())
    self.assertTrue(est_v2.latest_checkpoint().startswith(self._new_ckpt_dir))
    # Make sure estimator v2 restores from the converted checkpoint, and
    # continues training extra steps.
    self.assertEqual(num_steps + extra_steps,
                     est_v2.get_variable_value(tf.compat.v1.GraphKeys.GLOBAL_STEP))

  def _create_input_fn(self, label_dimension, batch_size):
    """Creates input_fn for integration test."""
    data = np.linspace(0., 2., batch_size * label_dimension, dtype=np.float32)
    data = data.reshape(batch_size, label_dimension)
    # learn y = x
    train_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, y=data, batch_size=batch_size, shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, batch_size=batch_size, shuffle=False)

    return train_input_fn, eval_input_fn, predict_input_fn

  def _test_ckpt_converter_with_an_optimizer(self, dnn_opt, linear_opt):
    """Tests checkpoint converter with an optimizer."""
    label_dimension = 2
    batch_size = 10
    train_input_fn, eval_input_fn, predict_input_fn = self._create_input_fn(
        label_dimension, batch_size)

    self._test_ckpt_converter(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        input_dimension=label_dimension,
        label_dimension=label_dimension,
        batch_size=batch_size,
        dnn_optimizer=dnn_opt,
        linear_optimizer=linear_opt)

  def test_ckpt_converter_with_adagrad(self):
    """Tests checkpoint converter with Adagrad."""
    self._test_ckpt_converter_with_an_optimizer('Adagrad', 'RMSProp')

  def test_ckpt_converter_with_rmsprop(self):
    """Tests checkpoint converter with RMSProp."""
    self._test_ckpt_converter_with_an_optimizer('RMSProp', 'Ftrl')

  def test_ckpt_converter_with_ftrl(self):
    """Tests checkpoint converter with Ftrl."""
    self._test_ckpt_converter_with_an_optimizer('Ftrl', 'Adam')

  def test_ckpt_converter_with_adam(self):
    """Tests checkpoint converter with Adam."""
    self._test_ckpt_converter_with_an_optimizer('Adam', 'SGD')

  def test_ckpt_converter_with_sgd(self):
    """Tests checkpoint converter with SGD."""
    self._test_ckpt_converter_with_an_optimizer('SGD', 'Adagrad')


if __name__ == '__main__':
  tf.test.main()
