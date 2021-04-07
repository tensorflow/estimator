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
"""Tests for rnn.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tempfile

from absl.testing import parameterized
import numpy as np
import six
import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import layers as keras_layers
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import parsing_utils
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.canned import rnn
from tensorflow_estimator.python.estimator.export import export
from tensorflow_estimator.python.estimator.head import multi_class_head as multi_head_lib
from tensorflow_estimator.python.estimator.head import sequential_head as seq_head_lib
from tensorflow_estimator.python.estimator.inputs import numpy_io

# Names of variables created by BasicRNNCell model.
CELL_KERNEL_NAME = 'rnn_model/rnn/kernel'
CELL_RECURRENT_KERNEL_NAME = 'rnn_model/rnn/recurrent_kernel'
CELL_BIAS_NAME = 'rnn_model/rnn/bias'
LOGITS_WEIGHTS_NAME = 'rnn_model/logits/kernel'
LOGITS_BIAS_NAME = 'rnn_model/logits/bias'


def _assert_close(expected, actual, rtol=1e-04, name='assert_close'):
  with ops.name_scope(name, 'assert_close', (expected, actual, rtol)) as scope:
    expected = ops.convert_to_tensor(expected, name='expected')
    actual = ops.convert_to_tensor(actual, name='actual')
    rdiff = tf.math.abs(expected - actual, 'diff') / tf.math.abs(expected)
    rtol = ops.convert_to_tensor(rtol, name='rtol')
    return tf.compat.v1.debugging.assert_less(
        rdiff,
        rtol,
        data=('Condition expected =~ actual did not hold element-wise:'
              'expected = ', expected, 'actual = ', actual, 'rdiff = ', rdiff,
              'rtol = ', rtol,),
        name=scope)


def create_checkpoint(kernel, recurrent, bias, dense_kernel, dense_bias,
                      global_step, model_dir):
  """Create checkpoint file with provided model weights.

  Args:
    kernel: Iterable of values of input weights for the RNN cell.
    recurrent: Iterable of values of recurrent weights for the RNN cell.
    bias: Iterable of values of biases for the RNN cell.
    dense_kernel: Iterable of values for matrix connecting RNN output to logits.
    dense_bias: Iterable of values for logits bias term.
    global_step: Initial global step to save in checkpoint.
    model_dir: Directory into which checkpoint is saved.
  """
  model_weights = {}
  model_weights[CELL_KERNEL_NAME] = kernel
  model_weights[CELL_RECURRENT_KERNEL_NAME] = recurrent
  model_weights[CELL_BIAS_NAME] = bias
  model_weights[LOGITS_WEIGHTS_NAME] = dense_kernel
  model_weights[LOGITS_BIAS_NAME] = dense_bias

  with tf.Graph().as_default():
    # Create model variables.
    for k, v in six.iteritems(model_weights):
      tf.Variable(v, name=k, dtype=tf.dtypes.float32)

    # Create non-model variables.
    global_step_var = tf.compat.v1.train.create_global_step()
    assign_op = global_step_var.assign(global_step)

    # Initialize vars and save checkpoint.
    with tf.compat.v1.train.MonitoredTrainingSession(
        checkpoint_dir=model_dir) as sess:
      sess.run(assign_op)


def _make_rnn_layer(rnn_cell_fn=None,
                    units=None,
                    cell_type=rnn.USE_DEFAULT,
                    return_sequences=False):
  return rnn._make_rnn_layer(
      rnn_cell_fn=rnn_cell_fn,
      units=units,
      cell_type=cell_type,
      return_sequences=return_sequences)


@test_util.run_all_in_graph_and_eager_modes
class RNNLayerFnTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for rnn layer function."""

  def testWrongClassProvided(self):
    """Tests that an error is raised if the class doesn't have a call method."""
    with self.assertRaisesRegexp(
        ValueError, 'RNN cell should have a `call` and `state_size` method.'):
      _make_rnn_layer(units=[10], cell_type=lambda units: object())

  def testWrongStringProvided(self):
    """Tests that an error is raised if cell type is unknown."""
    with self.assertRaisesRegexp(
        ValueError,
        'cell_type` should be a class producing a RNN cell, or a string .*.'):
      _make_rnn_layer(units=[10], cell_type='unknown-cell-name')

  @parameterized.parameters(['simple_rnn', rnn.USE_DEFAULT])
  def testDefaultCellProvided(self, cell_type):
    """Tests behavior when the default cell type is provided."""
    layer = _make_rnn_layer(cell_type=cell_type, units=[1])
    self.assertIsInstance(layer, tf.keras.layers.RNN)
    self.assertIsInstance(layer.cell, tf.keras.layers.SimpleRNNCell)

  @parameterized.parameters([('gru', tf.keras.layers.GRU),
                             ('lstm', tf.keras.layers.LSTM),
                             ('simple_rnn', tf.keras.layers.SimpleRNN)])
  def testSpecificLayerTypeProvided(self, cell_type, layer_type):
    """Tests specific layer type for GRU and LSTM."""
    layer = _make_rnn_layer(cell_type=cell_type, units=1)
    self.assertIsInstance(layer, layer_type)

  def testSpecificLayerTypeArguments(self):
    """Tests arguments for specific layer types (GRU and LSTM)."""
    mock_layer_type = tf.compat.v1.test.mock.Mock()
    with tf.compat.v1.test.mock.patch.object(rnn, '_CELL_TYPE_TO_LAYER_MAPPING',
                                             {'custom-type': mock_layer_type}):
      _make_rnn_layer(
          cell_type='custom-type',
          units=11,
          return_sequences='return-seq-value')
      mock_layer_type.assert_called_once_with(
          units=11, return_sequences='return-seq-value')

  @tf.compat.v1.test.mock.patch.object(keras_layers, 'RNN')
  def testCustomCellProvided(self, mock_rnn_layer_type):
    """Tests behavior when a custom cell type is provided."""
    mock_custom_cell = tf.compat.v1.test.mock.Mock()
    _make_rnn_layer(
        units=[10],
        cell_type=lambda units: mock_custom_cell,
        return_sequences='return-seq-value')
    mock_rnn_layer_type.assert_called_once_with(
        cell=mock_custom_cell, return_sequences='return-seq-value')

  def testMultipleCellsProvided(self):
    """Tests behavior when multiple cells are provided."""
    layer = _make_rnn_layer(cell_type='simple_rnn', units=[1, 2])
    self.assertIsInstance(layer, tf.keras.layers.RNN)
    self.assertIsInstance(layer.cell, tf.keras.layers.StackedRNNCells)
    self.assertLen(layer.cell.cells, 2)
    self.assertIsInstance(layer.cell.cells[0], tf.keras.layers.SimpleRNNCell)

  @tf.compat.v1.test.mock.patch.object(keras_layers, 'RNN')
  def testCustomCellFnProvided(self, mock_rnn_layer_type):
    """Tests behavior when a custom cell function is provided."""
    mock_cell_fn = tf.compat.v1.test.mock.Mock(return_value='custom-cell')
    _make_rnn_layer(
        rnn_cell_fn=mock_cell_fn, return_sequences='return-seq-value')
    mock_rnn_layer_type.assert_called_once_with(
        cell='custom-cell', return_sequences='return-seq-value')


def _mock_logits_layer(kernel, bias):
  """Sets initialization values to dense `logits` layers used in context."""

  class _MockDenseLayer(tf.keras.layers.Dense):

    def __init__(self, units, activation, name):
      kwargs = {}
      if name == 'logits':
        kwargs = {
            'kernel_initializer': tf.compat.v1.initializers.constant(kernel),
            'bias_initializer': tf.compat.v1.initializers.constant(bias)
        }

      super(_MockDenseLayer, self).__init__(
          units=units, name=name, activation=activation, **kwargs)

  return tf.compat.v1.test.mock.patch.object(keras_layers, 'Dense',
                                             _MockDenseLayer)


def _default_features_fn():
  return {
      'price':
          tf.sparse.SparseTensor(
              values=[10., 5.], indices=[[0, 0], [0, 1]], dense_shape=[1, 2]),
  }


def _get_mock_head():
  mock_head = multi_head_lib.MultiClassHead(3)
  mock_head.create_estimator_spec = tf.compat.v1.test.mock.Mock(
      return_value=model_fn.EstimatorSpec(None))
  return mock_head


@test_util.run_all_in_graph_and_eager_modes
class RNNLogitFnTest(tf.test.TestCase, parameterized.TestCase):
  """Tests correctness of logits calculated from RNNModel."""

  def setUp(self):
    # Sets layers default weights for testing purpose.
    self.kernel = [[.1, -.2]]
    self.recurrent = [[.2, -.3], [.3, -.4]]
    self.bias = [.2, .5]
    self.dense_kernel = [[-1.], [1.]]
    self.dense_bias = [0.3]
    self.sequence_feature_columns = [
        tf.feature_column.sequence_numeric_column('price', shape=(1,))
    ]
    self.context_feature_columns = []
    super(RNNLogitFnTest, self).setUp()

  def _mock_logits_layer(self):
    return _mock_logits_layer(self.dense_kernel, bias=self.dense_bias)

  def _test_logits(self,
                   logits_dimension,
                   features_fn,
                   expected_logits,
                   expected_mask,
                   return_sequences=False,
                   training=False):
    """Tests that the expected logits are calculated."""
    rnn_layer = tf.keras.layers.SimpleRNN(
        2,
        return_sequences=return_sequences,
        kernel_initializer=tf.compat.v1.initializers.constant(self.kernel),
        recurrent_initializer=tf.compat.v1.initializers.constant(
            self.recurrent),
        bias_initializer=tf.compat.v1.initializers.constant(self.bias))
    with self._mock_logits_layer():
      logit_layer = rnn.RNNModel(
          rnn_layer=rnn_layer,
          units=logits_dimension,
          sequence_feature_columns=self.sequence_feature_columns,
          context_feature_columns=self.context_feature_columns,
          return_sequences=return_sequences)
    logits = logit_layer(features_fn(), training=training)
    if return_sequences:
      logits = (logits, logits._keras_mask)
      expected_logits = (expected_logits, expected_mask)
    self.evaluate(tf.compat.v1.initializers.global_variables())
    self.assertAllClose(expected_logits, self.evaluate(logits), atol=1e-4)

  @parameterized.named_parameters(
      {
          'testcase_name': 'Static',
          'return_sequences': False,
          'expected_logits': [[-0.6033]]
      }, {
          'testcase_name': 'Sequential',
          'return_sequences': True,
          'expected_logits': [[[-1.4388], [-0.6033]]]
      }, {
          'testcase_name': 'SequentialTrain',
          'return_sequences': True,
          'expected_logits': [[[-1.4388], [-0.6033]]],
          'training': True
      }, {
          'testcase_name': 'SequentialInfer',
          'return_sequences': True,
          'expected_logits': [[[-1.4388], [-0.6033]]],
          'training': False
      })
  def testOneDimLogits(self, return_sequences, expected_logits, training=False):
    """Tests one-dimensional logits.

    Intermediate values are rounded for ease in reading.
    input_layer = [[[10]], [[5]]]
    sequence_mask = [[1, 1]]
    initial_state = [0, 0]
    rnn_output_timestep_1 = [[tanh(.1*10 + .2*0 + .3*0 +.2),
                              tanh(-.2*10 - .3*0 - .4*0 +.5)]]
                          = [[0.83, -0.91]]
    rnn_output_timestep_2 = [[tanh(.1*5 + .2*.83 - .3*.91 +.2),
                              tanh(-.2*5 - .3*.83 + .4*.91 +.5)]]
                          = [[0.53, -0.37]]
    logits_timestep_1 = [[-1*0.83 - 1*0.91 + 0.3]] = [[-1.4388]]
    logits_timestep_2 = [[-1*0.53 - 1*0.37 + 0.3]] = [[-0.6033]]

    Args:
      return_sequences: A boolean indicating whether to return the last output
        in the output sequence, or the full sequence.
      expected_logits: An array with expected logits result.
      training: Specifies if this training or evaluation / prediction mode.
    """
    expected_mask = [[1, 1]]

    self._test_logits(
        logits_dimension=1,
        features_fn=_default_features_fn,
        expected_mask=expected_mask,
        expected_logits=expected_logits,
        return_sequences=return_sequences,
        training=training)

  @parameterized.named_parameters(
      {
          'testcase_name': 'Static',
          'return_sequences': False,
          'expected_logits': [[-0.6033, 0.7777, 0.5698]]
      }, {
          'testcase_name': 'Sequential',
          'return_sequences': True,
          'expected_logits': [[[-1.4388, 1.0884, 0.5762],
                               [-0.6033, 0.7777, 0.5698]]]
      })
  def testMultiDimLogits(self, return_sequences, expected_logits):
    """Tests multi-dimensional logits.

    Intermediate values are rounded for ease in reading.
    input_layer = [[[10]], [[5]]]
    sequence_mask = [[1, 1]]
    initial_state = [0, 0]
    rnn_output_timestep_1 = [[tanh(.1*10 + .2*0 + .3*0 +.2),
                              tanh(-.2*10 - .3*0 - .4*0 +.5)]]
                          = [[0.83, -0.91]]
    rnn_output_timestep_2 = [[tanh(.1*5 + .2*.83 - .3*.91 +.2),
                              tanh(-.2*5 - .3*.83 + .4*.91 +.5)]]
                          = [[0.53, -0.37]]
    logits_timestep_1 = [[-1*0.83 - 1*0.91 + 0.3],
                         [0.5*0.83 + 0.3*0.91 + 0.4],
                         [0.2*0.83 - 0.1*0.91 + 0.5]]
                      = [[-1.4388, 1.0884, 0.5762]]
    logits_timestep_2 = [[-1*0.53 - 1*0.37 + 0.3],
                         [0.5*0.53 + 0.3*0.37 + 0.4],
                         [0.2*0.53 - 0.1*0.37 + 0.5]]
                      = [[-0.6033, 0.7777, 0.5698]]

    Args:
      return_sequences: A boolean indicating whether to return the last output
        in the output sequence, or the full sequence.
      expected_logits: An array with expected logits result.
    """
    expected_mask = [[1, 1]]

    self.dense_kernel = [[-1., 0.5, 0.2], [1., -0.3, 0.1]]
    self.dense_bias = [0.3, 0.4, 0.5]
    self._test_logits(
        logits_dimension=3,
        features_fn=_default_features_fn,
        expected_mask=expected_mask,
        expected_logits=expected_logits,
        return_sequences=return_sequences)

  @parameterized.named_parameters(
      {
          'testcase_name': 'Static',
          'return_sequences': False,
          'expected_logits': [[-0.6033, 0.7777, 0.5698],
                              [-1.2473, 1.0170, 0.5745]]
      }, {
          'testcase_name': 'Sequential',
          'return_sequences': True,
          'expected_logits': [[
              [-1.4388, 1.0884, 0.5762], [-0.6033, 0.7777, 0.5698]
          ], [[0.0197, 0.5601, 0.5860], [-1.2473, 1.0170, 0.5745]]]
      })
  def testMultiExampleMultiDim(self, return_sequences, expected_logits):
    """Tests multiple examples and multi-dimensional logits.

    Intermediate values are rounded for ease in reading.
    input_layer = [[[10], [5]], [[2], [7]]]
    sequence_mask = [[1, 1], [1, 1]]
    initial_state = [[0, 0], [0, 0]]
    rnn_output_timestep_1 = [[tanh(.1*10 + .2*0 + .3*0 +.2),
                              tanh(-.2*10 - .3*0 - .4*0 +.5)],
                             [tanh(.1*2 + .2*0 + .3*0 +.2),
                              tanh(-.2*2 - .3*0 - .4*0 +.5)]]
                          = [[0.83, -0.91], [0.38, 0.10]]
    rnn_output_timestep_2 = [[tanh(.1*5 + .2*.83 - .3*.91 +.2),
                              tanh(-.2*5 - .3*.83 + .4*.91 +.5)],
                             [tanh(.1*7 + .2*.38 + .3*.10 +.2),
                              tanh(-.2*7 - .3*.38 - .4*.10 +.5)]]
                          = [[0.53, -0.37], [0.76, -0.78]
    logits_timestep_1 = [[-1*0.83 - 1*0.91 + 0.3,
                          0.5*0.83 + 0.3*0.91 + 0.4,
                          0.2*0.83 - 0.1*0.91 + 0.5],
                         [-1*0.38 + 1*0.10 + 0.3,
                          0.5*0.38 - 0.3*0.10 + 0.4,
                          0.2*0.38 + 0.1*0.10 + 0.5]]
                      = [[-1.4388, 1.0884, 0.5762], [0.0197, 0.5601, 0.5860]]
    logits_timestep_2 = [[-1*0.53 - 1*0.37 + 0.3,
                          0.5*0.53 + 0.3*0.37 + 0.4,
                          0.2*0.53 - 0.1*0.37 + 0.5],
                         [-1*0.76 - 1*0.78 + 0.3,
                          0.5*0.76 +0.3*0.78 + 0.4,
                          0.2*0.76 -0.1*0.78 + 0.5]]
                      = [[-0.6033, 0.7777, 0.5698], [-1.2473, 1.0170, 0.5745]]

    Args:
      return_sequences: A boolean indicating whether to return the last output
        in the output sequence, or the full sequence.
      expected_logits: An array with expected logits result.
    """
    expected_mask = [[1, 1], [1, 1]]

    def features_fn():
      return {
          'price':
              tf.sparse.SparseTensor(
                  values=[10., 5., 2., 7.],
                  indices=[[0, 0], [0, 1], [1, 0], [1, 1]],
                  dense_shape=[2, 2]),
      }

    self.dense_kernel = [[-1., 0.5, 0.2], [1., -0.3, 0.1]]
    self.dense_bias = [0.3, 0.4, 0.5]
    self._test_logits(
        logits_dimension=3,
        features_fn=features_fn,
        expected_mask=expected_mask,
        expected_logits=expected_logits,
        return_sequences=return_sequences)

  @parameterized.named_parameters(
      {
          'testcase_name': 'Static',
          'return_sequences': False,
          'expected_logits': [[-0.6033], [0.0197]]
      }, {
          'testcase_name': 'Sequential',
          'return_sequences': True,
          'expected_logits': [[[-1.4388], [-0.6033]], [[0.0197], [0.0197]]]
      })
  def testMultiExamplesDifferentLength(self, return_sequences, expected_logits):
    """Tests multiple examples with different lengths.

    Intermediate values are rounded for ease in reading.
    input_layer = [[[10], [5]], [[2], [0]]]
    sequence_mask = [[1, 1], [1, 0]]
    initial_state = [[0, 0], [0, 0]]
    rnn_output_timestep_1 = [[tanh(.1*10 + .2*0 + .3*0 +.2),
                              tanh(-.2*10 - .3*0 - .4*0 +.5)],
                             [tanh(.1*2 + .2*0 + .3*0 +.2),
                              tanh(-.2*2 - .3*0 - .4*0 +.5)]]
                          = [[0.83, -0.91], [0.38, 0.10]]
    rnn_output_timestep_2 = [[tanh(.1*5 + .2*.83 - .3*.91 +.2),
                              tanh(-.2*5 - .3*.83 + .4*.91 +.5)],
                             [_]]
                          = [[0.53, -0.37], [_, _]]
    logits_timestep_1 = [[-1*0.83 - 1*0.91 + 0.3],
                         [-1*0.38 + 1*0.10 + 0.3]]
                      = [[-0.4388], [0.0197]]
    logits_timestep_2 = [[-1*0.53 - 1*0.37 + 0.3],
                         [_]]
                      = [[-0.6033], [_]]

    Args:
      return_sequences: A boolean indicating whether to return the last output
        in the output sequence, or the full sequence.
      expected_logits: An array with expected logits result.
    """
    expected_mask = [[1, 1], [1, 0]]

    def features_fn():
      return {
          'price':
              tf.sparse.SparseTensor(
                  values=[10., 5., 2.],
                  indices=[[0, 0], [0, 1], [1, 0]],
                  dense_shape=[2, 2]),
      }

    self._test_logits(
        logits_dimension=1,
        features_fn=features_fn,
        expected_mask=expected_mask,
        expected_logits=expected_logits,
        return_sequences=return_sequences)

  def testMultiExamplesWithContext(self):
    """Tests multiple examples with context features.

    Intermediate values are rounded for ease in reading.
    input_layer = [[[10, -0.5], [5, -0.5]], [[2, 0.8], [0, 0]]]
    sequence_mask = [[1, 1], [1, 0]]
    initial_state = [[0, 0], [0, 0]]
    rnn_output_timestep_1 = [[tanh(.1*10 - 1*.5 + .2*0 + .3*0 +.2),
                              tanh(-.2*10 - 0.9*.5 - .3*0 - .4*0 +.5)],
                             [tanh(.1*2 + 1*.8 + .2*0 + .3*0 +.2),
                              tanh(-.2*2 + .9*.8 - .3*0 - .4*0 +.5)]]
                          = [[0.60, -0.96], [0.83, 0.68]]
    rnn_output_timestep_2 = [[tanh(.1*5 - 1*.5 + .2*.60 - .3*.96 +.2),
                              tanh(-.2*5 - .9*.5 - .3*.60 + .4*.96 +.5)],
                             [<ignored-padding>]]
                          = [[0.03, -0.63], [<ignored-padding>]]
    logits = [[-1*0.03 - 1*0.63 + 0.3],
              [-1*0.83 + 1*0.68 + 0.3]]
           = [[-0.3662], [0.1414]]
    """
    expected_mask = [[1, 1], [1, 0]]

    def features_fn():
      return {
          'price':
              tf.sparse.SparseTensor(
                  values=[10., 5., 2.],
                  indices=[[0, 0], [0, 1], [1, 0]],
                  dense_shape=[2, 2]),
          'context': [[-0.5], [0.8]],
      }

    self.context_feature_columns = [
        tf.feature_column.numeric_column('context', shape=(1,))
    ]

    self.kernel = [[.1, -.2], [1., 0.9]]
    self._test_logits(
        logits_dimension=1,
        features_fn=features_fn,
        expected_mask=expected_mask,
        expected_logits=[[-0.3662], [0.1414]])

  def testMultiExamplesMultiFeatures(self):
    """Tests examples with multiple sequential feature columns.

    Intermediate values are rounded for ease in reading.
    input_layer = [[[1, 0, 10], [0, 1, 5]], [[1, 0, 2], [0, 0, 0]]]
    sequence_mask = [[1, 1], [1, 0]]
    initial_state = [[0, 0], [0, 0]]
    rnn_output_timestep_1 = [[tanh(.5*1 + 1*0 + .1*10 + .2*0 + .3*0 +.2),
                              tanh(-.5*1 - 1*0 - .2*10 - .3*0 - .4*0 +.5)],
                             [tanh(.5*1 + 1*0 + .1*2 + .2*0 + .3*0 +.2),
                              tanh(-.5*1 - 1*0 - .2*2 - .3*0 - .4*0 +.5)]]
                          = [[0.94, -0.96], [0.72, -0.38]]
    rnn_output_timestep_2 = [[tanh(.5*0 + 1*1 + .1*5 + .2*.94 - .3*.96 +.2),
                              tanh(-.5*0 - 1*1 - .2*5 - .3*.94 + .4*.96 +.5)],
                             [<ignored-padding>]]
                          = [[0.92, -0.88], [<ignored-padding>]]
    logits = [[-1*0.92 - 1*0.88 + 0.3],
              [-1*0.72 - 1*0.38 + 0.3]]
           = [[-1.5056], [-0.7962]]
    """
    expected_mask = [[1, 1], [1, 0]]

    def features_fn():
      return {
          'price':
              tf.sparse.SparseTensor(
                  values=[10., 5., 2.],
                  indices=[[0, 0], [0, 1], [1, 0]],
                  dense_shape=[2, 2]),
          'on_sale':
              tf.sparse.SparseTensor(
                  values=[0, 1, 0],
                  indices=[[0, 0], [0, 1], [1, 0]],
                  dense_shape=[2, 2]),
      }

    price_column = tf.feature_column.sequence_numeric_column(
        'price', shape=(1,))
    on_sale_column = tf.feature_column.indicator_column(
        tf.feature_column.sequence_categorical_column_with_identity(
            'on_sale', num_buckets=2))
    self.sequence_feature_columns = [price_column, on_sale_column]

    self.kernel = [[.5, -.5], [1., -1.], [.1, -.2]]
    self._test_logits(
        logits_dimension=1,
        features_fn=features_fn,
        expected_mask=expected_mask,
        expected_logits=[[-1.5056], [-0.7962]])

  @parameterized.parameters([(model_fn.ModeKeys.TRAIN, True),
                             (model_fn.ModeKeys.EVAL, False),
                             (model_fn.ModeKeys.PREDICT, False)])
  def testTrainingMode(self, mode, expected_training_mode):
    """Tests that `training` argument is properly used."""

    class _MockRNNCell(tf.keras.layers.SimpleRNNCell):
      """Used to test that `training` argument is properly used."""

      def __init__(self, test_case):
        self._test_case = test_case
        super(_MockRNNCell, self).__init__(units=10)

      def call(self, inputs, states, training=None):
        self._test_case.assertEqual(training, expected_training_mode)
        return super(_MockRNNCell, self).call(
            inputs=inputs, states=states, training=training)

    estimator = rnn.RNNEstimator(
        head=_get_mock_head(),
        rnn_cell_fn=lambda: _MockRNNCell(self),
        sequence_feature_columns=self.sequence_feature_columns)
    features = {
        'price':
            tf.sparse.SparseTensor(
                values=[
                    10.,
                ], indices=[[0, 0]], dense_shape=[1, 1]),
    }
    estimator.model_fn(features=features, labels=None, mode=mode, config=None)


class RNNModelTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for RNNModel."""

  def setUp(self):
    super(RNNModelTest, self).setUp()
    self.kernel = [[.1, -.2]]
    self.recurrent = [[.2, -.3], [.3, -.4]]
    self.bias = [.2, .5]
    self.dense_kernel = [[-1.], [1.]]
    self.dense_bias = [0.3]
    self.sequence_feature_columns = [
        tf.feature_column.sequence_numeric_column('price', shape=(1,))
    ]
    self.x = {
        'price':
            tf.sparse.SparseTensor(
                values=[10., 5., 2.],
                indices=[[0, 0], [0, 1], [1, 0]],
                dense_shape=[2, 2]),
    }
    self.y = ops.convert_to_tensor([[[0], [1]], [[0], [1]]])

  def _get_compiled_model(self,
                          return_sequences=False,
                          optimizer='Adam',
                          **kwargs):
    """Initializes and compiles a RNN model with specific weights."""
    rnn_layer = tf.keras.layers.SimpleRNN(
        2,
        return_sequences=return_sequences,
        kernel_initializer=tf.compat.v1.initializers.constant(self.kernel),
        recurrent_initializer=tf.compat.v1.initializers.constant(
            self.recurrent),
        bias_initializer=tf.compat.v1.initializers.constant(self.bias))
    with _mock_logits_layer(self.dense_kernel, bias=self.dense_bias):
      model = rnn.RNNModel(
          units=1,
          rnn_layer=rnn_layer,
          sequence_feature_columns=self.sequence_feature_columns,
          activation=tf.keras.activations.sigmoid,
          return_sequences=return_sequences,
          **kwargs)
      model.compile(
          optimizer=optimizer,
          loss=tf.keras.losses.BinaryCrossentropy(reduction='sum'),
          metrics=['accuracy'])
    return model

  def testModelWeights(self):
    """Tests that the layers weights are properly added to the model weights."""
    col = tf.feature_column.categorical_column_with_hash_bucket(
        'tokens', hash_bucket_size=1)
    context_feature_columns = [
        tf.feature_column.embedding_column(col, dimension=1)
    ]
    seq_col = tf.feature_column.sequence_categorical_column_with_hash_bucket(
        'seq-tokens', hash_bucket_size=1)
    sequence_feature_columns = [
        tf.feature_column.embedding_column(seq_col, dimension=1)
    ]
    model = rnn.RNNModel(
        units=1,
        rnn_layer=tf.keras.layers.SimpleRNN(2),
        sequence_feature_columns=sequence_feature_columns,
        activation=tf.keras.activations.sigmoid,
        context_feature_columns=context_feature_columns)
    model.compile(
        optimizer='Adam',
        loss=tf.keras.losses.BinaryCrossentropy(reduction='sum'),
        metrics=['accuracy'])

    model.predict(
        x={
            'tokens': ops.convert_to_tensor([['a']]),
            'seq-tokens': ops.convert_to_tensor([[['a']]])
        },
        steps=1)
    # Weights included are:
    # - recurrent, kernel and bias from RNN layer
    # - kernel and bias from logits layer
    # - sequential feature column embedding
    # - context feature column embedding.
    self.assertLen(model.get_weights(), 7)

  def _testModelConfig(self, **kwargs):
    """Tests the parameters of a RNNModel stored to and restored from config.

    Args:
      **kwargs: Additional keyword arguments to initialize the RNNModel before
        calling `get_config`.

    Returns:
      A dictionary with RNNModel initialization arguments from the `from_config`
      call.
    """
    seq_col = tf.feature_column.sequence_categorical_column_with_hash_bucket(
        'seq-tokens', hash_bucket_size=1)
    sequence_feature_columns = [
        tf.feature_column.embedding_column(
            seq_col, dimension=1, initializer=tf.compat.v1.initializers.zeros())
    ]
    model = rnn.RNNModel(
        units=11,
        rnn_layer=tf.keras.layers.SimpleRNN(3),
        sequence_feature_columns=sequence_feature_columns,
        return_sequences=True,
        name='rnn-model',
        **kwargs)

    with tf.compat.v1.test.mock.patch.object(
        rnn.RNNModel, '__init__', return_value=None) as init:
      rnn.RNNModel.from_config(
          model.get_config(),
          custom_objects={'Zeros': tf.compat.v1.initializers.zeros})
      return list(init.call_args_list[0])[1]

  def testModelConfig(self):
    """Tests that a RNNModel can be stored to and restored from config."""
    init_kwargs = self._testModelConfig()
    self.assertEqual(init_kwargs['name'], 'rnn-model')
    self.assertEqual(init_kwargs['units'], 11)
    self.assertEqual(init_kwargs['return_sequences'], True)
    self.assertEqual(
        init_kwargs['sequence_feature_columns'][0].categorical_column.name,
        'seq-tokens')
    self.assertEqual(init_kwargs['context_feature_columns'], None)
    self.assertEqual(init_kwargs['activation'].__name__, 'linear')
    self.assertEqual(init_kwargs['rnn_layer'].cell.units, 3)

  def testModelConfigWithActivation(self):
    """Tests store / restore from config with logits activation."""
    init_kwargs = self._testModelConfig(activation=tf.keras.activations.sigmoid)
    self.assertEqual(init_kwargs['activation'].__name__, 'sigmoid')

  def testModelConfigWithContextFeatures(self):
    """Tests store / restore from config with context features."""
    init_kwargs = self._testModelConfig(context_feature_columns=[
        tf.feature_column.numeric_column('context', shape=(1,))
    ])
    self.assertEqual(init_kwargs['context_feature_columns'][0].name, 'context')

  def DISABLED_testSaveModelWeights(self):  # See b/129842600.
    """Tests that model weights can be saved and restored."""
    model = self._get_compiled_model(return_sequences=True)
    model.fit(x=self.x, y=self.y, batch_size=1, steps_per_epoch=1, epochs=1)
    y1 = model.predict(x=self.x, steps=1)
    model.save_weights(self.get_temp_dir() + 'model')

    model = self._get_compiled_model(return_sequences=True, name='model-2')
    model.load_weights(self.get_temp_dir() + 'model')
    y2 = model.predict(x=self.x, steps=1)
    self.assertAllClose(y1, y2)

  def DISABLED_testEvaluationMetrics(self):  # See b/129842600.
    """Tests evaluation metrics computation in non-sequential case."""
    model = self._get_compiled_model()
    metrics = model.evaluate(
        x=self.x, y=ops.convert_to_tensor([[0], [1]]), steps=1)
    # See `RNNClassifierEvaluationTest` for details on computation.
    self.assertAllClose(metrics, (1.1196611, 1.), atol=1e-4)

  def DISABLED_testEvaluationSequential(self):  # See b/129842600.
    """Tests that the sequence mask is properly used to aggregate loss."""
    model = self._get_compiled_model(return_sequences=True)
    metrics = model.evaluate(x=self.x, y=self.y, steps=1)
    # See `RNNClassifierEvaluationTest` for details on computation.
    self.assertAllClose(metrics, (1.9556, 1. / 3.), atol=1e-4)

  def DISABLED_testPredictions(self):  # See b/129842600.
    """Tests predictions with RNN model."""
    model = self._get_compiled_model()
    # See `RNNClassifierPredictionTest` for details on computation.
    self.assertAllClose(
        model.predict(x=self.x, steps=1), [[0.353593], [0.5049296]], atol=1e-4)

  def DISABLED_testPredictionsSequential(self):  # See b/129842600.
    """Tests sequential predictions with RNN model."""
    model = self._get_compiled_model(return_sequences=True)
    # See `RNNClassifierPredictionTest` for details on computation.
    self.assertAllClose(
        model.predict(x=self.x, steps=1),
        [[[0.191731], [0.353593]], [[0.5049296], [0.5049296]]],
        atol=1e-4)

  @parameterized.named_parameters(
      ('StringOptimizer', 'Adam'),
      ('OptimizerInstance', tf.keras.optimizers.Adam()))
  def DISABLED_testTraining(self, optimizer):  # See b/129842600.
    """Tests the loss computed in training step."""
    model = self._get_compiled_model(optimizer=optimizer)
    history = model.fit(
        x=self.x,
        y=ops.convert_to_tensor([[0], [1]]),
        batch_size=1,
        steps_per_epoch=1)
    # See `RNNClassifierTrainingTest` for details on computation.
    self.assertAllClose(history.history['loss'], [1.1196611], atol=1e-4)

  def DISABLED_testTrainingSequential(self):  # See b/129842600.
    """Tests the loss computed in training step in sequential case."""
    model = self._get_compiled_model(return_sequences=True)
    history = model.fit(x=self.x, y=self.y, batch_size=1, steps_per_epoch=1)
    # See `RNNClassifierTrainingTest` for details on computation.
    self.assertAllClose(history.history['loss'], [1.9556], atol=1e-4)


@test_util.run_all_in_graph_and_eager_modes
class RNNEstimatorInitTest(tf.test.TestCase):

  def setUp(self):
    col = tf.feature_column.sequence_categorical_column_with_hash_bucket(
        'tokens', hash_bucket_size=10)
    self.feature_columns = [
        tf.feature_column.embedding_column(col, dimension=2)
    ]
    self.cell_units = [4, 2]
    super(RNNEstimatorInitTest, self).setUp()

  def testConflictingRNNCellFn(self):
    with self.assertRaisesRegexp(
        ValueError,
        'units and cell_type must not be specified when using rnn_cell_fn'):
      rnn.RNNClassifier(
          sequence_feature_columns=self.feature_columns,
          rnn_cell_fn=lambda: 'mock-cell',
          units=self.cell_units)

    with self.assertRaisesRegexp(
        ValueError,
        'units and cell_type must not be specified when using rnn_cell_fn'):
      rnn.RNNClassifier(
          sequence_feature_columns=self.feature_columns,
          rnn_cell_fn=lambda: 'mock-cell',
          cell_type='lstm')

  def testNonSequentialHeadProvided(self):
    with self.assertRaisesRegexp(
        ValueError, 'Provided head must be a `_SequentialHead` object when '
        '`return_sequences` is set to True.'):
      rnn.RNNEstimator(
          head=multi_head_lib.MultiClassHead(n_classes=3),
          sequence_feature_columns=self.feature_columns,
          return_sequences=True)

  def testWrongOptimizerTypeProvided(self):
    classifier = rnn.RNNClassifier(
        self.feature_columns, units=[1], optimizer=object())
    with self.assertRaisesRegexp(
        ValueError,
        'The given object is not a tf.keras.optimizers.Optimizer instance.'):
      classifier.model_fn(
          features=None, labels=None, mode=model_fn.ModeKeys.TRAIN, config=None)


@test_util.run_all_in_graph_and_eager_modes
class RNNClassifierTrainingTest(tf.test.TestCase):

  def setUp(self):
    self.kernel = [[.1, -.2]]
    self.recurrent = [[.2, -.3], [.3, -.4]]
    self.bias = [.2, .5]
    self.dense_kernel = [[-1.], [1.]]
    self.dense_bias = [0.3]
    self.sequence_feature_columns = [
        tf.feature_column.sequence_numeric_column('price', shape=(1,))
    ]
    super(RNNClassifierTrainingTest, self).setUp()

  def _assert_checkpoint(self, n_classes, input_units, cell_units,
                         expected_global_step):

    shapes = {
        name: shape
        for (name, shape) in tf.train.list_variables(self.get_temp_dir())
    }

    self.assertEqual([], shapes[tf.compat.v1.GraphKeys.GLOBAL_STEP])
    self.assertEqual(
        expected_global_step,
        tf.train.load_variable(self.get_temp_dir(),
                               tf.compat.v1.GraphKeys.GLOBAL_STEP))

    # RNN Cell variables.
    for i, cell_unit in enumerate(cell_units):
      name_suffix = '_%d' % i if i else ''
      self.assertEqual([input_units, cell_unit],
                       shapes[CELL_KERNEL_NAME + name_suffix])
      self.assertEqual([cell_unit, cell_unit],
                       shapes[CELL_RECURRENT_KERNEL_NAME + name_suffix])
      self.assertEqual([cell_unit], shapes[CELL_BIAS_NAME + name_suffix])
      input_units = cell_unit

    # Logits variables.
    logits_dimension = n_classes if n_classes > 2 else 1
    self.assertEqual([cell_units[-1], logits_dimension],
                     shapes[LOGITS_WEIGHTS_NAME])
    self.assertEqual([logits_dimension], shapes[LOGITS_BIAS_NAME])

  def _mock_optimizer(self, expected_loss=None):
    var_names = (CELL_BIAS_NAME, CELL_KERNEL_NAME, CELL_RECURRENT_KERNEL_NAME,
                 LOGITS_BIAS_NAME, LOGITS_WEIGHTS_NAME)
    expected_var_names = ['%s:0' % name for name in var_names]

    class _Optimizer(tf.keras.optimizers.Optimizer):
      """Mock optimizer checking that loss has the proper value."""

      def __init__(self, test_case):
        super(_Optimizer, self).__init__(name='my-optimizer')
        self.call_count = 0
        self._test_case = test_case

      def get_updates(self, loss, params):
        self.call_count += 1
        trainable_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        self._test_case.assertItemsEqual(expected_var_names,
                                         [var.name for var in trainable_vars])

        # Verify loss. We can't check the value directly so we add an assert op.
        self._test_case.assertEquals(0, loss.shape.ndims)
        if expected_loss is None:
          return [self.iterations.assign_add(1).op]
        assert_loss = _assert_close(
            tf.cast(expected_loss, name='expected', dtype=tf.dtypes.float32),
            loss,
            name='assert_loss')
        with tf.control_dependencies((assert_loss,)):
          return [self.iterations.assign_add(1).op]

      def get_config(self):
        pass

    return _Optimizer(test_case=self)

  def _testFromScratchWithDefaultOptimizer(self, n_classes):

    def train_input_fn():
      return {
          'tokens':
              tf.sparse.SparseTensor(
                  values=['the', 'cat', 'sat'],
                  indices=[[0, 0], [0, 1], [0, 2]],
                  dense_shape=[1, 3]),
      }, [[1]]

    col = tf.feature_column.sequence_categorical_column_with_hash_bucket(
        'tokens', hash_bucket_size=10)
    embed = tf.feature_column.embedding_column(col, dimension=2)
    input_units = 2

    cell_units = [4, 2]
    est = rnn.RNNClassifier(
        sequence_feature_columns=[embed],
        units=cell_units,
        n_classes=n_classes,
        model_dir=self.get_temp_dir())

    # Train for a few steps, and validate final checkpoint.
    num_steps = 10
    est.train(input_fn=train_input_fn, steps=num_steps)
    self._assert_checkpoint(n_classes, input_units, cell_units, num_steps)

  def testBinaryClassFromScratchWithDefaultOptimizer(self):
    self._testFromScratchWithDefaultOptimizer(n_classes=2)

  def testMultiClassFromScratchWithDefaultOptimizer(self):
    self._testFromScratchWithDefaultOptimizer(n_classes=4)

  def testFromScratchWithCustomRNNCellFn(self):

    def train_input_fn():
      return {
          'tokens':
              tf.sparse.SparseTensor(
                  values=['the', 'cat', 'sat'],
                  indices=[[0, 0], [0, 1], [0, 2]],
                  dense_shape=[1, 3]),
      }, [[1]]

    col = tf.feature_column.sequence_categorical_column_with_hash_bucket(
        'tokens', hash_bucket_size=10)
    embed = tf.feature_column.embedding_column(col, dimension=2)
    input_units = 2
    cell_units = [4, 2]
    n_classes = 2

    def rnn_cell_fn():
      cells = [tf.keras.layers.SimpleRNNCell(units=n) for n in cell_units]
      return tf.keras.layers.StackedRNNCells(cells)

    est = rnn.RNNClassifier(
        sequence_feature_columns=[embed],
        rnn_cell_fn=rnn_cell_fn,
        n_classes=n_classes,
        model_dir=self.get_temp_dir())

    # Train for a few steps, and validate final checkpoint.
    num_steps = 10
    est.train(input_fn=train_input_fn, steps=num_steps)
    self._assert_checkpoint(n_classes, input_units, cell_units, num_steps)

  def _testExampleWeight(self, n_classes):

    def train_input_fn():
      return {
          'tokens':
              tf.sparse.SparseTensor(
                  values=['the', 'cat', 'sat', 'dog', 'barked'],
                  indices=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],
                  dense_shape=[2, 3]),
          'w': [[1], [2]],
      }, [[1], [0]]

    col = tf.feature_column.sequence_categorical_column_with_hash_bucket(
        'tokens', hash_bucket_size=10)
    embed = tf.feature_column.embedding_column(col, dimension=2)
    input_units = 2

    cell_units = [4, 2]
    est = rnn.RNNClassifier(
        units=cell_units,
        sequence_feature_columns=[embed],
        n_classes=n_classes,
        weight_column='w',
        model_dir=self.get_temp_dir())

    # Train for a few steps, and validate final checkpoint.
    num_steps = 10
    est.train(input_fn=train_input_fn, steps=num_steps)
    self._assert_checkpoint(n_classes, input_units, cell_units, num_steps)

  def testBinaryClassWithExampleWeight(self):
    self._testExampleWeight(n_classes=2)

  def testMultiClassWithExampleWeight(self):
    self._testExampleWeight(n_classes=4)

  def _testFromCheckpoint(self, input_fn, expected_loss, **kwargs):
    """Loads classifier from checkpoint, runs training and checks loss."""
    create_checkpoint(
        kernel=self.kernel,
        recurrent=self.recurrent,
        bias=self.bias,
        dense_kernel=self.dense_kernel,
        dense_bias=self.dense_bias,
        global_step=100,
        model_dir=self.get_temp_dir())

    mock_optimizer = self._mock_optimizer(expected_loss=expected_loss)

    est = rnn.RNNClassifier(
        units=[2],
        sequence_feature_columns=self.sequence_feature_columns,
        optimizer=mock_optimizer,
        model_dir=self.get_temp_dir(),
        **kwargs)
    self.assertEqual(0, mock_optimizer.call_count)
    est.train(input_fn=input_fn, steps=10)
    self.assertEqual(1, mock_optimizer.call_count)

  def testBinaryClassFromCheckpoint(self):

    def train_input_fn():
      return {
          'price':
              tf.sparse.SparseTensor(
                  values=[10., 5., 2.],
                  indices=[[0, 0], [0, 1], [1, 0]],
                  dense_shape=[2, 2]),
      }, [[0], [1]]

    # Uses same checkpoint and examples as testBinaryClassEvaluationMetrics.
    # See that test for loss calculation.
    self._testFromCheckpoint(train_input_fn, expected_loss=0.559831)

  def testMultiClassFromCheckpoint(self):

    def train_input_fn():
      return {
          'price':
              tf.sparse.SparseTensor(
                  values=[10., 5., 2., 7.],
                  indices=[[0, 0], [0, 1], [1, 0], [1, 1]],
                  dense_shape=[2, 2]),
      }, [[0], [1]]

    # Uses same checkpoint and examples as testMultiClassEvaluationMetrics.
    # See that test for loss calculation.
    self.dense_kernel = [[-1., 0.5, 0.2], [1., -0.3, 0.1]]
    self.dense_bias = [0.3, 0.4, 0.5]
    self._testFromCheckpoint(
        train_input_fn, expected_loss=1.331465, n_classes=3)

  def testBinaryClassFromCheckpointSequential(self):

    def train_input_fn():
      return {
          'price':
              tf.sparse.SparseTensor(
                  values=[10., 5., 2.],
                  indices=[[0, 0], [0, 1], [1, 0]],
                  dense_shape=[2, 2]),
      }, tf.sparse.SparseTensor(
          values=[0, 1, 0],
          indices=[[0, 0], [0, 1], [1, 0]],
          dense_shape=[2, 2])

    # Same example as testBinaryClassEvaluationMetricsSequential.
    # logits = [[[-1.4388], [-0.6033]],
    #            [[0.0197], [_]]]
    # probability = np.exp(logits) / (1 + np.exp(logits))
    #             = [[0.1917, 0.3536],
    #                [0.5049, _]]
    # loss = -label * ln(p) - (1 - label) * ln(1 - p)
    # loss = [[0.2129,  1.0396],
    #         [0.7031, _]]
    # aggregated_loss = sum(loss) / 3
    # aggregated_loss = 0.6518
    self._testFromCheckpoint(
        train_input_fn, expected_loss=0.651841, return_sequences=True)

  def testBinaryClassFromCheckpointSequentialWithWeights(self):

    def train_input_fn():
      return {
          'price':
              tf.sparse.SparseTensor(
                  values=[10., 5., 2.],
                  indices=[[0, 0], [0, 1], [1, 0]],
                  dense_shape=[2, 2]),
          'weights':
              tf.sparse.SparseTensor(
                  values=[0., 0.5, 0.5],
                  indices=[[0, 0], [0, 1], [1, 0]],
                  dense_shape=[2, 2])
      }, tf.sparse.SparseTensor(
          values=[0, 0, 1],
          indices=[[0, 0], [0, 1], [1, 0]],
          dense_shape=[2, 2])

    # Checkpoint and input are the same as testBinaryClassEvaluationMetrics, and
    # expected loss is the same as we use non-zero weights only for the last
    # step of each sequence.
    # loss = [[_,  0.436326],
    #         [0.6833351, _]]
    # weights = [[0, 0.5], [0.5, 0]]
    # aggregated_loss = (0.436326 + 0.6833351) / 2.
    #                 = 0.559831
    self._testFromCheckpoint(
        train_input_fn,
        expected_loss=0.559831,
        return_sequences=True,
        weight_column='weights',
        loss_reduction=tf.keras.losses.Reduction.SUM)

  def testDefaultGradientClipping(self):
    """Tests that optimizer applies default gradient clipping value."""

    def train_input_fn():
      return {
          'price':
              tf.sparse.SparseTensor(
                  values=[
                      1.,
                  ], indices=[[0, 0]], dense_shape=[1, 1]),
      }, [[1]]

    def _wrap_create_estimator_spec(create_estimator_spec):
      """Wraps function and asserts that the optimizer applies clipping."""

      def _wrapped_create_estimator_spec(obj,
                                         features,
                                         mode,
                                         logits,
                                         labels=None,
                                         optimizer=None,
                                         trainable_variables=None,
                                         train_op_fn=None,
                                         update_ops=None,
                                         regularization_losses=None):
        var = tf.Variable([1.0])
        mock_loss = 10 * var
        gradients = optimizer.get_gradients(mock_loss, [var])
        self.assertLen(gradients, 1)
        # Initial gradient value is 10 and expected to be clipped to 5 (default
        # clipping value).
        with tf.control_dependencies(
            (tf.compat.v1.debugging.assert_equal(gradients[0], 5.0),)):
          return create_estimator_spec(obj, features, mode, logits, labels,
                                       optimizer, trainable_variables,
                                       train_op_fn, update_ops,
                                       regularization_losses)

      return _wrapped_create_estimator_spec

    with tf.compat.v1.test.mock.patch.object(
        multi_head_lib.MultiClassHead, 'create_estimator_spec',
        _wrap_create_estimator_spec(
            multi_head_lib.MultiClassHead.create_estimator_spec)):
      est = rnn.RNNClassifier(
          n_classes=3,
          sequence_feature_columns=[
              tf.feature_column.sequence_numeric_column('price')
          ],
          units=[2],
          model_dir=self.get_temp_dir())
      est.train(input_fn=train_input_fn, steps=1)


def sorted_key_dict(unsorted_dict):
  return {k: unsorted_dict[k] for k in sorted(unsorted_dict)}


@test_util.run_all_in_graph_and_eager_modes
class RNNClassifierEvaluationTest(tf.test.TestCase):

  def setUp(self):
    self.kernel = [[.1, -.2]]
    self.recurrent = [[.2, -.3], [.3, -.4]]
    self.bias = [.2, .5]
    self.dense_kernel = [[-1.], [1.]]
    self.dense_bias = [0.3]
    self.global_step = 100
    self.sequence_feature_columns = [
        tf.feature_column.sequence_numeric_column('price', shape=(1,))
    ]
    super(RNNClassifierEvaluationTest, self).setUp()

  def _testFromCheckpoint(self, input_fn, **kwargs):
    create_checkpoint(
        kernel=self.kernel,
        recurrent=self.recurrent,
        bias=self.bias,
        dense_kernel=self.dense_kernel,
        dense_bias=self.dense_bias,
        global_step=self.global_step,
        model_dir=self.get_temp_dir())

    est = rnn.RNNClassifier(
        units=[2],
        sequence_feature_columns=self.sequence_feature_columns,
        model_dir=self.get_temp_dir(),
        **kwargs)
    return est.evaluate(input_fn, steps=1)

  def testBinaryClassEvaluationMetrics(self):

    def eval_input_fn():
      return {
          'price':
              tf.sparse.SparseTensor(
                  values=[10., 5., 2.],
                  indices=[[0, 0], [0, 1], [1, 0]],
                  dense_shape=[2, 2]),
      }, [[0], [1]]

    eval_metrics = self._testFromCheckpoint(eval_input_fn)

    # Uses identical numbers to testMultiExamplesWithDifferentLength.
    # See that test for logits calculation.
    # logits = [[-0.603282], [0.019719]]
    # probability = exp(logits) / (1 + exp(logits)) = [[0.353593], [0.504930]]
    # loss = -label * ln(p) - (1 - label) * ln(1 - p)
    #      = [[0.436326], [0.683335]]
    # sum_over_batch_size = (0.436326 + 0.683335)/2
    expected_metrics = {
        tf.compat.v1.GraphKeys.GLOBAL_STEP: self.global_step,
        metric_keys.MetricKeys.LOSS: 0.559831,
        metric_keys.MetricKeys.LOSS_MEAN: 0.559831,
        metric_keys.MetricKeys.ACCURACY: 1.0,
        metric_keys.MetricKeys.PREDICTION_MEAN: 0.429262,
        metric_keys.MetricKeys.LABEL_MEAN: 0.5,
        metric_keys.MetricKeys.ACCURACY_BASELINE: 0.5,
        # With default threshold of 0.5, the model is a perfect classifier.
        metric_keys.MetricKeys.RECALL: 1.0,
        metric_keys.MetricKeys.PRECISION: 1.0,
        # Positive example is scored above negative, so AUC = 1.0.
        metric_keys.MetricKeys.AUC: 1.0,
        metric_keys.MetricKeys.AUC_PR: 1.0,
    }
    self.assertAllClose(
        sorted_key_dict(expected_metrics), sorted_key_dict(eval_metrics))

  def testBinaryClassEvaluationMetricsSequential(self):

    def eval_input_fn():
      return {
          'price':
              tf.sparse.SparseTensor(
                  values=[10., 5., 2.],
                  indices=[[0, 0], [0, 1], [1, 0]],
                  dense_shape=[2, 2]),
      }, tf.sparse.SparseTensor(
          values=[0, 1, 0],
          indices=[[0, 0], [0, 1], [1, 0]],
          dense_shape=[2, 2])

    eval_metrics = self._testFromCheckpoint(
        eval_input_fn, return_sequences=True)

    # logits = [[[-1.4388], [-0.6033]],
    #            [[0.0197], [_]]]
    # probability = np.exp(logits) / (1 + np.exp(logits))
    #             = [[0.1917, 0.3536],
    #                [0.5049, _]]
    # labels = [[0, 1],
    #           [0, _]]
    # loss = -label * ln(p) - (1 - label) * ln(1 - p)
    # loss = [[0.2129,  1.0396],
    #         [0.7031, _]]
    # aggregated_loss = sum(loss) / 3
    # aggregated_loss = 0.6518
    # accuracy = 1/3
    # prediction_mean = mean(probability) = 0.3501
    expected_metrics = {
        tf.compat.v1.GraphKeys.GLOBAL_STEP: self.global_step,
        metric_keys.MetricKeys.LOSS: 0.651841,
        metric_keys.MetricKeys.LOSS_MEAN: 0.651841,
        metric_keys.MetricKeys.ACCURACY: 1.0 / 3,
        metric_keys.MetricKeys.PREDICTION_MEAN: 0.350085,
        metric_keys.MetricKeys.LABEL_MEAN: 1.0 / 3,
        metric_keys.MetricKeys.ACCURACY_BASELINE: 2.0 / 3,
        metric_keys.MetricKeys.RECALL: 0.0,
        metric_keys.MetricKeys.PRECISION: 0.0,
        metric_keys.MetricKeys.AUC: 0.5,
        metric_keys.MetricKeys.AUC_PR: 0.30685282,
    }
    self.assertAllClose(
        sorted_key_dict(expected_metrics), sorted_key_dict(eval_metrics))

  def testMultiClassEvaluationMetrics(self):

    def eval_input_fn():
      return {
          'price':
              tf.sparse.SparseTensor(
                  values=[10., 5., 2., 7.],
                  indices=[[0, 0], [0, 1], [1, 0], [1, 1]],
                  dense_shape=[2, 2]),
      }, [[0], [1]]

    self.dense_kernel = [[-1., 0.5, 0.2], [1., -0.3, 0.1]]
    self.dense_bias = [0.3, 0.4, 0.5]
    # Uses identical numbers to testMultiExampleMultiDim.
    # See that test for logits calculation.
    # logits = [[-0.603282, 0.777708, 0.569756],
    #           [-1.247356, 1.017018, 0.574481]]
    # logits_exp = exp(logits) / (1 + exp(logits))
    #            = [[0.547013, 2.176468, 1.767836],
    #               [0.287263, 2.764937, 1.776208]]
    # softmax_probabilities = logits_exp / logits_exp.sum()
    #                       = [[0.121793, 0.484596, 0.393611],
    #                          [0.059494, 0.572639, 0.367866]]
    # loss = -1. * log(softmax[label])
    #      = [[2.105432], [0.557500]]
    # sum_over_batch_size = (2.105432 + 0.557500)/2
    eval_metrics = self._testFromCheckpoint(eval_input_fn, n_classes=3)

    expected_metrics = {
        tf.compat.v1.GraphKeys.GLOBAL_STEP: self.global_step,
        metric_keys.MetricKeys.LOSS: 1.331465,
        metric_keys.MetricKeys.LOSS_MEAN: 1.331466,
        metric_keys.MetricKeys.ACCURACY: 0.5,
    }

    self.assertAllClose(
        sorted_key_dict(expected_metrics), sorted_key_dict(eval_metrics))


@test_util.run_all_in_graph_and_eager_modes
class RNNClassifierPredictionTest(tf.test.TestCase):

  def setUp(self):
    self.kernel = [[.1, -.2]]
    self.recurrent = [[.2, -.3], [.3, -.4]]
    self.bias = [.2, .5]
    self.dense_kernel = [[-1.], [1.]]
    self.dense_bias = [0.3]
    self.sequence_feature_columns = [
        tf.feature_column.sequence_numeric_column('price', shape=(1,))
    ]
    super(RNNClassifierPredictionTest, self).setUp()

  def _testFromCheckpoint(self, input_fn, **kwargs):
    create_checkpoint(
        kernel=self.kernel,
        recurrent=self.recurrent,
        bias=self.bias,
        dense_kernel=self.dense_kernel,
        dense_bias=self.dense_bias,
        global_step=100,
        model_dir=self.get_temp_dir())

    n_classes = 2
    if 'n_classes' in kwargs:
      n_classes = kwargs['n_classes']
      assert n_classes >= 2
    label_vocabulary = [
        'class_{}'.format(class_idx) for class_idx in range(n_classes)
    ]

    est = rnn.RNNClassifier(
        units=[2],
        sequence_feature_columns=self.sequence_feature_columns,
        label_vocabulary=label_vocabulary,
        model_dir=self.get_temp_dir(),
        **kwargs)
    return next(est.predict(input_fn))

  def testBinaryClassPredictions(self):
    # Uses identical numbers to testOneDimLogits.
    # See that test for logits calculation.
    # logits = [-0.603282]
    # logistic = exp(-0.6033) / (1 + exp(-0.6033)) = [0.353593]
    # probabilities = [0.646407, 0.353593]
    # class_ids = argmax(probabilities) = [0]
    predictions = self._testFromCheckpoint(_default_features_fn)
    self.assertAllClose([-0.603282],
                        predictions[prediction_keys.PredictionKeys.LOGITS])
    self.assertAllClose([0.353593],
                        predictions[prediction_keys.PredictionKeys.LOGISTIC])
    self.assertAllClose(
        [0.646407, 0.353593],
        predictions[prediction_keys.PredictionKeys.PROBABILITIES])
    self.assertAllClose([0],
                        predictions[prediction_keys.PredictionKeys.CLASS_IDS])
    self.assertEqual([b'class_0'],
                     predictions[prediction_keys.PredictionKeys.CLASSES])

  def testMultiClassPredictions(self):
    self.dense_kernel = [[-1., 0.5, 0.2], [1., -0.3, 0.1]]
    self.dense_bias = [0.3, 0.4, 0.5]
    # Uses identical numbers to testMultiDimLogits.
    # See that test for logits calculation.
    # logits = [-0.603282, 0.777708, 0.569756]
    # logits_exp = exp(logits) = [0.547013, 2.176468, 1.767836]
    # softmax_probabilities = logits_exp / logits_exp.sum()
    #                       = [0.121793, 0.484596, 0.393611]
    # class_ids = argmax(probabilities) = [1]
    predictions = self._testFromCheckpoint(_default_features_fn, n_classes=3)
    self.assertAllClose([-0.603282, 0.777708, 0.569756],
                        predictions[prediction_keys.PredictionKeys.LOGITS])
    self.assertAllClose(
        [0.121793, 0.484596, 0.393611],
        predictions[prediction_keys.PredictionKeys.PROBABILITIES])
    self.assertAllClose([1],
                        predictions[prediction_keys.PredictionKeys.CLASS_IDS])
    self.assertEqual([b'class_1'],
                     predictions[prediction_keys.PredictionKeys.CLASSES])

  def testBinaryClassPredictionsSequential(self):

    def predict_input_fn():
      return {
          'price':
              tf.sparse.SparseTensor(
                  values=[10., 5.],
                  indices=[[0, 0], [0, 1]],
                  dense_shape=[1, 2]),
      }

    # Same as first record of testBinaryClassEvaluationMetricsSequential.
    # Last step values are carried over.
    # logits = [[-1.4388], [-0.6033], [_]]
    # probabilities = np.exp(logits) / (1 + np.exp(logits))
    #               = [[0.8083, 0.1917], [0.6464, 0.3536], [_, _]]
    # class_ids = [[0], [0], [_]]
    # classes = [['class_0'], ['class_0'], [_]]
    predictions = self._testFromCheckpoint(
        predict_input_fn, return_sequences=True, sequence_mask='my-mask')
    self.assertAllEqual([1, 1], predictions['my-mask'])
    self.assertAllClose([[-1.438803], [-0.603282]],
                        predictions[prediction_keys.PredictionKeys.LOGITS])
    self.assertAllClose([[0.191731], [0.353593]],
                        predictions[prediction_keys.PredictionKeys.LOGISTIC])
    self.assertAllClose(
        [[0.808269, 0.191731], [0.646407, 0.353593]],
        predictions[prediction_keys.PredictionKeys.PROBABILITIES])
    self.assertAllClose([[0], [0]],
                        predictions[prediction_keys.PredictionKeys.CLASS_IDS])
    self.assertAllEqual([[b'class_0'], [b'class_0']],
                        predictions[prediction_keys.PredictionKeys.CLASSES])


class BaseRNNClassificationIntegrationTest(object):

  def setUp(self):
    col = tf.feature_column.sequence_categorical_column_with_hash_bucket(
        'tokens', hash_bucket_size=10)
    embed = tf.feature_column.embedding_column(col, dimension=2)
    self.feature_columns = [embed]
    super(BaseRNNClassificationIntegrationTest, self).setUp()

  def __init__(self, _create_estimator_fn):
    self._create_estimator_fn = _create_estimator_fn

  def _test_complete_flow(self,
                          train_input_fn,
                          eval_input_fn,
                          predict_input_fn,
                          n_classes,
                          batch_size,
                          optimizer='Adam'):
    cell_units = [4, 2]
    est = self._create_estimator_fn(
        self.feature_columns,
        n_classes,
        cell_units,
        self.get_temp_dir(),
        optimizer=optimizer)

    # TRAIN
    num_steps = 10
    est.train(train_input_fn, steps=num_steps)

    # EVALUATE
    scores = est.evaluate(eval_input_fn)
    self.assertEqual(num_steps, scores[tf.compat.v1.GraphKeys.GLOBAL_STEP])
    self.assertIn('loss', six.iterkeys(scores))

    # PREDICT
    predicted_proba = np.array([
        x[prediction_keys.PredictionKeys.PROBABILITIES]
        for x in est.predict(predict_input_fn)
    ])
    self.assertAllEqual((batch_size, n_classes), predicted_proba.shape)

    # EXPORT
    feature_spec = parsing_utils.classifier_parse_example_spec(
        self.feature_columns, label_key='label', label_dtype=tf.dtypes.int64)
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    export_dir = est.export_savedmodel(tempfile.mkdtemp(),
                                       serving_input_receiver_fn)
    self.assertTrue(tf.compat.v1.gfile.Exists(export_dir))

  def _testNumpyInputFn(self, optimizer):
    """Tests complete flow with numpy_input_fn."""
    n_classes = 3
    batch_size = 10
    words = ['dog', 'cat', 'bird', 'the', 'a', 'sat', 'flew', 'slept']
    # Numpy only supports dense input, so all examples will have same length.
    # TODO(b/73160931): Update test when support for prepadded data exists.
    sequence_length = 3

    features = []
    for _ in range(batch_size):
      sentence = random.sample(words, sequence_length)
      features.append(sentence)

    x_data = np.array(features)
    y_data = np.random.randint(n_classes, size=batch_size)

    train_input_fn = numpy_io.numpy_input_fn(
        x={'tokens': x_data},
        y=y_data,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'tokens': x_data}, y=y_data, batch_size=batch_size, shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'tokens': x_data}, batch_size=batch_size, shuffle=False)

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        n_classes=n_classes,
        batch_size=batch_size,
        optimizer=optimizer)

  def testNumpyInputFnStringOptimizer(self):
    self._testNumpyInputFn(optimizer='Adam')

  def testNumpyInputFnOptimizerInstance(self):
    self._testNumpyInputFn(optimizer=tf.keras.optimizers.Adam())

  def testParseExampleInputFn(self):
    """Tests complete flow with input_fn constructed from parse_example."""
    n_classes = 3
    batch_size = 10
    words = [b'dog', b'cat', b'bird', b'the', b'a', b'sat', b'flew', b'slept']

    _, examples_file = tempfile.mkstemp()
    writer = tf.io.TFRecordWriter(examples_file)
    for _ in range(batch_size):
      sequence_length = random.randint(1, len(words))
      sentence = random.sample(words, sequence_length)
      label = random.randint(0, n_classes - 1)
      example = example_pb2.Example(
          features=feature_pb2.Features(
              feature={
                  'tokens':
                      feature_pb2.Feature(
                          bytes_list=feature_pb2.BytesList(value=sentence)),
                  'label':
                      feature_pb2.Feature(
                          int64_list=feature_pb2.Int64List(value=[label])),
              }))
      writer.write(example.SerializeToString())
    writer.close()

    feature_spec = parsing_utils.classifier_parse_example_spec(
        self.feature_columns, label_key='label', label_dtype=tf.dtypes.int64)

    def _train_input_fn():
      dataset = tf.compat.v1.data.experimental.make_batched_features_dataset(
          examples_file, batch_size, feature_spec)
      return dataset.map(lambda features: (features, features.pop('label')))

    def _eval_input_fn():
      dataset = tf.compat.v1.data.experimental.make_batched_features_dataset(
          examples_file, batch_size, feature_spec, num_epochs=1)
      return dataset.map(lambda features: (features, features.pop('label')))

    def _predict_input_fn():
      dataset = tf.compat.v1.data.experimental.make_batched_features_dataset(
          examples_file, batch_size, feature_spec, num_epochs=1)

      def features_fn(features):
        features.pop('label')
        return features

      return dataset.map(features_fn)

    self._test_complete_flow(
        train_input_fn=_train_input_fn,
        eval_input_fn=_eval_input_fn,
        predict_input_fn=_predict_input_fn,
        n_classes=n_classes,
        batch_size=batch_size)


def _rnn_classifier_fn(feature_columns, n_classes, cell_units, model_dir,
                       optimizer):
  return rnn.RNNClassifier(
      units=cell_units,
      sequence_feature_columns=feature_columns,
      n_classes=n_classes,
      optimizer=optimizer,
      model_dir=model_dir)


@test_util.run_all_in_graph_and_eager_modes
class RNNClassifierIntegrationTest(BaseRNNClassificationIntegrationTest,
                                   tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    BaseRNNClassificationIntegrationTest.__init__(self, _rnn_classifier_fn)


def _rnn_classifier_dropout_fn(feature_columns, n_classes, cell_units,
                               model_dir, optimizer):

  def _rnn_cell_fn():
    cells = []
    for units in cell_units:
      cells.append(tf.keras.layers.SimpleRNNCell(units, dropout=0.5))
    return tf.keras.layers.StackedRNNCells(cells)

  return rnn.RNNClassifier(
      rnn_cell_fn=_rnn_cell_fn,
      sequence_feature_columns=feature_columns,
      n_classes=n_classes,
      optimizer=optimizer,
      model_dir=model_dir)


@test_util.run_all_in_graph_and_eager_modes
class RNNClassifierDropoutIntegrationTest(BaseRNNClassificationIntegrationTest,
                                          tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    BaseRNNClassificationIntegrationTest.__init__(self,
                                                  _rnn_classifier_dropout_fn)


def _rnn_estimator_fn(feature_columns, n_classes, cell_units, model_dir,
                      optimizer):
  return rnn.RNNEstimator(
      head=multi_head_lib.MultiClassHead(n_classes=n_classes),
      units=cell_units,
      sequence_feature_columns=feature_columns,
      optimizer=optimizer,
      model_dir=model_dir)


@test_util.run_all_in_graph_and_eager_modes
class RNNEstimatorIntegrationTest(BaseRNNClassificationIntegrationTest,
                                  tf.test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    tf.test.TestCase.__init__(self, methodName)
    BaseRNNClassificationIntegrationTest.__init__(self, _rnn_estimator_fn)


@test_util.run_all_in_graph_and_eager_modes
class ModelFnTest(tf.test.TestCase):
  """Tests correctness of RNNEstimator's model function."""

  def _test_sequential_mask_in_head(self, mask=None):
    features = {
        'price':
            tf.sparse.SparseTensor(
                values=[10., 5., 4.],
                indices=[[0, 0], [0, 1], [1, 0]],
                dense_shape=[2, 2])
    }
    if mask:
      features['sequence_mask'] = ops.convert_to_tensor(mask)
    expected_mask = mask or [[1, 1], [1, 0]]

    sequence_feature_columns = [
        tf.feature_column.sequence_numeric_column('price', shape=(1,))
    ]

    mock_head = _get_mock_head()
    seq_head = seq_head_lib.SequentialHeadWrapper(
        mock_head, sequence_length_mask='sequence_mask')
    estimator = rnn.RNNEstimator(
        head=seq_head,
        units=[10],
        sequence_feature_columns=sequence_feature_columns,
        return_sequences=True)
    estimator.model_fn(
        features=features,
        labels=None,
        mode=model_fn.ModeKeys.PREDICT,
        config=None)
    passed_features = list(
        mock_head.create_estimator_spec.call_args)[1]['features']
    self.assertIn('sequence_mask', passed_features)
    sequence_mask = self.evaluate(passed_features['sequence_mask'])
    self.assertAllEqual(sequence_mask, expected_mask)

  def testSequentialMaskInHead(self):
    self._test_sequential_mask_in_head()

  def testSequentialMaskInHeadWithMasks(self):
    self._test_sequential_mask_in_head([[1, 1], [1, 1]])


if __name__ == '__main__':
  tf.test.main()
