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

from tensorflow.contrib.feature_column.python.feature_column import sequence_feature_column as seq_fc
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.feature_column import feature_column_lib as fc
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import monitored_session
from tensorflow.python.training import optimizer as optimizer_lib
from tensorflow.python.training import training_util
from tensorflow_estimator.contrib.estimator.python.estimator import rnn_v2 as rnn
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import parsing_utils
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.export import export
from tensorflow_estimator.python.estimator.head import multi_class_head as multi_head_lib
from tensorflow_estimator.python.estimator.head import sequential_head as seq_head_lib
from tensorflow_estimator.python.estimator.inputs import numpy_io

# Names of variables created by BasicRNNCell model.
CELL_KERNEL_NAME = 'rnn/kernel'
CELL_RECURRENT_KERNEL_NAME = 'rnn/recurrent_kernel'
CELL_BIAS_NAME = 'rnn/bias'
LOGITS_WEIGHTS_NAME = 'logits/kernel'
LOGITS_BIAS_NAME = 'logits/bias'


def _assert_close(expected, actual, rtol=1e-04, name='assert_close'):
  with ops.name_scope(name, 'assert_close', (expected, actual, rtol)) as scope:
    expected = ops.convert_to_tensor(expected, name='expected')
    actual = ops.convert_to_tensor(actual, name='actual')
    rdiff = math_ops.abs(expected - actual, 'diff') / math_ops.abs(expected)
    rtol = ops.convert_to_tensor(rtol, name='rtol')
    return check_ops.assert_less(
        rdiff,
        rtol,
        data=('Condition expected =~ actual did not hold element-wise:'
              'expected = ', expected, 'actual = ', actual, 'rdiff = ', rdiff,
              'rtol = ', rtol,),
        name=scope)


def create_checkpoint(kernel, recurrent_kernel, rnn_biases, logits_weights,
                      logits_biases, global_step, model_dir):
  """Create checkpoint file with provided model weights.

  Args:
    kernel: Iterable of values of input weights for the RNN cell.
    recurrent_kernel: Iterable of values of recurrent weights for the RNN cell.
    rnn_biases: Iterable of values of biases for the RNN cell.
    logits_weights: Iterable of values for matrix connecting RNN output to
      logits.
    logits_biases: Iterable of values for logits bias term.
    global_step: Initial global step to save in checkpoint.
    model_dir: Directory into which checkpoint is saved.
  """
  model_weights = {}
  model_weights[CELL_KERNEL_NAME] = kernel
  model_weights[CELL_RECURRENT_KERNEL_NAME] = recurrent_kernel
  model_weights[CELL_BIAS_NAME] = rnn_biases
  model_weights[LOGITS_WEIGHTS_NAME] = logits_weights
  model_weights[LOGITS_BIAS_NAME] = logits_biases

  with ops.Graph().as_default():
    # Create model variables.
    for k, v in six.iteritems(model_weights):
      variables_lib.Variable(v, name=k, dtype=dtypes.float32)

    # Create non-model variables.
    global_step_var = training_util.create_global_step()
    assign_op = global_step_var.assign(global_step)

    # Initialize vars and save checkpoint.
    with monitored_session.MonitoredTrainingSession(
        checkpoint_dir=model_dir) as sess:
      sess.run(assign_op)


def _make_rnn_layer(rnn_cell_fn=None, units=None, cell_type=rnn.USE_DEFAULT,
                    return_sequences=False):
  layer_fn = rnn._make_rnn_layer_fn(
      rnn_cell_fn=rnn_cell_fn, units=units, cell_type=cell_type,
      return_sequences=return_sequences)
  return layer_fn()


@test_util.run_all_in_graph_and_eager_modes
class RNNLayerFnTest(test.TestCase, parameterized.TestCase):
  """Tests for rnn layer function."""

  def testWrongClassProvided(self):
    """Tests that an error is raised if the class doesn't have a call method."""
    with self.assertRaisesRegexp(
        ValueError,
        'RNN cell should have a `call` and `state_size` method.'):
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
    self.assertIsInstance(layer, keras_layers.RNN)
    self.assertIsInstance(layer.cell, keras_layers.SimpleRNNCell)

  @parameterized.parameters([('gru', keras_layers.UnifiedGRU),
                             ('lstm', keras_layers.UnifiedLSTM)])
  def testSpecificLayerTypeProvided(self, cell_type, layer_type):
    """Tests specific layer type for GRU and LSTM."""
    layer = _make_rnn_layer(cell_type=cell_type, units=[1])
    self.assertIsInstance(layer, layer_type)

  def testSpecificLayerTypeArguments(self):
    """Tests arguments for specific layer types (GRU and LSTM)."""
    mock_layer_type = test.mock.Mock()
    with test.mock.patch.object(
        rnn, '_LAYER_TYPES', {'custom-type': mock_layer_type}):
      _make_rnn_layer(
          cell_type='custom-type', units='units-value',
          return_sequences='return-seq-value')
      mock_layer_type.assert_called_once_with(
          units='units-value', return_sequences='return-seq-value')

  @test.mock.patch.object(keras_layers, 'RNN')
  def testCustomCellProvided(self, mock_rnn_layer_type):
    """Tests behavior when a custom cell type is provided."""
    mock_custom_cell = test.mock.Mock()
    _make_rnn_layer(
        units=[10], cell_type=lambda units: mock_custom_cell,
        return_sequences='return-seq-value')
    mock_rnn_layer_type.assert_called_once_with(
        cell=mock_custom_cell, return_sequences='return-seq-value')

  def testMultipleCellsProvided(self):
    """Tests behavior when multiple cells are provided."""
    layer = _make_rnn_layer(cell_type='simple_rnn', units=[1, 2])
    self.assertIsInstance(layer, keras_layers.RNN)
    self.assertIsInstance(layer.cell, keras_layers.StackedRNNCells)
    self.assertLen(layer.cell.cells, 2)
    self.assertIsInstance(layer.cell.cells[0], keras_layers.SimpleRNNCell)

  @test.mock.patch.object(keras_layers, 'RNN')
  def testCustomCellFnProvided(self, mock_rnn_layer_type):
    """Tests behavior when a custom cell function is provided."""
    mock_cell_fn = test.mock.Mock(return_value='custom-cell')
    _make_rnn_layer(
        rnn_cell_fn=mock_cell_fn, return_sequences='return-seq-value')
    mock_rnn_layer_type.assert_called_once_with(
        cell='custom-cell', return_sequences='return-seq-value')


def _mock_rnn_cell(kernel, recurrent, bias):
  """Sets initialization values to `SimpleRNNCell` layers used in context."""

  class _MockRNNCell(keras_layers.SimpleRNNCell):

    def __init__(self, units):
      super(_MockRNNCell, self).__init__(
          units=units,
          kernel_initializer=init_ops.Constant(kernel),
          recurrent_initializer=init_ops.Constant(recurrent),
          bias_initializer=init_ops.Constant(bias))

  return test.mock.patch.object(keras_layers, 'SimpleRNNCell', _MockRNNCell)


def _mock_logits_layer(kernel, bias):
  """Sets initialization values to dense `logits` layers used in context."""

  class _MockDenseLayer(keras_layers.Dense):

    def __init__(self, units, name):
      kwargs = {}
      if name == 'logits':
        kwargs = {
            'kernel_initializer': init_ops.Constant(kernel),
            'bias_initializer': init_ops.Constant(bias)}

      super(_MockDenseLayer, self).__init__(
          units=units, name=name, **kwargs)

  return test.mock.patch.object(keras_layers, 'Dense', _MockDenseLayer)


@test_util.run_all_in_graph_and_eager_modes
class RNNLogitFnTest(test.TestCase, parameterized.TestCase):
  """Tests correctness of logits calculated from _rnn_logit_fn_builder."""

  def setUp(self):
    # Sets layers default weights for testing purpose.
    self.kernel = [[.1, -.2]]
    self.recurrent = [[.2, -.3], [.3, -.4]]
    self.bias = [.2, .5]
    self.dense_kernel = [[-1.], [1.]]
    self.dense_bias = [0.3]
    super(RNNLogitFnTest, self).setUp()

  def _mock_rnn_cell(self):
    return _mock_rnn_cell(self.kernel, recurrent=self.recurrent, bias=self.bias)

  def _mock_logits_layer(self):
    return _mock_logits_layer(self.dense_kernel, bias=self.dense_bias)

  def _test_logits(self, mode, rnn_units, logits_dimension, features_fn,
                   sequence_feature_columns, context_feature_columns,
                   expected_logits, return_sequences=False):
    """Tests that the expected logits are calculated."""
    logit_fn = rnn._rnn_logit_fn_builder(
        output_units=logits_dimension,
        rnn_layer_fn=rnn._make_rnn_layer_fn(
            units=rnn_units, cell_type=rnn.USE_DEFAULT,
            return_sequences=return_sequences, rnn_cell_fn=None),
        sequence_feature_columns=sequence_feature_columns,
        context_feature_columns=context_feature_columns)
    # Features are constructed within this function, otherwise the Tensors
    # containing the features would be defined outside this graph.
    logits = logit_fn(features=features_fn(), mode=mode)
    self.evaluate(variables_lib.global_variables_initializer())
    self.assertAllClose(expected_logits, self.evaluate(logits), atol=1e-4)

  @parameterized.named_parameters(
      {'testcase_name': 'Static',
       'return_sequences': False,
       'expected_logits': [[-0.6033]]},
      {'testcase_name': 'Sequential',
       'return_sequences': True,
       'expected_logits': [[[-1.4388], [-0.6033]]]})
  def testOneDimLogits(self, return_sequences, expected_logits):
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
    """
    expected_mask = [[1, 1]]

    def features_fn():
      return {
          'price':
              sparse_tensor.SparseTensor(
                  values=[10., 5.],
                  indices=[[0, 0], [0, 1]],
                  dense_shape=[1, 2]),
      }

    sequence_feature_columns = [
        seq_fc.sequence_numeric_column('price', shape=(1,))]
    context_feature_columns = []

    with self._mock_rnn_cell():
      with self._mock_logits_layer():
        for mode in [
            model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
            model_fn.ModeKeys.PREDICT
        ]:
          self._test_logits(
              mode,
              rnn_units=[2],
              logits_dimension=1,
              features_fn=features_fn,
              sequence_feature_columns=sequence_feature_columns,
              context_feature_columns=context_feature_columns,
              expected_logits=(expected_logits, expected_mask),
              return_sequences=return_sequences)

  @parameterized.named_parameters(
      {'testcase_name': 'Static',
       'return_sequences': False,
       'expected_logits': [[-0.6033, 0.7777, 0.5698]]},
      {'testcase_name': 'Sequential',
       'return_sequences': True,
       'expected_logits': [[
           [-1.4388, 1.0884, 0.5762],
           [-0.6033, 0.7777, 0.5698]]]})
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

    def features_fn():
      return {
          'price':
              sparse_tensor.SparseTensor(
                  values=[10., 5.],
                  indices=[[0, 0], [0, 1]],
                  dense_shape=[1, 2]),
      }

    sequence_feature_columns = [
        seq_fc.sequence_numeric_column('price', shape=(1,))]
    context_feature_columns = []

    self.dense_kernel = [[-1., 0.5, 0.2], [1., -0.3, 0.1]]
    self.dense_bias = [0.3, 0.4, 0.5]
    with self._mock_rnn_cell():
      with self._mock_logits_layer():
        for mode in [
            model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
            model_fn.ModeKeys.PREDICT
        ]:
          self._test_logits(
              mode,
              rnn_units=[2],
              logits_dimension=3,
              features_fn=features_fn,
              sequence_feature_columns=sequence_feature_columns,
              context_feature_columns=context_feature_columns,
              expected_logits=(expected_logits, expected_mask),
              return_sequences=return_sequences)

  @parameterized.named_parameters(
      {'testcase_name': 'Static',
       'return_sequences': False,
       'expected_logits': [[-0.6033, 0.7777, 0.5698],
                           [-1.2473, 1.0170, 0.5745]]},
      {'testcase_name': 'Sequential',
       'return_sequences': True,
       'expected_logits': [[[-1.4388, 1.0884, 0.5762],
                            [-0.6033, 0.7777, 0.5698]],
                           [[0.0197, 0.5601, 0.5860],
                            [-1.2473, 1.0170, 0.5745]]]})
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
              sparse_tensor.SparseTensor(
                  values=[10., 5., 2., 7.],
                  indices=[[0, 0], [0, 1], [1, 0], [1, 1]],
                  dense_shape=[2, 2]),
      }

    sequence_feature_columns = [
        seq_fc.sequence_numeric_column('price', shape=(1,))
    ]
    context_feature_columns = []

    self.dense_kernel = [[-1., 0.5, 0.2], [1., -0.3, 0.1]]
    self.dense_bias = [0.3, 0.4, 0.5]
    with self._mock_rnn_cell():
      with self._mock_logits_layer():
        for mode in [
            model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
            model_fn.ModeKeys.PREDICT
        ]:
          self._test_logits(
              mode,
              rnn_units=[2],
              logits_dimension=3,
              features_fn=features_fn,
              sequence_feature_columns=sequence_feature_columns,
              context_feature_columns=context_feature_columns,
              expected_logits=(expected_logits, expected_mask),
              return_sequences=return_sequences)

  @parameterized.named_parameters(
      {'testcase_name': 'Static',
       'return_sequences': False,
       'expected_logits': [[-0.6033], [0.0197]]},
      {'testcase_name': 'Sequential',
       'return_sequences': True,
       'expected_logits': [[[-1.4388], [-0.6033]],
                           [[0.0197], [0.0197]]]})
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
              sparse_tensor.SparseTensor(
                  values=[10., 5., 2.],
                  indices=[[0, 0], [0, 1], [1, 0]],
                  dense_shape=[2, 2]),
      }

    sequence_feature_columns = [
        seq_fc.sequence_numeric_column('price', shape=(1,))]
    context_feature_columns = []

    with self._mock_rnn_cell():
      with self._mock_logits_layer():
        for mode in [
            model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
            model_fn.ModeKeys.PREDICT
        ]:
          self._test_logits(
              mode,
              rnn_units=[2],
              logits_dimension=1,
              features_fn=features_fn,
              sequence_feature_columns=sequence_feature_columns,
              context_feature_columns=context_feature_columns,
              expected_logits=(expected_logits, expected_mask),
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
              sparse_tensor.SparseTensor(
                  values=[10., 5., 2.],
                  indices=[[0, 0], [0, 1], [1, 0]],
                  dense_shape=[2, 2]),
          'context': [[-0.5], [0.8]],
      }

    sequence_feature_columns = [
        seq_fc.sequence_numeric_column('price', shape=(1,))]
    context_feature_columns = [fc.numeric_column('context', shape=(1,))]

    self.kernel = [[.1, -.2], [1., 0.9]]
    with self._mock_rnn_cell():
      with self._mock_logits_layer():
        for mode in [
            model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
            model_fn.ModeKeys.PREDICT
        ]:
          self._test_logits(
              mode,
              rnn_units=[2],
              logits_dimension=1,
              features_fn=features_fn,
              sequence_feature_columns=sequence_feature_columns,
              context_feature_columns=context_feature_columns,
              expected_logits=([[-0.3662], [0.1414]], expected_mask))

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
              sparse_tensor.SparseTensor(
                  values=[10., 5., 2.],
                  indices=[[0, 0], [0, 1], [1, 0]],
                  dense_shape=[2, 2]),
          'on_sale':
              sparse_tensor.SparseTensor(
                  values=[0, 1, 0],
                  indices=[[0, 0], [0, 1], [1, 0]],
                  dense_shape=[2, 2]),
      }

    price_column = seq_fc.sequence_numeric_column('price', shape=(1,))
    on_sale_column = fc.indicator_column(
        seq_fc.sequence_categorical_column_with_identity(
            'on_sale', num_buckets=2))
    sequence_feature_columns = [price_column, on_sale_column]
    context_feature_columns = []

    self.kernel = [[.5, -.5], [1., -1.], [.1, -.2]]
    with self._mock_rnn_cell():
      with self._mock_logits_layer():
        for mode in [
            model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
            model_fn.ModeKeys.PREDICT
        ]:
          self._test_logits(
              mode,
              rnn_units=[2],
              logits_dimension=1,
              features_fn=features_fn,
              sequence_feature_columns=sequence_feature_columns,
              context_feature_columns=context_feature_columns,
              expected_logits=([[-1.5056], [-0.7962]], expected_mask))

  @parameterized.parameters([
      (model_fn.ModeKeys.TRAIN, True),
      (model_fn.ModeKeys.EVAL, False),
      (model_fn.ModeKeys.PREDICT, False)])
  def testTrainingMode(self, mode, expected_training_mode):
    """Tests that `training` argument is properly used."""
    sequence_feature_columns = [
        seq_fc.sequence_numeric_column('price', shape=(1,))]

    class _MockRNNCell(keras_layers.SimpleRNNCell):
      """Used to test that `training` argument is properly used."""

      def __init__(self, test_case):
        self._test_case = test_case
        super(_MockRNNCell, self).__init__(units=10)

      def call(self, inputs, states, training=None):
        self._test_case.assertEqual(training, expected_training_mode)
        return super(_MockRNNCell, self).call(
            inputs=inputs, states=states, training=training)

    logit_fn = rnn._rnn_logit_fn_builder(
        output_units=1,
        rnn_layer_fn=rnn._make_rnn_layer_fn(
            rnn_cell_fn=lambda: _MockRNNCell(self),
            units=None,
            cell_type=rnn.USE_DEFAULT,
            return_sequences=False),
        sequence_feature_columns=sequence_feature_columns,
        context_feature_columns=None)
    features = {
        'price':
            sparse_tensor.SparseTensor(
                values=[10.,],
                indices=[[0, 0]],
                dense_shape=[1, 1]),
    }
    logit_fn(features=features, mode=mode)


@test_util.run_all_in_graph_and_eager_modes
class RNNClassifierTrainingTest(test.TestCase):

  def _assert_checkpoint(
      self, n_classes, input_units, cell_units, expected_global_step):

    shapes = {
        name: shape for (name, shape) in
        checkpoint_utils.list_variables(self.get_temp_dir())
    }

    self.assertEqual([], shapes[ops.GraphKeys.GLOBAL_STEP])
    self.assertEqual(
        expected_global_step,
        checkpoint_utils.load_variable(
            self.get_temp_dir(), ops.GraphKeys.GLOBAL_STEP))

    # RNN Cell variables.
    for i, cell_unit in enumerate(cell_units):
      name_suffix = '_%d' % i if i else ''
      self.assertEqual([input_units, cell_unit],
                       shapes[CELL_KERNEL_NAME + name_suffix])
      self.assertEqual([cell_unit, cell_unit],
                       shapes[CELL_RECURRENT_KERNEL_NAME + name_suffix])
      self.assertEqual([cell_unit],
                       shapes[CELL_BIAS_NAME + name_suffix])
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

    def _minimize(loss, global_step):
      trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertItemsEqual(
          expected_var_names,
          [var.name for var in trainable_vars])

      # Verify loss. We can't check the value directly, so we add an assert op.
      self.assertEquals(0, loss.shape.ndims)
      if expected_loss is None:
        return state_ops.assign_add(global_step, 1).op
      assert_loss = _assert_close(
          math_ops.to_float(expected_loss, name='expected'),
          loss,
          name='assert_loss')
      with ops.control_dependencies((assert_loss,)):
        return state_ops.assign_add(global_step, 1).op

    mock_optimizer = test.mock.NonCallableMock(
        spec=optimizer_lib.Optimizer,
        wraps=optimizer_lib.Optimizer(use_locking=False, name='my_optimizer'))
    mock_optimizer.minimize = test.mock.MagicMock(wraps=_minimize)

    # NOTE: Estimator.params performs a deepcopy, which wreaks havoc with mocks.
    # So, return mock_optimizer itself for deepcopy.
    mock_optimizer.__deepcopy__ = lambda _: mock_optimizer
    return mock_optimizer

  def testConflictingRNNCellFn(self):
    col = seq_fc.sequence_categorical_column_with_hash_bucket(
        'tokens', hash_bucket_size=10)
    embed = fc.embedding_column(col, dimension=2)
    cell_units = [4, 2]

    with self.assertRaisesRegexp(
        ValueError,
        'units and cell_type must not be specified when using rnn_cell_fn'):
      rnn.RNNClassifier(
          sequence_feature_columns=[embed],
          rnn_cell_fn=lambda: 'mock-cell',
          units=cell_units)

    with self.assertRaisesRegexp(
        ValueError,
        'units and cell_type must not be specified when using rnn_cell_fn'):
      rnn.RNNClassifier(
          sequence_feature_columns=[embed],
          rnn_cell_fn=lambda: 'mock-cell',
          cell_type='lstm')

    with self.assertRaisesRegexp(
        ValueError,
        'Provided head must be a `_SequentialHead` object when '
        '`return_sequences` is set to True.'):
      rnn.RNNEstimator(
          head=multi_head_lib.MultiClassHead(n_classes=3),
          sequence_feature_columns=[embed],
          return_sequences=True)

  def _testFromScratchWithDefaultOptimizer(self, n_classes):
    def train_input_fn():
      return {
          'tokens':
              sparse_tensor.SparseTensor(
                  values=['the', 'cat', 'sat'],
                  indices=[[0, 0], [0, 1], [0, 2]],
                  dense_shape=[1, 3]),
      }, [[1]]

    col = seq_fc.sequence_categorical_column_with_hash_bucket(
        'tokens', hash_bucket_size=10)
    embed = fc.embedding_column(col, dimension=2)
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
              sparse_tensor.SparseTensor(
                  values=['the', 'cat', 'sat'],
                  indices=[[0, 0], [0, 1], [0, 2]],
                  dense_shape=[1, 3]),
      }, [[1]]

    col = seq_fc.sequence_categorical_column_with_hash_bucket(
        'tokens', hash_bucket_size=10)
    embed = fc.embedding_column(col, dimension=2)
    input_units = 2
    cell_units = [4, 2]
    n_classes = 2

    def rnn_cell_fn():
      cells = [keras_layers.SimpleRNNCell(units=n) for n in cell_units]
      return keras_layers.StackedRNNCells(cells)

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
              sparse_tensor.SparseTensor(
                  values=['the', 'cat', 'sat', 'dog', 'barked'],
                  indices=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],
                  dense_shape=[2, 3]),
          'w': [[1], [2]],
      }, [[1], [0]]

    col = seq_fc.sequence_categorical_column_with_hash_bucket(
        'tokens', hash_bucket_size=10)
    embed = fc.embedding_column(col, dimension=2)
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

  def testBinaryClassFromCheckpoint(self):
    initial_global_step = 100
    create_checkpoint(
        kernel=[[.1, -.2]],
        recurrent_kernel=[[.2, -.3], [.3, -.4]],
        rnn_biases=[.2, .5],
        logits_weights=[[-1.], [1.]],
        logits_biases=[0.3],
        global_step=initial_global_step,
        model_dir=self.get_temp_dir())

    def train_input_fn():
      return {
          'price':
              sparse_tensor.SparseTensor(
                  values=[10., 5., 2.],
                  indices=[[0, 0], [0, 1], [1, 0]],
                  dense_shape=[2, 2]),
      }, [[0], [1]]

    # Uses same checkpoint and examples as testBinaryClassEvaluationMetrics.
    # See that test for loss calculation.
    mock_optimizer = self._mock_optimizer(expected_loss=0.559831)

    sequence_feature_columns = [
        seq_fc.sequence_numeric_column('price', shape=(1,))]
    est = rnn.RNNClassifier(
        units=[2],
        sequence_feature_columns=sequence_feature_columns,
        n_classes=2,
        optimizer=mock_optimizer,
        model_dir=self.get_temp_dir())
    self.assertEqual(0, mock_optimizer.minimize.call_count)
    est.train(input_fn=train_input_fn, steps=10)
    self.assertEqual(1, mock_optimizer.minimize.call_count)

  def testMultiClassFromCheckpoint(self):
    initial_global_step = 100
    create_checkpoint(
        kernel=[[.1, -.2]],
        recurrent_kernel=[[.2, -.3], [.3, -.4]],
        rnn_biases=[.2, .5],
        logits_weights=[[-1., 0.5, 0.2], [1., -0.3, 0.1]],
        logits_biases=[0.3, 0.4, 0.5],
        global_step=initial_global_step,
        model_dir=self.get_temp_dir())

    def train_input_fn():
      return {
          'price':
              sparse_tensor.SparseTensor(
                  values=[10., 5., 2., 7.],
                  indices=[[0, 0], [0, 1], [1, 0], [1, 1]],
                  dense_shape=[2, 2]),
      }, [[0], [1]]

    # Uses same checkpoint and examples as testMultiClassEvaluationMetrics.
    # See that test for loss calculation.
    mock_optimizer = self._mock_optimizer(expected_loss=1.331465)

    sequence_feature_columns = [
        seq_fc.sequence_numeric_column('price', shape=(1,))]
    est = rnn.RNNClassifier(
        units=[2],
        sequence_feature_columns=sequence_feature_columns,
        n_classes=3,
        optimizer=mock_optimizer,
        model_dir=self.get_temp_dir())
    self.assertEqual(0, mock_optimizer.minimize.call_count)
    est.train(input_fn=train_input_fn, steps=10)
    self.assertEqual(1, mock_optimizer.minimize.call_count)


def sorted_key_dict(unsorted_dict):
  return {k: unsorted_dict[k] for k in sorted(unsorted_dict)}


@test_util.run_all_in_graph_and_eager_modes
class RNNClassifierEvaluationTest(test.TestCase):

  def testBinaryClassEvaluationMetrics(self):
    global_step = 100
    create_checkpoint(
        kernel=[[.1, -.2]],
        recurrent_kernel=[[.2, -.3], [.3, -.4]],
        rnn_biases=[.2, .5],
        logits_weights=[[-1.], [1.]],
        logits_biases=[0.3],
        global_step=global_step,
        model_dir=self.get_temp_dir())

    def eval_input_fn():
      return {
          'price':
              sparse_tensor.SparseTensor(
                  values=[10., 5., 2.],
                  indices=[[0, 0], [0, 1], [1, 0]],
                  dense_shape=[2, 2]),
      }, [[0], [1]]

    sequence_feature_columns = [
        seq_fc.sequence_numeric_column('price', shape=(1,))]

    est = rnn.RNNClassifier(
        units=[2],
        sequence_feature_columns=sequence_feature_columns,
        n_classes=2,
        model_dir=self.get_temp_dir())
    eval_metrics = est.evaluate(eval_input_fn, steps=1)

    # Uses identical numbers to testMultiExamplesWithDifferentLength.
    # See that test for logits calculation.
    # logits = [[-0.603282], [0.019719]]
    # probability = exp(logits) / (1 + exp(logits)) = [[0.353593], [0.504930]]
    # loss = -label * ln(p) - (1 - label) * ln(1 - p)
    #      = [[0.436326], [0.683335]]
    # sum_over_batch_size = (0.436326 + 0.683335)/2
    expected_metrics = {
        ops.GraphKeys.GLOBAL_STEP:
            global_step,
        metric_keys.MetricKeys.LOSS:
            0.559831,
        metric_keys.MetricKeys.LOSS_MEAN:
            0.559831,
        metric_keys.MetricKeys.ACCURACY:
            1.0,
        metric_keys.MetricKeys.PREDICTION_MEAN:
            0.429262,
        metric_keys.MetricKeys.LABEL_MEAN:
            0.5,
        metric_keys.MetricKeys.ACCURACY_BASELINE:
            0.5,
        # With default threshold of 0.5, the model is a perfect classifier.
        metric_keys.MetricKeys.RECALL:
            1.0,
        metric_keys.MetricKeys.PRECISION:
            1.0,
        # Positive example is scored above negative, so AUC = 1.0.
        metric_keys.MetricKeys.AUC:
            1.0,
        metric_keys.MetricKeys.AUC_PR:
            1.0,
    }
    self.assertAllClose(
        sorted_key_dict(expected_metrics), sorted_key_dict(eval_metrics))

  def testMultiClassEvaluationMetrics(self):
    global_step = 100
    create_checkpoint(
        kernel=[[.1, -.2]],
        recurrent_kernel=[[.2, -.3], [.3, -.4]],
        rnn_biases=[.2, .5],
        logits_weights=[[-1., 0.5, 0.2], [1., -0.3, 0.1]],
        logits_biases=[0.3, 0.4, 0.5],
        global_step=global_step,
        model_dir=self.get_temp_dir())

    def eval_input_fn():
      return {
          'price':
              sparse_tensor.SparseTensor(
                  values=[10., 5., 2., 7.],
                  indices=[[0, 0], [0, 1], [1, 0], [1, 1]],
                  dense_shape=[2, 2]),
      }, [[0], [1]]

    sequence_feature_columns = [
        seq_fc.sequence_numeric_column('price', shape=(1,))]

    est = rnn.RNNClassifier(
        units=[2],
        sequence_feature_columns=sequence_feature_columns,
        n_classes=3,
        model_dir=self.get_temp_dir())
    eval_metrics = est.evaluate(eval_input_fn, steps=1)

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
    expected_metrics = {
        ops.GraphKeys.GLOBAL_STEP: global_step,
        metric_keys.MetricKeys.LOSS: 1.331465,
        metric_keys.MetricKeys.LOSS_MEAN: 1.331466,
        metric_keys.MetricKeys.ACCURACY: 0.5,
    }

    self.assertAllClose(
        sorted_key_dict(expected_metrics), sorted_key_dict(eval_metrics))


@test_util.run_all_in_graph_and_eager_modes
class RNNClassifierPredictionTest(test.TestCase):

  def testBinaryClassPredictions(self):
    create_checkpoint(
        kernel=[[.1, -.2]],
        recurrent_kernel=[[.2, -.3], [.3, -.4]],
        rnn_biases=[.2, .5],
        logits_weights=[[-1.], [1.]],
        logits_biases=[0.3],
        global_step=0,
        model_dir=self.get_temp_dir())

    def predict_input_fn():
      return {
          'price':
              sparse_tensor.SparseTensor(
                  values=[10., 5.],
                  indices=[[0, 0], [0, 1]],
                  dense_shape=[1, 2]),
      }

    sequence_feature_columns = [
        seq_fc.sequence_numeric_column('price', shape=(1,))]
    label_vocabulary = ['class_0', 'class_1']

    est = rnn.RNNClassifier(
        units=[2],
        sequence_feature_columns=sequence_feature_columns,
        n_classes=2,
        label_vocabulary=label_vocabulary,
        model_dir=self.get_temp_dir())
    # Uses identical numbers to testOneDimLogits.
    # See that test for logits calculation.
    # logits = [-0.603282]
    # logistic = exp(-0.6033) / (1 + exp(-0.6033)) = [0.353593]
    # probabilities = [0.646407, 0.353593]
    # class_ids = argmax(probabilities) = [0]
    predictions = next(est.predict(predict_input_fn))
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
    create_checkpoint(
        kernel=[[.1, -.2]],
        recurrent_kernel=[[.2, -.3], [.3, -.4]],
        rnn_biases=[.2, .5],
        logits_weights=[[-1., 0.5, 0.2], [1., -0.3, 0.1]],
        logits_biases=[0.3, 0.4, 0.5],
        global_step=0,
        model_dir=self.get_temp_dir())

    def predict_input_fn():
      return {
          'price':
              sparse_tensor.SparseTensor(
                  values=[10., 5.],
                  indices=[[0, 0], [0, 1]],
                  dense_shape=[1, 2]),
      }

    sequence_feature_columns = [
        seq_fc.sequence_numeric_column('price', shape=(1,))]
    label_vocabulary = ['class_0', 'class_1', 'class_2']

    est = rnn.RNNClassifier(
        units=[2],
        sequence_feature_columns=sequence_feature_columns,
        n_classes=3,
        label_vocabulary=label_vocabulary,
        model_dir=self.get_temp_dir())
    # Uses identical numbers to testMultiDimLogits.
    # See that test for logits calculation.
    # logits = [-0.603282, 0.777708, 0.569756]
    # logits_exp = exp(logits) = [0.547013, 2.176468, 1.767836]
    # softmax_probabilities = logits_exp / logits_exp.sum()
    #                       = [0.121793, 0.484596, 0.393611]
    # class_ids = argmax(probabilities) = [1]
    predictions = next(est.predict(predict_input_fn))
    self.assertAllClose([-0.603282, 0.777708, 0.569756],
                        predictions[prediction_keys.PredictionKeys.LOGITS])
    self.assertAllClose(
        [0.121793, 0.484596, 0.393611],
        predictions[prediction_keys.PredictionKeys.PROBABILITIES])
    self.assertAllClose([1],
                        predictions[prediction_keys.PredictionKeys.CLASS_IDS])
    self.assertEqual([b'class_1'],
                     predictions[prediction_keys.PredictionKeys.CLASSES])


class BaseRNNClassificationIntegrationTest(object):

  def __init__(self, _create_estimator_fn):
    self._create_estimator_fn = _create_estimator_fn

  def _test_complete_flow(self, feature_columns, train_input_fn, eval_input_fn,
                          predict_input_fn, n_classes, batch_size):
    cell_units = [4, 2]
    est = self._create_estimator_fn(feature_columns, n_classes, cell_units,
                                    self.get_temp_dir())

    # TRAIN
    num_steps = 10
    est.train(train_input_fn, steps=num_steps)

    # EVALUATE
    scores = est.evaluate(eval_input_fn)
    self.assertEqual(num_steps, scores[ops.GraphKeys.GLOBAL_STEP])
    self.assertIn('loss', six.iterkeys(scores))

    # PREDICT
    predicted_proba = np.array([
        x[prediction_keys.PredictionKeys.PROBABILITIES]
        for x in est.predict(predict_input_fn)
    ])
    self.assertAllEqual((batch_size, n_classes), predicted_proba.shape)

    # EXPORT
    feature_spec = parsing_utils.classifier_parse_example_spec(
        feature_columns,
        label_key='label',
        label_dtype=dtypes.int64)
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    export_dir = est.export_savedmodel(tempfile.mkdtemp(),
                                       serving_input_receiver_fn)
    self.assertTrue(gfile.Exists(export_dir))

  def testNumpyInputFn(self):
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
        x={'tokens': x_data},
        y=y_data,
        batch_size=batch_size,
        shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'tokens': x_data},
        batch_size=batch_size,
        shuffle=False)

    col = seq_fc.sequence_categorical_column_with_hash_bucket(
        'tokens', hash_bucket_size=10)
    embed = fc.embedding_column(col, dimension=2)
    feature_columns = [embed]

    self._test_complete_flow(
        feature_columns=feature_columns,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        n_classes=n_classes,
        batch_size=batch_size)

  def testParseExampleInputFn(self):
    """Tests complete flow with input_fn constructed from parse_example."""
    n_classes = 3
    batch_size = 10
    words = [b'dog', b'cat', b'bird', b'the', b'a', b'sat', b'flew', b'slept']

    _, examples_file = tempfile.mkstemp()
    writer = python_io.TFRecordWriter(examples_file)
    for _ in range(batch_size):
      sequence_length = random.randint(1, len(words))
      sentence = random.sample(words, sequence_length)
      label = random.randint(0, n_classes - 1)
      example = example_pb2.Example(features=feature_pb2.Features(
          feature={
              'tokens':
                  feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                      value=sentence)),
              'label':
                  feature_pb2.Feature(int64_list=feature_pb2.Int64List(
                      value=[label])),
          }))
      writer.write(example.SerializeToString())
    writer.close()

    col = seq_fc.sequence_categorical_column_with_hash_bucket(
        'tokens', hash_bucket_size=10)
    embed = fc.embedding_column(col, dimension=2)
    feature_columns = [embed]
    feature_spec = parsing_utils.classifier_parse_example_spec(
        feature_columns,
        label_key='label',
        label_dtype=dtypes.int64)

    def _train_input_fn():
      dataset = readers.make_batched_features_dataset(
          examples_file, batch_size, feature_spec)
      return dataset.map(lambda features: (features, features.pop('label')))
    def _eval_input_fn():
      dataset = readers.make_batched_features_dataset(
          examples_file, batch_size, feature_spec, num_epochs=1)
      return dataset.map(lambda features: (features, features.pop('label')))
    def _predict_input_fn():
      dataset = readers.make_batched_features_dataset(
          examples_file, batch_size, feature_spec, num_epochs=1)
      def features_fn(features):
        features.pop('label')
        return features
      return dataset.map(features_fn)

    self._test_complete_flow(
        feature_columns=feature_columns,
        train_input_fn=_train_input_fn,
        eval_input_fn=_eval_input_fn,
        predict_input_fn=_predict_input_fn,
        n_classes=n_classes,
        batch_size=batch_size)


def _rnn_classifier_fn(feature_columns, n_classes, cell_units, model_dir):
  return rnn.RNNClassifier(
      units=cell_units,
      sequence_feature_columns=feature_columns,
      n_classes=n_classes,
      model_dir=model_dir)


@test_util.run_all_in_graph_and_eager_modes
class RNNClassifierIntegrationTest(BaseRNNClassificationIntegrationTest,
                                   test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    BaseRNNClassificationIntegrationTest.__init__(self, _rnn_classifier_fn)


def _rnn_classifier_dropout_fn(
    feature_columns, n_classes, cell_units, model_dir):
  def _rnn_cell_fn():
    cells = []
    for units in cell_units:
      cells.append(keras_layers.SimpleRNNCell(units, dropout=0.5))
    return keras_layers.StackedRNNCells(cells)

  return rnn.RNNClassifier(
      rnn_cell_fn=_rnn_cell_fn,
      sequence_feature_columns=feature_columns,
      n_classes=n_classes,
      model_dir=model_dir)


@test_util.run_all_in_graph_and_eager_modes
class RNNClassifierDropoutIntegrationTest(BaseRNNClassificationIntegrationTest,
                                          test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    BaseRNNClassificationIntegrationTest.__init__(
        self, _rnn_classifier_dropout_fn)


def _rnn_estimator_fn(feature_columns, n_classes, cell_units, model_dir):
  return rnn.RNNEstimator(
      head=multi_head_lib.MultiClassHead(n_classes=n_classes),
      units=cell_units,
      sequence_feature_columns=feature_columns,
      model_dir=model_dir)


@test_util.run_all_in_graph_and_eager_modes
class RNNEstimatorIntegrationTest(BaseRNNClassificationIntegrationTest,
                                  test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    BaseRNNClassificationIntegrationTest.__init__(self, _rnn_estimator_fn)


class _MockSeqHead(seq_head_lib._SequentialHead,
                   multi_head_lib.MultiClassHead):
  """Used to test that the sequence mask is properly passed to the head."""

  @property
  def input_sequence_mask_key(self,):
    return 'sequence_mask'

  def create_estimator_spec(
      self, features, mode, logits, labels=None, optimizer=None,
      train_op_fn=None, regularization_losses=None):
    return features


@test_util.run_all_in_graph_and_eager_modes
class ModelFnTest(test.TestCase):
  """Tests correctness of RNNEstimator's model function."""

  def _test_sequential_mask_in_head(self, mask=None):
    features = {
        'price': sparse_tensor.SparseTensor(
            values=[10., 5., 4.],
            indices=[[0, 0], [0, 1], [1, 0]],
            dense_shape=[2, 2])}
    if mask:
      features['sequence_mask'] = ops.convert_to_tensor(mask)
    expected_mask = mask or [[1, 1], [1, 0]]

    sequence_feature_columns = [
        seq_fc.sequence_numeric_column('price', shape=(1,))]

    passed_features = rnn._rnn_model_fn(
        features=features,
        labels=None,
        mode=model_fn.ModeKeys.PREDICT,
        head=_MockSeqHead(n_classes=3),
        rnn_layer_fn=rnn._make_rnn_layer_fn(
            rnn_cell_fn=None, units=[10], cell_type=rnn._SIMPLE_RNN_KEY,
            return_sequences=False),
        sequence_feature_columns=sequence_feature_columns,
        context_feature_columns=[],
        return_sequences=True)
    self.assertIn('sequence_mask', passed_features)
    sequence_mask = self.evaluate(passed_features['sequence_mask'])
    self.assertAllEqual(sequence_mask, expected_mask)

  def testSequentialMaskInHead(self):
    self._test_sequential_mask_in_head()

  def testSequentialMaskInHeadWithMasks(self):
    self._test_sequential_mask_in_head([[1, 1], [1, 1]])

if __name__ == '__main__':
  test.main()
