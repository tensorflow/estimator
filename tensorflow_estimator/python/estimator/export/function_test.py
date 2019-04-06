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
"""Tests for Estimator function objects."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import six as six

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.training import training
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.export import function
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys


def _string_fix(obj):
  return nest.map_structure(
      lambda x: compat.as_bytes(x) if isinstance(x, six.string_types) else x,
      obj)


def _model_fn(features, labels, mode):
  v = variables.Variable(constant_op.constant(23), name='v')
  if mode == ModeKeys.PREDICT:
    return model_fn_lib.EstimatorSpec(
        ModeKeys.PREDICT,
        predictions=features + 1)
  elif mode == ModeKeys.EVAL:
    return model_fn_lib.EstimatorSpec(
        ModeKeys.EVAL,
        loss=constant_op.constant(5) + v,
        predictions=features + labels)
  elif mode == ModeKeys.TRAIN:
    return model_fn_lib.EstimatorSpec(
        ModeKeys.TRAIN,
        predictions=features * labels,
        loss=constant_op.constant(5) + v,
        train_op=state_ops.assign_add(training.get_global_step(), 1))


def _model_fn_train_only(features, labels):
  v = variables.Variable(constant_op.constant(23), name='v')
  return model_fn_lib.EstimatorSpec(
      ModeKeys.TRAIN,
      predictions=features * labels,
      loss=constant_op.constant(5) + v,
      train_op=state_ops.assign_add(training.get_global_step(), 1))


def _model_fn_predict_only(features):
  return model_fn_lib.EstimatorSpec(
      ModeKeys.PREDICT,
      predictions=features + 1)


# TODO(kathywu): Re-enable test after def_function changes are built into
# nightlies.
@test_util.run_all_in_graph_and_eager_modes
class ModelFunctionTest(object):

  def test_from_function(self):
    mfn = function.ModelFunction.from_function(_model_fn)
    out = mfn.train(constant_op.constant(3), constant_op.constant(5))

    self.evaluate(variables.variables_initializer(mfn.variables.values()))

    self.assertEqual(15, self.evaluate(out['predictions']))
    out = mfn.evaluate(constant_op.constant(7), constant_op.constant(9))
    self.assertEqual(16, self.evaluate(out['predictions']))
    out = mfn.predict(constant_op.constant(10))
    self.assertEqual(11, self.evaluate(out['predictions']))

  def test_model_fn_train_only(self):
    mfn = function.ModelFunction()
    mfn.add_mode(_model_fn_train_only, ModeKeys.TRAIN)
    out = mfn.train(constant_op.constant(4), constant_op.constant(6))

    self.evaluate(variables.variables_initializer(mfn.variables.values()))

    self.assertEqual(24, self.evaluate(out['predictions']))

    with self.assertRaisesRegexp(ValueError, 'not defined'):
      out = mfn.evaluate(constant_op.constant(7), constant_op.constant(9))

  def test_model_fn_predict_only(self):
    mfn = function.ModelFunction()
    mfn.add_mode(_model_fn_predict_only, ModeKeys.PREDICT)
    out = mfn.predict(constant_op.constant(4))

    self.evaluate(variables.variables_initializer(mfn.variables.values()))

    self.assertEqual(5, self.evaluate(out['predictions']))

    with self.assertRaisesRegexp(ValueError, 'not defined'):
      out = mfn.evaluate(constant_op.constant(7), constant_op.constant(9))

  def test_save_and_load(self):
    mfn = function.ModelFunction.from_function(_model_fn)

    out = mfn.train(constant_op.constant(3), constant_op.constant(5))
    self.evaluate(variables.variables_initializer(mfn.variables.values()))
    self.evaluate(out['predictions'])

    for _ in range(2):
      out = mfn.train(constant_op.constant(3), constant_op.constant(5))
      self.evaluate(out['predictions'])
    self.assertEqual(
        3, self.evaluate(mfn._variable_holder.variables['global_step']))

    mfn.evaluate(constant_op.constant(7), constant_op.constant(9))
    mfn.predict(constant_op.constant(10))

    save_dir = os.path.join(self.get_temp_dir(), 'model_function')
    save.save(mfn, save_dir)

    obj = load.load(save_dir)
    variables_by_name = obj._variables_by_name

    self.evaluate(variables.variables_initializer(
        variables_by_name._unconditional_dependency_names.values()))
    self.assertEqual(3, self.evaluate(variables_by_name.global_step))

    out = obj._functions['train'](constant_op.constant(3),
                                  constant_op.constant(5))
    self.assertEqual(15, self.evaluate(out['predictions']))
    self.assertEqual(4, self.evaluate(variables_by_name.global_step))

    out = obj._functions['eval'](constant_op.constant(7),
                                 constant_op.constant(9))
    self.assertEqual(16, self.evaluate(out['predictions']))

    out = obj._functions['infer'](constant_op.constant(10))
    self.assertEqual(11, self.evaluate(out['predictions']))


def _model_fn_callable_variable_initializers(features, labels, mode):
  """Model_fn with callable variable initializers (for WrappedGraph tests)."""
  _ = features, labels
  v = variables.Variable(lambda: constant_op.constant(23), name='v')
  if mode == ModeKeys.PREDICT:
    return model_fn_lib.EstimatorSpec(
        ModeKeys.PREDICT,
        predictions=features + 1)
  elif mode == ModeKeys.EVAL:
    return model_fn_lib.EstimatorSpec(
        ModeKeys.EVAL,
        loss=constant_op.constant(5) + v,
        predictions=features + labels)
  elif mode == ModeKeys.TRAIN:
    return model_fn_lib.EstimatorSpec(
        ModeKeys.TRAIN,
        predictions=features * labels,
        loss=constant_op.constant(5) + v,
        train_op=state_ops.assign_add(training.get_global_step(), 1))


@test_util.run_all_in_graph_and_eager_modes
class EstimatorWrappedGraphTest(test.TestCase):

  def test_wrap_model_fn_train(self):
    graph = function._EstimatorWrappedGraph()
    features = constant_op.constant(3)
    labels = constant_op.constant(4)
    mode = ModeKeys.TRAIN
    fn = graph.wrap_model_fn(
        _model_fn_callable_variable_initializers,
        mode=mode, args=[features, labels, mode], kwargs={})
    self.evaluate(variables.variables_initializer(graph.variables.values()))
    self.assertEqual(0, self.evaluate(graph.global_step))
    self.assertEqual(12, self.evaluate(fn(features, labels)['predictions']))
    self.assertEqual(1, self.evaluate(graph.global_step))

    self.assertEqual('AssignAddVariableOp',
                     graph.estimator_spec.train_op.type)

  def test_wrap_model_fn_eval(self):
    graph = function._EstimatorWrappedGraph()
    features = constant_op.constant(5)
    labels = constant_op.constant(6)
    mode = ModeKeys.EVAL
    fn = graph.wrap_model_fn(
        _model_fn_callable_variable_initializers,
        mode=mode, args=[features, labels, mode], kwargs={})
    self.assertDictEqual({'predictions': 11},
                         self.evaluate(fn(features, labels)))

  def test_wrap_model_fn_predict(self):
    graph = function._EstimatorWrappedGraph()
    features = constant_op.constant(7)
    mode = ModeKeys.PREDICT
    fn = graph.wrap_model_fn(
        _model_fn_callable_variable_initializers,
        mode=mode, args=[features, None, mode], kwargs={})
    self.assertDictEqual({'predictions': 8},
                         self.evaluate(fn(features)))

  def test_wrap_input_receiver_fn(self):

    def serving_input_fn():
      receiver_1 = array_ops.placeholder(dtypes.string)
      receiver_2 = array_ops.placeholder(dtypes.string)

      receiver_tensors = {
          'rec1': receiver_1,
          u'rec2': receiver_2,
      }

      concat = string_ops.string_join([receiver_1, receiver_2])
      concat2 = array_ops.identity(concat)
      features = {
          'feature0': string_ops.string_join([concat, concat2], ':'),
          u'feature1': constant_op.constant([1])
      }

      alternate_tensors = {
          'alt_name_1': concat,
          'alt_name_2': {
              'tensor1': concat,
              'tensor2': concat2}
      }
      return export_lib.ServingInputReceiver(
          features, receiver_tensors, alternate_tensors)

    graph = function._EstimatorWrappedGraph()
    fns = graph.wrap_input_receiver_fn(serving_input_fn)

    for fn, name in fns:
      if name is None:
        out = fn(constant_op.constant('1'), constant_op.constant('2'))
        self.assertDictEqual(
            _string_fix({'feature0': '12:12', 'feature1': [1]}),
            _string_fix(self.evaluate(out)))
      elif name == 'alt_name_1':
        out = fn(constant_op.constant('3'))
        self.assertDictEqual(
            _string_fix({'feature0': '3:3', 'feature1': [1]}),
            _string_fix(self.evaluate(out)))
      elif name == 'alt_name_2':
        out = fn(constant_op.constant('4'), constant_op.constant('5'))
        self.assertDictEqual(
            _string_fix({'feature0': '4:5', 'feature1': [1]}),
            _string_fix(self.evaluate(out)))


if __name__ == '__main__':
  test.main()
