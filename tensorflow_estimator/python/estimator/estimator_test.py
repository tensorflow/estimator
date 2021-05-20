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
"""Tests for Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import glob
import json
import os
import socket
import tempfile

import numpy as np
import six
import tensorflow.compat.v1 as tf
from google.protobuf import text_format

from absl.testing import parameterized
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import combinations
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.random_ops import random_uniform
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.platform import gfile
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.training import checkpoint_state_pb2
from tensorflow.python.training import saver_test_utils
from tensorflow.python.training import training
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import training as estimator_training
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys

_TMP_DIR = '/tmp'
_ANOTHER_TMP_DIR = '/another_tmp'


def dummy_model_fn(features, labels, params):
  _, _, _ = features, labels, params


def summaries_with_matching_keyword(keyword, dir_):
  """Yields summary protos matching given keyword from event file."""

  tf.summary.FileWriterCache.clear()

  event_paths = glob.glob(os.path.join(dir_, 'events*'))
  for event in tf.train.summary_iterator(event_paths[-1]):
    if event.summary is not None:
      for value in event.summary.value:
        if keyword in value.tag:
          yield event.summary


def check_eventfile_for_keyword(keyword, dir_):
  """Checks event files for the keyword."""
  return any(summaries_with_matching_keyword(keyword, dir_))


def get_mock_saver():
  real_saver = tf.train.Saver()
  return tf.test.mock.Mock(wraps=real_saver, saver_def=real_saver.saver_def)


class EstimatorInheritanceConstraintTest(tf.test.TestCase):
  """Tests that sub classes cannot override methods of Estimator."""

  @property
  def random_estimator(self):
    switch = np.random.random()
    return estimator.EstimatorV2 if switch > 0.5 else estimator.EstimatorV2

  def test_override_a_method(self):

    class _Estimator(self.random_estimator):

      def __init__(self):
        super(_Estimator, self).__init__(model_fn=dummy_model_fn)

      def predict(self, input_fn, predict_keys=None, hooks=None):
        pass

    with self.assertRaisesRegexp(
        ValueError, 'cannot override members of Estimator.*predict'):
      _Estimator()

  def test_extension_of_api_is_ok(self):

    class _Estimator(self.random_estimator):

      def __init__(self):
        super(_Estimator, self).__init__(model_fn=dummy_model_fn)

      def predict_proba(self, input_fn, predict_keys=None, hooks=None):
        pass

    _Estimator()

  def test_override_allowed_method(self):

    class _Estimator(self.random_estimator):

      def __init__(self):
        super(_Estimator, self).__init__(model_fn=dummy_model_fn)

      def _tf_api_names(self):
        pass

    _Estimator()


class EstimatorConstructorTest(tf.test.TestCase):

  def test_config_must_be_a_run_config(self):
    with self.assertRaisesRegexp(ValueError, 'an instance of `RunConfig`'):
      estimator.EstimatorV2(model_fn=None, config='NotARunConfig')

  def test_model_fn_must_be_provided(self):
    with self.assertRaisesRegexp(ValueError, 'model_fn.* must be'):
      estimator.EstimatorV2(model_fn=None)

  def test_property_accessors(self):

    def model_fn(features, labels, params):
      _, _, _ = features, labels, params

    class FakeConfig(run_config.RunConfig):
      pass

    params = {'hidden_layers': [3, 4]}
    est = estimator.EstimatorV2(
        model_fn=model_fn, model_dir='bla', config=FakeConfig(), params=params)
    self.assertTrue(isinstance(est.config, FakeConfig))
    self.assertEqual(params, est.params)
    self.assertEqual('bla', est.model_dir)

  def test_default_config(self):

    def model_fn(features, labels):
      _, _ = features, labels

    est = estimator.EstimatorV2(model_fn=model_fn)
    self.assertTrue(isinstance(est.config, run_config.RunConfig))
    self.assertTrue(est._session_config.allow_soft_placement)
    rewrite_options = est._session_config.graph_options.rewrite_options
    self.assertEqual(rewrite_options.meta_optimizer_iterations,
                     rewriter_config_pb2.RewriterConfig.ONE)

  def test_default_model_dir(self):

    def model_fn(features, labels):
      _, _ = features, labels

    with tf.test.mock.patch.object(tempfile, 'mkdtemp', return_value=_TMP_DIR):
      est = estimator.EstimatorV2(model_fn=model_fn)
      self.assertEqual(_TMP_DIR, est.config.model_dir)
      self.assertEqual(_TMP_DIR, est.model_dir)

  def test_model_dir_in_constructor(self):

    def model_fn(features, labels):
      _, _ = features, labels

    est = estimator.EstimatorV2(model_fn=model_fn, model_dir=_TMP_DIR)
    self.assertEqual(_TMP_DIR, est.config.model_dir)
    self.assertEqual(_TMP_DIR, est.model_dir)

  def test_empty_model_dir(self):

    def model_fn(features, labels):
      _, _ = features, labels

    with tf.test.mock.patch.object(tempfile, 'mkdtemp', return_value=_TMP_DIR):
      est = estimator.EstimatorV2(model_fn=model_fn, model_dir='')
      self.assertEqual(_TMP_DIR, est.config.model_dir)
      self.assertEqual(_TMP_DIR, est.model_dir)

  def test_model_dir_in_run_config(self):

    class FakeConfig(run_config.RunConfig):

      @property
      def model_dir(self):
        return _TMP_DIR

    def model_fn(features, labels):
      _, _ = features, labels

    est = estimator.EstimatorV2(model_fn=model_fn, config=FakeConfig())
    self.assertEqual(_TMP_DIR, est.config.model_dir)
    self.assertEqual(_TMP_DIR, est.model_dir)

  def test_same_model_dir_in_constructor_and_run_config(self):

    class FakeConfig(run_config.RunConfig):

      @property
      def model_dir(self):
        return _TMP_DIR

    def model_fn(features, labels):
      _, _ = features, labels

    est = estimator.EstimatorV2(
        model_fn=model_fn, config=FakeConfig(), model_dir=_TMP_DIR)
    self.assertEqual(_TMP_DIR, est.config.model_dir)
    self.assertEqual(_TMP_DIR, est.model_dir)

  def test_different_model_dir_in_constructor_and_run_config(self):

    class FakeConfig(run_config.RunConfig):

      @property
      def model_dir(self):
        return _TMP_DIR

    def model_fn(features, labels):
      _, _ = features, labels

    with self.assertRaisesRegexp(
        ValueError,
        '`model_dir` are set both in constructor and `RunConfig`, but '
        'with different values'):
      estimator.EstimatorV2(
          model_fn=model_fn, config=FakeConfig(), model_dir=_ANOTHER_TMP_DIR)

  def test_model_fn_args_must_include_features(self):

    def model_fn(x, labels):
      _, _ = x, labels

    with self.assertRaisesRegexp(ValueError, 'features'):
      estimator.EstimatorV2(model_fn=model_fn)

  def test_model_fn_args_labels_is_optional(self):

    def model_fn(features):
      _ = features

    estimator.EstimatorV2(model_fn=model_fn)

  def test_if_params_provided_then_model_fn_should_accept_it(self):

    def model_fn(features, labels):
      _, _ = features, labels

    estimator.EstimatorV2(model_fn=model_fn)
    with self.assertRaisesRegexp(ValueError, 'params'):
      estimator.EstimatorV2(model_fn=model_fn, params={'hidden_layers': 4})

  def test_internal_params_is_a_deepcopy(self):

    def model_fn(features, labels, params):
      _, _, _ = features, labels, params

    params = {'hidden_layers': 4}
    est = estimator.EstimatorV2(model_fn=model_fn, params=params)

    params['hidden_layers'] = 5
    self.assertEqual(4, est.params['hidden_layers'])

  def test_not_known_model_fn_args(self):

    def model_fn(features, labels, something):
      _, _, _ = features, labels, something

    with self.assertRaisesRegexp(ValueError, 'something'):
      estimator.EstimatorV2(model_fn=model_fn)

  def test_not_known_model_fn_args_handled_by_lambda(self):

    def model_fn(features, labels, something):
      _, _, _ = features, labels, something

    new_model_fn = lambda features, labels: model_fn(  # pylint: disable=g-long-lambda
        features, labels, 'something')
    estimator.EstimatorV2(model_fn=new_model_fn)

  def test_if_model_fn_is_a_member_function_of_a_class(self):

    class ModelFnClass(object):

      def __init__(self):
        estimator.EstimatorV2(model_fn=self.model_fn)

      def model_fn(self, features, labels, mode):
        _, _, _ = features, labels, mode

    ModelFnClass()

  def test_model_fn_property_binds_params(self):

    def model_fn(features, labels, mode, config, params):
      _, _, _, _, _ = features, labels, mode, config, params

    est = estimator.EstimatorV2(model_fn=model_fn)
    model_fn_args = function_utils.fn_args(est.model_fn)
    self.assertEqual(
        set(['features', 'labels', 'mode', 'config']), set(model_fn_args))

  def test_model_fn_property_returns_fixed_signature(self):

    def model_fn(features, labels):
      _, _ = features, labels

    est = estimator.EstimatorV2(model_fn=model_fn)
    model_fn_args = function_utils.fn_args(est.model_fn)
    self.assertEqual(
        set(['features', 'labels', 'mode', 'config']), set(model_fn_args))


def dummy_input_fn():
  return ({'x': tf.constant([[1], [1]])}, tf.constant([[1], [1]]))


def model_fn_global_step_incrementer(features, labels, mode):
  _, _ = features, labels
  global_step = tf.train.get_global_step()
  return model_fn_lib.EstimatorSpec(
      mode, loss=tf.constant(1.), train_op=tf.assign_add(global_step, 1))


def assert_features_op(expected_features, actual_features):
  return [
      tf.debugging.assert_equal(
          expected_features[k], actual_features[k], name='assert_%s' % k)
      for k in expected_features
  ]


def _estimator_spec(expected_features, expected_labels, actual_features,
                    actual_labels, mode):
  assert_ops = tuple(
      assert_features_op(expected_features, actual_features) + [
          tf.debugging.assert_equal(
              expected_labels, actual_labels, name='assert_labels')
      ])
  global_step = tf.train.get_global_step()
  with tf.control_dependencies(assert_ops):
    return model_fn_lib.EstimatorSpec(
        mode=mode,
        predictions=tf.constant(0.),
        loss=tf.constant(0.),
        train_op=tf.assign_add(global_step, 1))


def _make_input_fn(features, labels):

  def _input_fn():
    return {k: tf.constant(v) for k, v in six.iteritems(features)
           }, tf.constant(labels)

  return _input_fn


class EstimatorTrainTest(tf.test.TestCase):

  def test_callable_model_fn(self):
    expected_features = {'x': 42., 'y': 43.}
    expected_labels = 44.

    model_fn_call_count = [0]

    test_self = self

    class ModelFn(object):

      def __call__(self, features, labels):
        model_fn_call_count[0] += 1
        test_self.assertItemsEqual(expected_features.keys(), features.keys())
        return _estimator_spec(expected_features, expected_labels, features,
                               labels, ModeKeys.TRAIN)

    with self.assertRaisesRegexp(ValueError, 'does not include params'):
      estimator.EstimatorV2(model_fn=ModelFn(), params={'a': 'b'})
    est = estimator.EstimatorV2(
        model_fn=ModelFn(), config=run_config.RunConfig())
    self.assertEqual(0, model_fn_call_count[0])
    est.train(
        input_fn=_make_input_fn(expected_features, expected_labels), steps=1)
    self.assertEqual(1, model_fn_call_count[0])

  def test_callable_input_fn(self):
    expected_mode = ModeKeys.TRAIN
    expected_params = {'batch_size': 10}
    expected_config = run_config.RunConfig().replace(tf_random_seed=4321)
    input_fn_call_count = [0]

    def _model_fn(features, labels, mode, params, config):
      del params, config
      return model_fn_global_step_incrementer(features, labels, mode)

    test_self = self

    class InputFn(object):

      def __call__(self, mode, params, config):
        input_fn_call_count[0] += 1
        test_self.assertEqual(expected_mode, mode)
        test_self.assertEqual(expected_params, params)
        test_self.assertEqual(4321, config.tf_random_seed)
        return dummy_input_fn()

    est = estimator.EstimatorV2(
        model_fn=_model_fn, params=expected_params, config=expected_config)
    self.assertEqual(0, input_fn_call_count[0])
    est.train(InputFn(), steps=1)
    self.assertEqual(1, input_fn_call_count[0])

  def test_nested_input_fn(self):
    expected_params = {'batch_size': 10}

    def _input_fn():
      dataset_features = tf.data.Dataset.from_tensor_slices(
          (random_uniform([4]),
           random_uniform([4, 100], maxval=100, dtype=tf.dtypes.int32)))
      dataset_labels = tf.data.Dataset.from_tensor_slices(
          random_uniform([4, 10]))
      dataset = tf.data.Dataset.zip((dataset_features, dataset_labels))
      dataset = dataset.repeat(-1)
      iterator = tf.data.make_initializable_iterator(dataset)
      return iterator.get_next()

    def _model_fn(features, labels, mode, params, config):
      del params, config
      return model_fn_global_step_incrementer(features, labels, mode)

    expected_config = run_config.RunConfig().replace(tf_random_seed=4321)
    est = estimator.EstimatorV2(
        model_fn=_model_fn, params=expected_params, config=expected_config)
    est.train(_input_fn, steps=4)

  def test_input_fn_args(self):
    expected_mode = ModeKeys.TRAIN
    expected_params = {'batch_size': 10}
    expected_config = run_config.RunConfig().replace(tf_random_seed=4321)
    input_fn_call_count = [0]

    def _model_fn(features, labels, mode, params, config):
      del params, config
      return model_fn_global_step_incrementer(features, labels, mode)

    def _input_fn(mode, params, config):
      input_fn_call_count[0] += 1
      self.assertEqual(expected_mode, mode)
      self.assertEqual(expected_params, params)
      self.assertEqual(4321, config.tf_random_seed)
      return dummy_input_fn()

    est = estimator.EstimatorV2(
        model_fn=_model_fn, params=expected_params, config=expected_config)
    self.assertEqual(0, input_fn_call_count[0])
    est.train(_input_fn, steps=1)
    self.assertEqual(1, input_fn_call_count[0])

  def test_minimal_model_fn_args(self):
    expected_features = {'x': 4, 'y': 5}

    def _input_fn():
      return expected_features

    model_fn_call_count = [0]

    def _model_fn(features):
      model_fn_call_count[0] += 1
      self.assertItemsEqual(expected_features.keys(), features.keys())
      with tf.control_dependencies(
          assert_features_op(expected_features, features)):
        return model_fn_lib.EstimatorSpec(
            mode=None,
            predictions=tf.constant(0.),
            loss=tf.constant(0.),
            train_op=tf.assign_add(tf.train.get_global_step(), 1))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    self.assertEqual(0, model_fn_call_count[0])
    est.train(input_fn=_input_fn, steps=1)
    self.assertEqual(1, model_fn_call_count[0])

  def test_labels_should_be_none_if_model_fn_does_not_use_labels(self):

    def _input_fn_with_labels():
      return {'x': 4, 'y': 5}, [4]

    def _model_fn(features):
      _ = features
      return model_fn_lib.EstimatorSpec(
          mode=None,
          predictions=tf.constant(0.),
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    with self.assertRaisesRegexp(ValueError, 'model_fn does not take labels'):
      est.train(input_fn=_input_fn_with_labels, steps=1)

  def test_input_fn_len_should_be_2_if_tuple_or_list(self):

    def _input_fn():
      return 4, 5, 6

    def _model_fn(features):
      _ = features

    est = estimator.EstimatorV2(model_fn=_model_fn)
    with self.assertRaisesRegexp(ValueError, 'len 2 tuple'):
      est.train(input_fn=_input_fn, steps=1)

  def test_all_model_fn_args(self):
    expected_features = {'x': 42., 'y': 43.}
    expected_labels = 44.
    expected_params = {'some_param': 'some_value'}
    expected_config = run_config.RunConfig()
    expected_config.i_am_test = True

    # TODO(ptucker): We have to roll our own mock since Estimator._get_arguments
    # doesn't work with mock fns.
    model_fn_call_count = [0]

    # Note that args are all passed by keyword, so can be in any order.
    def _model_fn(mode, params, features, labels, config):
      model_fn_call_count[0] += 1
      self.assertItemsEqual(expected_features.keys(), features.keys())
      self.assertEqual(ModeKeys.TRAIN, mode)
      self.assertEqual(expected_params, params)
      self.assertTrue(config.i_am_test)
      return _estimator_spec(expected_features, expected_labels, features,
                             labels, mode)

    est = estimator.EstimatorV2(
        model_fn=_model_fn, params=expected_params, config=expected_config)
    self.assertEqual(0, model_fn_call_count[0])
    est.train(
        input_fn=_make_input_fn(expected_features, expected_labels), steps=1)
    self.assertEqual(1, model_fn_call_count[0])

  def test_partial_model_fn_args(self):
    expected_features = {'x': 42., 'y': 43.}
    expected_labels = 44.
    expected_params = {'some_param': 'some_value'}
    expected_config = run_config.RunConfig()
    expected_config.i_am_test = True
    expected_foo = 45.
    expected_bar = 46.

    # TODO(ptucker): We have to roll our own mock since Estimator._get_arguments
    # doesn't work with mock fns.
    model_fn_call_count = [0]

    def _model_fn(features, labels, foo, mode, params, config, bar):
      model_fn_call_count[0] += 1
      self.assertEqual(expected_foo, foo)
      self.assertEqual(expected_bar, bar)
      self.assertItemsEqual(expected_features.keys(), features.keys())
      self.assertEqual(ModeKeys.TRAIN, mode)
      self.assertEqual(expected_params, params)
      self.assertTrue(config.i_am_test)
      return _estimator_spec(expected_features, expected_labels, features,
                             labels, mode)

    partial_model_fn = functools.partial(
        _model_fn, foo=expected_foo, bar=expected_bar)

    est = estimator.EstimatorV2(
        model_fn=partial_model_fn,
        params=expected_params,
        config=expected_config)
    self.assertEqual(0, model_fn_call_count[0])
    est.train(
        input_fn=_make_input_fn(expected_features, expected_labels), steps=1)
    self.assertEqual(1, model_fn_call_count[0])

  def test_model_fn_must_return_estimator_spec(self):

    def model_fn(features, labels):
      _, _ = features, labels
      return 'NotGoodNotGood'

    est = estimator.EstimatorV2(model_fn=model_fn)
    with self.assertRaisesRegexp(ValueError, 'EstimatorSpec'):
      est.train(dummy_input_fn, steps=1)

  def test_run_train_op_and_saves_at_the_end(self):
    est = estimator.EstimatorV2(model_fn=model_fn_global_step_incrementer)
    est.train(dummy_input_fn, steps=5)
    self.assertEqual(
        5, estimator._load_global_step_from_checkpoint_dir(est.model_dir))

  def test_loss_summary(self):
    est = estimator.EstimatorV2(
        model_fn=model_fn_global_step_incrementer,
        config=run_config.RunConfig(save_summary_steps=1))
    est.train(dummy_input_fn, steps=1)

    # Make sure nothing is stuck in limbo.
    tf.summary.FileWriterCache.clear()

    if check_eventfile_for_keyword('loss', est.model_dir):
      return
    self.fail('{} should be part of reported summaries.'.format('loss'))

  def test_latest_checkpoint(self):
    est = estimator.EstimatorV2(model_fn=model_fn_global_step_incrementer)
    self.assertIsNone(est.latest_checkpoint())
    est.train(dummy_input_fn, steps=5)
    self.assertIsNotNone(est.latest_checkpoint())
    self.assertTrue(est.latest_checkpoint().startswith(est.model_dir))

  def test_steps_and_saves_reloads(self):
    est = estimator.EstimatorV2(model_fn=model_fn_global_step_incrementer)
    est.train(dummy_input_fn, steps=5)
    self.assertEqual(
        5, estimator._load_global_step_from_checkpoint_dir(est.model_dir))
    est.train(dummy_input_fn, steps=5)
    self.assertEqual(
        10, estimator._load_global_step_from_checkpoint_dir(est.model_dir))

  def test_warm_starts(self):

    def _make_model_fn(x):

      def _variable_creating_model_fn(features, labels, mode):
        _, _ = features, labels
        tf.get_variable('x', initializer=x)
        global_step = tf.train.get_global_step()
        return model_fn_lib.EstimatorSpec(
            mode, loss=tf.constant(1.), train_op=tf.assign_add(global_step, 1))

      return _variable_creating_model_fn

    est = estimator.EstimatorV2(model_fn=_make_model_fn(42.))
    est.train(dummy_input_fn, steps=10)

    warm_started_est = estimator.EstimatorV2(
        model_fn=_make_model_fn(36.), warm_start_from=est.model_dir)
    warm_started_est.train(dummy_input_fn, steps=5)
    # warm_start is called after the model_fn, so x should have the value
    # from the checkpoint.
    self.assertEqual(42., warm_started_est.get_variable_value('x'))
    # global_step should not be warm-started.
    self.assertEqual(
        5,
        estimator._load_global_step_from_checkpoint_dir(
            warm_started_est.model_dir))

  @test_util.run_v1_only('b/119219961')
  def test_warm_starts_from_savedmodel(self):

    def _make_model_fn(x):

      def _variable_creating_and_export_model_fn(features, labels, mode):
        _, _ = features, labels
        tf.get_variable('x', initializer=x)
        global_step = tf.train.get_global_step()
        return model_fn_lib.EstimatorSpec(
            mode,
            predictions={'y': tf.constant(1.0)},
            loss=tf.constant(1.),
            train_op=tf.assign_add(global_step, 1),
            export_outputs={
                'test':
                    export_lib.ClassificationOutput(
                        tf.constant([4.2]), tf.constant(['label']))
            })

      return _variable_creating_and_export_model_fn

    est = estimator.EstimatorV2(model_fn=_make_model_fn(42.))
    est.train(dummy_input_fn, steps=10)
    feature_spec = {
        'x': tf.io.VarLenFeature(dtype=tf.dtypes.int64),
        'y': tf.io.VarLenFeature(dtype=tf.dtypes.int64)
    }
    serving_input_receiver_fn = (
        export_lib.build_parsing_serving_input_receiver_fn(feature_spec))
    tmpdir = tempfile.mkdtemp()
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('export'))
    export_dir = est.export_saved_model(export_dir_base,
                                        serving_input_receiver_fn)

    # warm start using export dir
    warm_started_est = estimator.EstimatorV2(
        model_fn=_make_model_fn(36.), warm_start_from=export_dir)
    warm_started_est.train(dummy_input_fn, steps=5)
    # warm_start is called after the model_fn, so x should have the value
    # from the SavedModel.
    self.assertEqual(42., warm_started_est.get_variable_value('x'))

    # warm start using WarmStartSettings variables dir
    warm_start_settings = estimator.WarmStartSettings(
        ckpt_to_initialize_from=os.path.join(
            export_dir, tf.compat.as_bytes('variables')))
    warm_started_est = estimator.EstimatorV2(
        model_fn=_make_model_fn(46.), warm_start_from=warm_start_settings)
    warm_started_est.train(dummy_input_fn, steps=5)
    # warm_start is called after the model_fn, so x should have the value
    # from the SavedModel.
    self.assertEqual(42., warm_started_est.get_variable_value('x'))

  def test_max_step(self):
    est = estimator.EstimatorV2(model_fn=model_fn_global_step_incrementer)
    est.train(dummy_input_fn, max_steps=5)
    self.assertEqual(
        5, estimator._load_global_step_from_checkpoint_dir(est.model_dir))
    est.train(dummy_input_fn, max_steps=5)
    self.assertEqual(
        5, estimator._load_global_step_from_checkpoint_dir(est.model_dir))

  def test_checkpoint_contains_relative_paths(self):
    tmpdir = tempfile.mkdtemp()
    est = estimator.EstimatorV2(
        model_dir=tmpdir, model_fn=model_fn_global_step_incrementer)
    est.train(dummy_input_fn, steps=5)

    checkpoint_file_content = file_io.read_file_to_string(
        os.path.join(tmpdir, 'checkpoint'))
    ckpt = checkpoint_state_pb2.CheckpointState()
    text_format.Merge(checkpoint_file_content, ckpt)
    self.assertEqual(ckpt.model_checkpoint_path, 'model.ckpt-5')
    # TODO(b/78461127): Please modify tests to not directly rely on names of
    # checkpoints.
    self.assertAllEqual(['model.ckpt-0', 'model.ckpt-5'],
                        ckpt.all_model_checkpoint_paths)

  def test_train_save_copy_reload(self):
    tmpdir = tempfile.mkdtemp()
    model_dir1 = os.path.join(tmpdir, 'model_dir1')
    est1 = estimator.EstimatorV2(
        model_dir=model_dir1, model_fn=model_fn_global_step_incrementer)
    est1.train(dummy_input_fn, steps=5)

    # We have to clear the cache before we can rename the directory,
    # otherwise open file handles will prevent the delete on Windows.
    tf.summary.FileWriterCache.clear()
    model_dir2 = os.path.join(tmpdir, 'model_dir2')
    os.renames(model_dir1, model_dir2)

    est2 = estimator.EstimatorV2(
        model_dir=model_dir2, model_fn=model_fn_global_step_incrementer)
    self.assertEqual(
        5, estimator._load_global_step_from_checkpoint_dir(est2.model_dir))
    est2.train(dummy_input_fn, steps=5)
    self.assertEqual(
        10, estimator._load_global_step_from_checkpoint_dir(est2.model_dir))

  def test_steps0_raises_error(self):
    est = estimator.EstimatorV2(model_fn=_model_fn_with_eval_metric_ops)
    with self.assertRaisesRegexp(ValueError, 'Must specify steps > 0'):
      est.train(dummy_input_fn, steps=0)

  def test_steps_negative_raises_error(self):
    est = estimator.EstimatorV2(model_fn=_model_fn_with_eval_metric_ops)
    with self.assertRaisesRegexp(ValueError, 'Must specify steps > 0'):
      est.train(dummy_input_fn, steps=-1)

  def test_max_steps0_raises_error(self):
    est = estimator.EstimatorV2(model_fn=_model_fn_with_eval_metric_ops)
    with self.assertRaisesRegexp(ValueError, 'Must specify max_steps > 0'):
      est.train(dummy_input_fn, max_steps=0)

  def test_max_steps_negative_raises_error(self):
    est = estimator.EstimatorV2(model_fn=_model_fn_with_eval_metric_ops)
    with self.assertRaisesRegexp(ValueError, 'Must specify max_steps > 0'):
      est.train(dummy_input_fn, max_steps=-1)

  def test_scaffold_is_used(self):
    self.is_init_fn_called = False

    def _init_fn(scaffold, sess):
      _, _ = scaffold, sess
      self.is_init_fn_called = True

    def _model_fn_scaffold(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          scaffold=tf.train.Scaffold(init_fn=_init_fn))

    est = estimator.EstimatorV2(model_fn=_model_fn_scaffold)
    est.train(dummy_input_fn, steps=1)
    self.assertTrue(self.is_init_fn_called)

  def test_hooks_should_be_session_run_hook(self):
    est = estimator.EstimatorV2(model_fn=model_fn_global_step_incrementer)
    with self.assertRaisesRegexp(TypeError, 'must be a SessionRunHook'):
      est.train(dummy_input_fn, steps=1, hooks=['NotAHook'])

  def test_training_hooks_are_used(self):
    chief_hook = tf.test.mock.MagicMock(
        wraps=tf.train.SessionRunHook(), spec=tf.train.SessionRunHook)
    hook = tf.test.mock.MagicMock(
        wraps=tf.train.SessionRunHook(), spec=tf.train.SessionRunHook)

    def _model_fn_hooks(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          training_chief_hooks=[chief_hook],
          training_hooks=[hook])

    est = estimator.EstimatorV2(model_fn=_model_fn_hooks)
    self.assertFalse(chief_hook.begin.called)
    self.assertFalse(hook.begin.called)
    est.train(dummy_input_fn, steps=1)
    self.assertTrue(chief_hook.begin.called)
    self.assertTrue(hook.begin.called)

  def test_saving_listeners_are_used(self):
    listener = tf.test.mock.Mock(spec=tf.train.CheckpointSaverListener)
    listener.after_save.return_value = None
    est = estimator.EstimatorV2(
        model_fn=model_fn_global_step_incrementer,
        config=run_config.RunConfig(save_checkpoints_steps=10))
    est.train(dummy_input_fn, steps=26, saving_listeners=[listener])
    self.assertEqual(4, listener.before_save.call_count)
    self.assertEqual(4, listener.after_save.call_count)

  def test_saver_hook_should_exist_to_use_saving_listeners(self):
    listener = tf.test.mock.Mock(spec=tf.train.CheckpointSaverListener)
    est = estimator.EstimatorV2(
        model_fn=model_fn_global_step_incrementer,
        config=run_config.RunConfig(
            save_checkpoints_steps=None, save_checkpoints_secs=None))
    with self.assertRaisesRegexp(ValueError,
                                 'CheckpointSaverHook to use saving_listeners'):
      est.train(dummy_input_fn, steps=1, saving_listeners=[listener])

  def test_listeners_should_be_listeners(self):
    est = estimator.EstimatorV2(model_fn=model_fn_global_step_incrementer)
    with self.assertRaisesRegexp(TypeError,
                                 'must be a list of CheckpointSaverListener'):
      est.train(dummy_input_fn, steps=1, saving_listeners=['not-a-listener'])

  def test_chief_only_hook_should_not_be_called_on_non_chief(self):
    chief_hook = tf.test.mock.MagicMock(
        wraps=tf.train.SessionRunHook(), spec=tf.train.SessionRunHook)
    hook = tf.test.mock.MagicMock(
        wraps=tf.train.SessionRunHook(), spec=tf.train.SessionRunHook)

    def _model_fn_hooks(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          training_chief_hooks=[chief_hook],
          training_hooks=[hook])

    class NonChiefRunConfig(run_config.RunConfig):

      @property
      def is_chief(self):  # pylint: disable=g-wrong-blank-lines
        return False

    # Mocking the SessionManager.wait_for_session, so that worker doesn't wait
    # for chief.
    def get_initialized_session(*args, **kwargs):
      # Session doesn't take 'max_wait_secs' argument.
      kwargs.pop('max_wait_secs', None)
      scaffold = tf.train.Scaffold().finalize()
      sess = tf.Session(*args, **kwargs)
      sess.run(scaffold.init_op)
      return sess

    with tf.test.mock.patch.object(
        tf.train.SessionManager,
        'wait_for_session',
        side_effect=get_initialized_session):
      est = estimator.EstimatorV2(
          model_fn=_model_fn_hooks, config=NonChiefRunConfig())
      self.assertFalse(chief_hook.begin.called)
      self.assertFalse(hook.begin.called)
      est.train(dummy_input_fn, steps=1)
      self.assertFalse(chief_hook.begin.called)
      self.assertTrue(hook.begin.called)

  def test_features_labels_mode(self):
    given_features = {'test-features': [[1], [1]]}
    given_labels = {'test-labels': [[1], [1]]}

    def _input_fn():
      return given_features, given_labels

    def _model_fn(features, labels, mode):
      self.features, self.labels, self.mode = features, labels, mode
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=tf.constant([[0.]]))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(_input_fn, steps=1)
    self.assertEqual(given_features, self.features)
    self.assertEqual(given_labels, self.labels)
    self.assertEqual(ModeKeys.TRAIN, self.mode)

  def test_graph_initialization_global_step_and_random_seed(self):
    expected_random_seed = run_config.RunConfig().tf_random_seed

    def _model_fn(features, labels, mode):
      _, _, _ = features, labels, mode
      self.assertIsNotNone(tf.train.get_global_step())
      self.assertEqual(expected_random_seed, tf.get_default_graph().seed)
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=tf.constant([[0.]]))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)

  def test_config_should_not_be_evaluator_or_ps(self):

    class FakeEvaluatorConfig(run_config.RunConfig):

      @property
      def task_type(self):
        return run_config.TaskType.EVALUATOR

    est = estimator.EstimatorV2(
        model_fn=dummy_model_fn, config=FakeEvaluatorConfig())
    with self.assertRaisesRegexp(ValueError, 'train_and_evaluate'):
      est.train(dummy_input_fn, steps=1)

  def test_master_distributed_hooks(self):
    tf_config = json.dumps({
        'cluster': {
            run_config.TaskType.PS: ['localhost:1234'],
            run_config.TaskType.WORKER: ['localhost:1235'],
            run_config.TaskType.MASTER: ['localhost:1236']
        },
        'task': {
            'type': run_config.TaskType.MASTER,
            'index': 0
        }
    })
    with tf.test.mock.patch.dict('os.environ', {'TF_CONFIG': tf_config}):
      est = estimator.EstimatorV2(
          model_fn=model_fn_global_step_incrementer,
          config=run_config.RunConfig())

    with tf.test.mock.patch.object(training,
                                   'MonitoredTrainingSession') as mock_sess:
      est.train(dummy_input_fn, steps=1)
      self.assertFalse(
          any(
              isinstance(hook, tf.train.SummarySaverHook)
              for hook in mock_sess.call_args[1]['hooks']))
      self.assertFalse(
          any(
              isinstance(hook, tf.train.StepCounterHook)
              for hook in mock_sess.call_args[1]['hooks']))
      self.assertEqual(0, mock_sess.call_args[1]['save_summaries_steps'])
      self.assertIsNone(mock_sess.call_args[1]['log_step_count_steps'])

  def test_master_distributed_hooks_for_worker_0(self):
    tf_config = json.dumps({
        'cluster': {
            run_config.TaskType.PS: ['localhost:1234'],
            run_config.TaskType.WORKER: ['localhost:1235'],
            run_config.TaskType.MASTER: ['localhost:1236']
        },
        'task': {
            'type': run_config.TaskType.WORKER,
            'index': 0
        }
    })
    with tf.test.mock.patch.dict('os.environ', {'TF_CONFIG': tf_config}):
      est = estimator.EstimatorV2(
          model_fn=model_fn_global_step_incrementer,
          config=run_config.RunConfig())

    with tf.test.mock.patch.object(training,
                                   'MonitoredTrainingSession') as mock_sess:
      est.train(dummy_input_fn, steps=1)
      self.assertTrue(
          any(
              isinstance(hook, tf.train.SummarySaverHook)
              for hook in mock_sess.call_args[1]['hooks']))
      self.assertTrue(
          any(
              isinstance(hook, tf.train.StepCounterHook)
              for hook in mock_sess.call_args[1]['hooks']))
      self.assertEqual(0, mock_sess.call_args[1]['save_summaries_steps'])
      self.assertIsNone(mock_sess.call_args[1]['log_step_count_steps'])

  def test_master_distributed_hooks_for_worker_nonzero(self):
    tf_config = json.dumps({
        'cluster': {
            run_config.TaskType.PS: ['localhost:1234'],
            run_config.TaskType.WORKER: ['localhost:1235', 'localhost:1237'],
            run_config.TaskType.MASTER: ['localhost:1236']
        },
        'task': {
            'type': run_config.TaskType.WORKER,
            'index': 1
        }
    })
    with tf.test.mock.patch.dict('os.environ', {'TF_CONFIG': tf_config}):
      est = estimator.EstimatorV2(
          model_fn=model_fn_global_step_incrementer,
          config=run_config.RunConfig())

    with tf.test.mock.patch.object(training,
                                   'MonitoredTrainingSession') as mock_sess:
      est.train(dummy_input_fn, steps=1)
      self.assertFalse(
          any(
              isinstance(hook, tf.train.SummarySaverHook)
              for hook in mock_sess.call_args[1]['hooks']))
      self.assertFalse(
          any(
              isinstance(hook, tf.train.StepCounterHook)
              for hook in mock_sess.call_args[1]['hooks']))
      self.assertEqual(0, mock_sess.call_args[1]['save_summaries_steps'])
      self.assertIsNone(mock_sess.call_args[1]['log_step_count_steps'])

  def test_master_hooks_single_replica(self):
    tf_config = json.dumps({
        'cluster': {
            run_config.TaskType.MASTER: ['localhost:1234']
        },
        'task': {
            'type': run_config.TaskType.MASTER,
            'index': 0
        }
    })
    with tf.test.mock.patch.dict('os.environ', {'TF_CONFIG': tf_config}):
      est = estimator.EstimatorV2(
          model_fn=model_fn_global_step_incrementer,
          config=run_config.RunConfig(
              save_summary_steps=100, log_step_count_steps=200))

    with tf.test.mock.patch.object(training,
                                   'MonitoredTrainingSession') as mock_sess:
      est.train(dummy_input_fn, steps=1)
      self.assertFalse(
          any(
              isinstance(hook, tf.train.SummarySaverHook)
              for hook in mock_sess.call_args[1]['hooks']))
      self.assertFalse(
          any(
              isinstance(hook, tf.train.StepCounterHook)
              for hook in mock_sess.call_args[1]['hooks']))
      self.assertEqual(100, mock_sess.call_args[1]['save_summaries_steps'])
      self.assertEqual(200, mock_sess.call_args[1]['log_step_count_steps'])

  def test_master_hooks_single_replica_with_ps(self):
    tf_config = json.dumps({
        'cluster': {
            run_config.TaskType.MASTER: ['localhost:1234'],
            run_config.TaskType.PS: ['localhost: 1235'],
        },
        'task': {
            'type': run_config.TaskType.MASTER,
            'index': 0
        }
    })
    with tf.test.mock.patch.dict('os.environ', {'TF_CONFIG': tf_config}):
      est = estimator.EstimatorV2(
          model_fn=model_fn_global_step_incrementer,
          config=run_config.RunConfig(
              save_summary_steps=100, log_step_count_steps=200))

    with tf.test.mock.patch.object(training,
                                   'MonitoredTrainingSession') as mock_sess:
      est.train(dummy_input_fn, steps=1)
      self.assertFalse(
          any(
              isinstance(hook, tf.train.SummarySaverHook)
              for hook in mock_sess.call_args[1]['hooks']))
      self.assertFalse(
          any(
              isinstance(hook, tf.train.StepCounterHook)
              for hook in mock_sess.call_args[1]['hooks']))
      self.assertEqual(100, mock_sess.call_args[1]['save_summaries_steps'])
      self.assertEqual(200, mock_sess.call_args[1]['log_step_count_steps'])

  def test_hooks_with_distributed_collective_ops(self):
    if tf.executing_eagerly():
      self.skipTest('n/a: legacy graph only')
    tf_config = json.dumps({
        'cluster': {
            run_config.TaskType.WORKER: ['', ''],
        },
        'task': {
            'type': run_config.TaskType.WORKER,
            'index': 0
        }
    })
    # We let it skip setting eager context in multi-worker path by creating a
    # single-worker strategy and then passing cluster info into it.
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    strategy.configure(
        cluster_spec={
            run_config.TaskType.WORKER: ['', ''],
        },
        task_type=run_config.TaskType.WORKER,
        task_id=0)
    with tf.test.mock.patch.dict('os.environ', {'TF_CONFIG': tf_config}):
      config = run_config.RunConfig(
          train_distribute=strategy,
          save_summary_steps=1000,
          save_checkpoints_steps=500)
      config._distribute_coordinator_mode = None  # Skip distribute coordintor.
      est = estimator.EstimatorV2(
          model_fn=model_fn_global_step_incrementer, config=config)

    def input_fn():
      return tf.data.Dataset.from_tensors(({
          'x': tf.constant([[1], [1]])
      }, tf.constant([[1], [1]])))

    with tf.test.mock.patch.object(training,
                                   'MonitoredTrainingSession') as mock_sess:
      est.train(input_fn, steps=1)
      self.assertFalse(
          any(
              isinstance(hook, tf.train.SummarySaverHook)
              for hook in mock_sess.call_args[1]['hooks']))
      self.assertFalse(
          any(
              isinstance(hook, tf.train.StepCounterHook)
              for hook in mock_sess.call_args[1]['hooks']))
      self.assertFalse(
          any(
              isinstance(hook, tf.train.CheckpointSaverHook)
              for hook in mock_sess.call_args[1]['hooks']))
      self.assertEqual(1000, mock_sess.call_args[1]['save_summaries_steps'])
      self.assertEqual(500, mock_sess.call_args[1]['save_checkpoint_steps'])
      self.assertEqual(100, mock_sess.call_args[1]['log_step_count_steps'])


def _model_fn_with_eval_metric_ops(features, labels, mode, params):
  _, _ = features, labels
  global_step = tf.train.get_global_step()
  loss = tf.constant(1.)
  metric_name_1 = params.get('metric_name') or 'metric'
  metric_value_1 = params.get('metric_value') or 2.
  metric_name_2 = params.get('metric_name_2') or 'metric2'
  metric_value_2 = params.get('metric_value_2') or 2.

  metric_update_op = loss.op
  metric_tensor = control_flow_ops.with_dependencies(
      [metric_update_op], tf.constant(metric_value_1))

  mean = tf.keras.metrics.Mean()
  mean.update_state(metric_value_2)
  return model_fn_lib.EstimatorSpec(
      mode,
      loss=loss,
      predictions={'predictions': tf.constant(1.)},
      train_op=tf.assign_add(global_step, 1),
      eval_metric_ops={
          metric_name_1: (metric_tensor, metric_update_op),
          metric_name_2: mean,
      })


class _StepCounterHook(tf.train.SessionRunHook):
  """Hooks that counts the number of times it is called."""

  def __init__(self):
    self._steps = 0

  def before_run(self, run_context):
    del run_context
    self._steps += 1

  @property
  def steps(self):
    return self._steps


class EstimatorGetVariablesTest(tf.test.TestCase):

  def test_model_should_be_trained(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      tf.Variable(1., name='one')
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    with self.assertRaisesRegexp(ValueError, 'not find trained model'):
      est.get_variable_names()
    with self.assertRaisesRegexp(ValueError, 'not find trained model'):
      est.get_variable_value('one')

  def test_get_variable_utils(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      tf.Variable(1., name='one')
      tf.Variable(3., name='three')
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(input_fn=dummy_input_fn, steps=1)
    self.assertEqual(
        set(['one', 'three', 'global_step']), set(est.get_variable_names()))
    self.assertEqual(1., est.get_variable_value('one'))
    self.assertEqual(3., est.get_variable_value('three'))


class EstimatorTraceTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    self._profiler_dir = os.path.join(self.get_temp_dir(), 'profiler')

  expected_features = {'x': 42., 'y': 43.}
  expected_labels = 44.
  model_fn_call_count = [0]
  input_fn = _make_input_fn(expected_features, expected_labels)

  class ModelFn(object):

    def __call__(self, features, labels):
      EstimatorTraceTest.model_fn_call_count[0] += 1
      return _estimator_spec(EstimatorTraceTest.expected_features,
                             EstimatorTraceTest.expected_labels, features,
                             labels, ModeKeys.TRAIN)

  def assert_profiler_captures_steps(self):
    profile_dir = os.path.join(self._profiler_dir, 'plugins', 'profile')
    run = gfile.ListDirectory(profile_dir)[0]
    hostname = socket.gethostname()
    overview_page = os.path.join(profile_dir, run,
                                 hostname + '.overview_page.pb')
    with open(overview_page, 'r', encoding='latin-1') as f:
      overview_page_content = f.read()
      # Asserts step time is profiled
      self.assertIn('PerGenericStepDetails', overview_page_content)
      self.assertNotIn('No step time measured', overview_page_content)

  @combinations.generate(
      combinations.combine(use_train_and_evaluate=[True, False]))
  def test_profiler_traces_steps(self, use_train_and_evaluate):
    est = estimator.EstimatorV2(
        model_fn=EstimatorTraceTest.ModelFn(), config=run_config.RunConfig())
    steps = 1
    profiler.start(self._profiler_dir)
    if use_train_and_evaluate:
      estimator_training.train_and_evaluate(
          est,
          estimator_training.TrainSpec(
              EstimatorTraceTest.input_fn, max_steps=steps),
          estimator_training.EvalSpec(EstimatorTraceTest.input_fn))
    else:
      est.train(input_fn=EstimatorTraceTest.input_fn, steps=steps)
    profiler.stop()
    # With https://github.com/tensorflow/estimator/pull/68, estimator should be able to work with profiler
    self.assert_profiler_captures_steps()


class EstimatorDatasetIntegrationTest(tf.test.TestCase):
  """Tests dataset integration."""

  def test_returned_by_input_fn(self):

    def _input_fn():
      return tf.data.Dataset.from_tensors(([1.], [2.]))

    def _model_fn(features, labels, mode):
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=features + labels,  # 1 + 2
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(_input_fn, steps=1)
    scores = est.evaluate(_input_fn, steps=1)
    self.assertEqual(3., scores[model_fn_lib.LOSS_METRIC_KEY])

  def test_with_none_labels(self):

    def _input_fn():
      return tf.data.Dataset.from_tensors([7.])

    def _model_fn(features, labels, mode):
      self.assertIsNone(labels)
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=features,  # 7
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(_input_fn, steps=1)
    scores = est.evaluate(_input_fn, steps=1)
    self.assertEqual(7., scores[model_fn_lib.LOSS_METRIC_KEY])

  def test_with_predict(self):

    def _input_fn():
      return tf.data.Dataset.from_tensors([10.])

    def _model_fn(features, labels, mode):
      _ = labels
      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=features,  # 10
          loss=features,  # 10
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(_input_fn, steps=1)
    self.assertEqual([10.], next(est.predict(input_fn=_input_fn)))

  def test_batching(self):

    def _input_fn():
      return tf.data.Dataset.from_tensor_slices(
          ([[1.], [2.]], [[10.], [20.]])).batch(1)

    def _model_fn(features, labels, mode):
      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=features,
          loss=features + (0 if labels is None else labels),  # 11, 22
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(_input_fn)
    scores = est.evaluate(_input_fn)
    # (11 + 22)/2 = 16.5
    self.assertEqual(16.5, scores[model_fn_lib.LOSS_METRIC_KEY])
    self.assertEqual([1., 2.], list(est.predict(_input_fn)))


class EstimatorEvaluateTest(tf.test.TestCase):

  def test_eval_dir(self):
    est = estimator.EstimatorV2(
        model_fn=model_fn_global_step_incrementer, model_dir='some_path')
    expected_eval_dir = os.path.join('some_path', 'eval')
    self.assertEqual(expected_eval_dir, est.eval_dir())
    expected_eval_dir_name = os.path.join('some_path', 'eval_a_name')
    self.assertEqual(expected_eval_dir_name, est.eval_dir('a_name'))

  def test_input_fn_args(self):
    expected_mode = ModeKeys.EVAL
    expected_params = {'batch_size': 10}
    expected_config = run_config.RunConfig().replace(tf_random_seed=4321)
    input_fn_call_count = [0]

    def _model_fn(features, labels, mode, params, config):
      del params, config
      return model_fn_global_step_incrementer(features, labels, mode)

    def _input_fn(mode, params, config):
      input_fn_call_count[0] += 1
      self.assertEqual(expected_mode, mode)
      self.assertEqual(expected_params, params)
      self.assertEqual(4321, config.tf_random_seed)
      return dummy_input_fn()

    est = estimator.EstimatorV2(
        model_fn=_model_fn, params=expected_params, config=expected_config)
    est.train(dummy_input_fn, steps=1)
    self.assertEqual(0, input_fn_call_count[0])
    est.evaluate(_input_fn, steps=1)
    self.assertEqual(1, input_fn_call_count[0])

  def test_model_fn_must_return_estimator_spec(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      if mode == ModeKeys.EVAL:
        return 'NotGoodNotGood'
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(1.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    with self.assertRaisesRegexp(ValueError,
                                 'model_fn should return an EstimatorSpec'):
      est.evaluate(dummy_input_fn, steps=1)

  def test_no_checkpoint_uses_init(self):

    def _model_fn(features, labels, mode, params):
      del features, labels, params
      mean = tf.keras.metrics.Mean()
      mean.update_state(tf.Variable(2.) + 1)
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(1.),
          eval_metric_ops={
              'mean1': mean,
              'mean2': tf.metrics.mean(tf.compat.v1.Variable(2.) + 1)
          })

    est = estimator.EstimatorV2(model_fn=_model_fn)
    scores = est.evaluate(dummy_input_fn, steps=1)
    # Metric value here is set to 1 + the value of the Variable that is newly
    # initialized (since there is no checkpoint).
    self.assertEqual(3., scores['mean1'])
    self.assertEqual(3., scores['mean2'])

  @test_util.run_v1_only('b/119219961')
  def test_no_checkpoint_uses_init_with_warm_starting(self):

    def _make_model_fn(x):

      def _variable_creating_and_export_model_fn(features, labels, mode):
        _, _ = features, labels
        x_var = tf.get_variable('x', initializer=x)
        global_step = tf.train.get_global_step()
        mean = tf.keras.metrics.Mean()
        mean.update_state(x_var + 1)
        return model_fn_lib.EstimatorSpec(
            mode,
            predictions={'y': tf.constant(1.0)},
            loss=tf.constant(1.),
            eval_metric_ops={
                'mean1': mean,
                'mean2': tf.metrics.mean(x_var + 1)
            },
            train_op=tf.assign_add(global_step, 1),
            export_outputs={
                'test':
                    export_lib.ClassificationOutput(
                        tf.constant([4.2]), tf.constant(['label']))
            })

      return _variable_creating_and_export_model_fn

    first_est = estimator.EstimatorV2(model_fn=_make_model_fn(42.))
    first_est.train(dummy_input_fn, steps=10)
    feature_spec = {
        'x': tf.io.VarLenFeature(dtype=tf.dtypes.int64),
        'y': tf.io.VarLenFeature(dtype=tf.dtypes.int64)
    }
    serving_input_receiver_fn = (
        export_lib.build_parsing_serving_input_receiver_fn(feature_spec))
    tmpdir = tempfile.mkdtemp()
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('export'))
    exported_path = first_est.export_saved_model(export_dir_base,
                                                 serving_input_receiver_fn)

    # Test that we can pass either warm_start_from as an external checkpoint
    # or an exported SavedModel.
    est = estimator.EstimatorV2(
        model_fn=_make_model_fn(52.), warm_start_from=exported_path)
    eval_metrics = est.evaluate(dummy_input_fn, steps=1)
    # Metric value here is set to 1 + the value of the Variable that is
    # warm-started from the SavedModel of the first model (42.), as opposed to
    # the initialization in the new model_fn (52.).
    self.assertEqual(43., eval_metrics['mean1'])
    self.assertEqual(43., eval_metrics['mean2'])

    est = estimator.EstimatorV2(
        model_fn=_make_model_fn(62.), warm_start_from=first_est.model_dir)
    eval_metrics = est.evaluate(dummy_input_fn, steps=1)
    # Metric value here is set to 1 + the value of the Variable that is
    # warm-started from a checkpoint of the first model (42.), as opposed to
    # the initialization in the new model_fn (52.).
    self.assertEqual(43., eval_metrics['mean1'])
    self.assertEqual(43., eval_metrics['mean2'])

    # Also test that we can use WarmStartSettings with an exported SavedModel's
    # variables dir
    warm_start_settings = estimator.WarmStartSettings(
        ckpt_to_initialize_from=os.path.join(
            exported_path, tf.compat.as_bytes('variables')))
    est = estimator.EstimatorV2(
        model_fn=_make_model_fn(72.), warm_start_from=warm_start_settings)
    eval_metrics = est.evaluate(dummy_input_fn, steps=1)
    # Metric value here is set to 1 + the value of the Variable that is
    # warm-started from a checkpoint of the first model (42.), as opposed to
    # the initialization in the new model_fn (52.).
    self.assertEqual(43., eval_metrics['mean1'])
    self.assertEqual(43., eval_metrics['mean2'])

  def test_scores(self):
    est = estimator.EstimatorV2(
        model_fn=_model_fn_with_eval_metric_ops,
        params={
            'metric_name': 'metric',
            'metric_value': 2.,
            'metric_name_2': 'metric2',
            'metric_value_2': 3.,
        })
    est.train(dummy_input_fn, steps=5)
    scores = est.evaluate(dummy_input_fn, steps=1)
    self.assertIn('metric', scores)
    self.assertAlmostEqual(2., scores['metric'])
    self.assertIn('metric2', scores)
    self.assertAlmostEqual(3., scores['metric2'])

  def test_tuple_metrics(self):

    def _model_fn(features, labels, mode):
      del features  # unused
      del labels
      return model_fn_lib.EstimatorSpec(
          mode,
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          loss=tf.constant(1.),
          eval_metric_ops={
              'nested_metric': (
                  ((tf.constant(2.), tf.constant(1)),
                   tf.constant(3., dtype=tf.dtypes.float64)), tf.no_op())
          })

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    evaluation = est.evaluate(dummy_input_fn, steps=1)
    ((two_float, one_integer), three_double) = evaluation['nested_metric']
    self.assertAlmostEqual(2., two_float)
    self.assertEqual(1, one_integer)
    self.assertAlmostEqual(3., three_double)

  def test_steps0_raises_error(self):
    est = estimator.EstimatorV2(model_fn=_model_fn_with_eval_metric_ops)
    est.train(dummy_input_fn, steps=5)
    with self.assertRaisesRegexp(ValueError, 'Must specify steps > 0'):
      est.evaluate(dummy_input_fn, steps=0)

  def test_steps_negative_raises_error(self):
    est = estimator.EstimatorV2(model_fn=_model_fn_with_eval_metric_ops)
    est.train(dummy_input_fn, steps=5)
    with self.assertRaisesRegexp(ValueError, 'Must specify steps > 0'):
      est.evaluate(dummy_input_fn, steps=-1)

  def test_global_step_metric_raises_error(self):
    est = estimator.EstimatorV2(
        model_fn=_model_fn_with_eval_metric_ops,
        params={
            'metric_name': 'global_step',
            'metric_value': 2.
        })
    est.train(dummy_input_fn, steps=5)
    with self.assertRaisesRegexp(
        ValueError, 'Metric with name `global_step` is not allowed'):
      est.evaluate(dummy_input_fn, steps=1)

  def test_global_step_is_reported(self):
    est = estimator.EstimatorV2(
        model_fn=_model_fn_with_eval_metric_ops,
        params={
            'metric_name': 'metric',
            'metric_value': 2.,
            'metric_name_2': 'metric2',
            'metric_value_2': 3.,
        })
    est.train(dummy_input_fn, steps=5)
    scores = est.evaluate(dummy_input_fn, steps=1)
    self.assertIn('global_step', scores)
    self.assertEqual(5, scores['global_step'])

  def test_loss_metric_is_reported(self):

    def _model_fn_with_incremental_loss(features, labels, mode):
      _, _ = features, labels
      local_weight = tf.Variable(
          0., name='local_weight', collections=[tf.GraphKeys.LOCAL_VARIABLES])
      # Loss will be 2, 4, 6, ...
      loss = 2 * tf.assign_add(local_weight, 1.)
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=loss,
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1))

    est = estimator.EstimatorV2(model_fn=_model_fn_with_incremental_loss)
    est.train(dummy_input_fn, steps=1)
    scores = est.evaluate(dummy_input_fn, steps=5)
    self.assertIn(model_fn_lib.LOSS_METRIC_KEY, scores)
    # Average loss will be (2 + 4 + 6 + 8 + 10)/5=6
    self.assertAlmostEqual(6., scores[model_fn_lib.LOSS_METRIC_KEY])

  def test_hooks_should_be_session_run_hook(self):
    est = estimator.EstimatorV2(model_fn=model_fn_global_step_incrementer)
    est.train(dummy_input_fn, steps=1)
    with self.assertRaisesRegexp(TypeError, 'must be a SessionRunHook'):
      est.evaluate(dummy_input_fn, steps=5, hooks=['NotAHook'])

  def test_hooks_are_used(self):
    step_counter_hook = _StepCounterHook()

    est = estimator.EstimatorV2(model_fn=_model_fn_with_eval_metric_ops)
    est.train(dummy_input_fn, steps=1)
    est.evaluate(dummy_input_fn, steps=5, hooks=[step_counter_hook])
    self.assertEqual(5, step_counter_hook.steps)

  def test_evaluate_from_checkpoint(self):
    params = {
        'metric_name': 'metric',
        'metric_value': 2.,
        'metric_name_2': 'metric2',
        'metric_value_2': 3.,
    }
    est1 = estimator.EstimatorV2(
        model_fn=_model_fn_with_eval_metric_ops, params=params)
    est1.train(dummy_input_fn, steps=5)
    est2 = estimator.EstimatorV2(
        model_fn=_model_fn_with_eval_metric_ops, params=params)
    scores = est2.evaluate(
        dummy_input_fn, steps=1, checkpoint_path=est1.latest_checkpoint())
    self.assertEqual(5, scores['global_step'])

  @test_util.run_v1_only('VariableV1 is only exported in v1')
  def test_wrong_shape_throws_reasonable_error(self):
    """Make sure we are helpful when model_fns change. See b/110263146."""

    def _get_model_fn(val=1):

      def _model_fn(features, labels, mode):
        del features, labels  # unused
        tf.Variable(val, name='weight')
        return model_fn_lib.EstimatorSpec(
            mode=mode,
            predictions=tf.constant([[1.]]),
            loss=tf.constant(0.),
            train_op=tf.assign_add(tf.train.get_global_step(), 1))

      return _model_fn

    model_fn_1 = _get_model_fn()
    model_fn_2 = _get_model_fn(val=[1])

    est1 = estimator.EstimatorV2(model_fn=model_fn_1)
    est1.train(dummy_input_fn, steps=5)
    est2 = estimator.EstimatorV2(model_fn=model_fn_2, model_dir=est1.model_dir)

    expected_msg = 'Restoring from checkpoint failed.*a mismatch between'
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, expected_msg):
      est2.train(
          dummy_input_fn,
          steps=1,
      )

  def test_scaffold_is_used(self):

    def _model_fn_scaffold(features, labels, mode):
      _, _ = features, labels
      tf.Variable(1., name='weight')
      self.mock_saver = get_mock_saver()
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          predictions=tf.constant([[1.]]),
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          scaffold=tf.train.Scaffold(saver=self.mock_saver))

    est = estimator.EstimatorV2(model_fn=_model_fn_scaffold)
    est.train(dummy_input_fn, steps=1)
    est.evaluate(dummy_input_fn, steps=1)
    self.assertTrue(self.mock_saver.restore.called)

  def test_features_labels_mode(self):
    given_features = {'test-features': [[1], [1]]}
    given_labels = {'test-labels': [[1], [1]]}

    def _input_fn():
      return given_features, given_labels

    def _model_fn(features, labels, mode):
      self.features, self.labels, self.mode = features, labels, mode
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=tf.constant([[0.]]))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(_input_fn, steps=1)
    est.evaluate(_input_fn, steps=1)
    self.assertEqual(given_features, self.features)
    self.assertEqual(given_labels, self.labels)
    self.assertEqual(ModeKeys.EVAL, self.mode)

  def test_graph_initialization_global_step_and_random_seed(self):
    expected_random_seed = run_config.RunConfig().tf_random_seed

    def _model_fn(features, labels, mode):
      _, _, _ = features, labels, mode
      self.assertIsNotNone(tf.train.get_global_step())
      self.assertEqual(expected_random_seed, tf.get_default_graph().seed)
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=tf.constant([[0.]]))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    est.evaluate(dummy_input_fn, steps=1)

  def test_evaluation_hooks_are_used(self):
    hook = tf.test.mock.MagicMock(
        wraps=tf.train.SessionRunHook(), spec=tf.train.SessionRunHook)

    def _model_fn_hooks(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          evaluation_hooks=[hook])

    est = estimator.EstimatorV2(model_fn=_model_fn_hooks)
    est.train(dummy_input_fn, steps=1)
    self.assertFalse(hook.begin.called)
    est.evaluate(dummy_input_fn, steps=1)
    self.assertTrue(hook.begin.called)

  def test_summary_writing_with_summary_proto(self):

    def model_fn_global_step_incrementer_image(features, labels, mode):
      _, _ = features, labels
      global_step = tf.train.get_global_step()

      image = tf.zeros([5, 3, 3, 1])
      eval_metric_ops = {
          'foo': (tf.summary.image('image', image,
                                   max_outputs=3), tf.constant(1))
      }
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(1.),
          train_op=tf.assign_add(global_step, 1),
          eval_metric_ops=eval_metric_ops)

    est = estimator.EstimatorV2(
        model_fn=model_fn_global_step_incrementer_image,
        config=run_config.RunConfig(save_summary_steps=1))
    est.train(dummy_input_fn, steps=200)
    est.evaluate(
        input_fn=dummy_input_fn,
        steps=200,
    )

    # Make sure nothing is stuck in limbo.
    tf.summary.FileWriterCache.clear()

    # Get last evaluation Event written.
    for key in ['foo/0', 'foo/1', 'foo/2']:
      self.assertTrue(
          check_eventfile_for_keyword(key, est.eval_dir()),
          '{} should be part of reported summaries.'.format(key))

    # Verify that evaluated checkpoint path is written to event file.
    checkpoint_path_tag = 'checkpoint_path'
    self.assertTrue(
        check_eventfile_for_keyword(checkpoint_path_tag, est.eval_dir()),
        '{} should be part of reported summaries.'.format(checkpoint_path_tag))

    expected_tensor_proto = tf.make_tensor_proto(
        est.latest_checkpoint(), dtype=tf.dtypes.string)
    summaries = summaries_with_matching_keyword(checkpoint_path_tag,
                                                est.eval_dir())
    self.assertProtoEquals(expected_tensor_proto,
                           next(summaries).value[0].tensor)

  def test_summary_writing_with_tensor(self):

    def model_fn_with_prediction_mean_tensor_eval_metric_ops(
        features, labels, mode, params):
      _, _ = features, labels
      global_step = tf.train.get_global_step()

      metric_name = params.get('metric_name') or 'metric'
      predictions = tf.constant([1., .5, 0.])
      eval_metric_ops = {metric_name: tf.metrics.mean_tensor(predictions)}
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(1.),
          predictions={'predictions': predictions},
          train_op=tf.assign_add(global_step, 1),
          eval_metric_ops=eval_metric_ops)

    metric_key = 'PMT'
    params = {
        'metric_name': metric_key,
    }
    est = estimator.EstimatorV2(
        model_fn=model_fn_with_prediction_mean_tensor_eval_metric_ops,
        params=params,
        config=run_config.RunConfig(save_summary_steps=1))
    est.train(input_fn=dummy_input_fn, steps=10)
    est.evaluate(
        input_fn=dummy_input_fn,
        steps=10,
    )

    tf.summary.FileWriterCache.clear()

    self.assertTrue(
        check_eventfile_for_keyword(metric_key, est.eval_dir()),
        '{} should be part of reported summaries.'.format(metric_key))

    summaries = summaries_with_matching_keyword(metric_key, est.eval_dir())
    for value in next(summaries).value:
      if value.tag == metric_key:
        self.assertTrue(value.HasField('tensor'))


class EstimatorPredictTest(tf.test.TestCase):

  def test_input_fn_args(self):
    expected_mode = ModeKeys.PREDICT
    expected_params = {'batch_size': 10}
    expected_config = run_config.RunConfig().replace(tf_random_seed=4321)
    input_fn_call_count = [0]

    def _model_fn(features, labels, mode, params, config):
      del features, labels, params, config
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=tf.constant([[10.]]))

    def _input_fn(mode, params, config):
      input_fn_call_count[0] += 1
      self.assertEqual(expected_mode, mode)
      self.assertEqual(expected_params, params)
      self.assertEqual(4321, config.tf_random_seed)
      return dummy_input_fn()

    est = estimator.EstimatorV2(
        model_fn=_model_fn, params=expected_params, config=expected_config)
    est.train(dummy_input_fn, steps=1)
    self.assertEqual(0, input_fn_call_count[0])
    next(est.predict(_input_fn))
    self.assertEqual(1, input_fn_call_count[0])

  def test_no_checkpoint_uses_init(self):

    def _model_fn(features, labels, mode, params, config):
      del features, labels, params, config
      x = tf.Variable([[3.]], name='x')
      return model_fn_lib.EstimatorSpec(mode, predictions=tf.math.add(x, 1.))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    # Expected prediction value is 1 + the value of the Variable that is newly
    # initialized (since there is no checkpoint).
    self.assertEqual(4., next(est.predict(dummy_input_fn)))

  @test_util.run_v1_only('b/119219961')
  def test_no_checkpoint_uses_init_with_warm_starting(self):

    def _make_model_fn(x):

      def _variable_creating_and_export_model_fn(features, labels, mode):
        _, _ = features, labels
        x_var = tf.Variable([[x]], name='x')
        return model_fn_lib.EstimatorSpec(
            mode,
            predictions=tf.math.add(x_var, 1.),
            loss=tf.constant(1.),
            train_op=tf.assign_add(tf.train.get_global_step(), 1),
            export_outputs={
                'test':
                    export_lib.ClassificationOutput(
                        tf.constant([4.2]), tf.constant(['label']))
            })

      return _variable_creating_and_export_model_fn

    first_est = estimator.EstimatorV2(model_fn=_make_model_fn(3.))
    first_est.train(dummy_input_fn, steps=10)
    feature_spec = {
        'x': tf.io.VarLenFeature(dtype=tf.dtypes.int64),
        'y': tf.io.VarLenFeature(dtype=tf.dtypes.int64)
    }
    serving_input_receiver_fn = (
        export_lib.build_parsing_serving_input_receiver_fn(feature_spec))
    tmpdir = tempfile.mkdtemp()
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('export'))
    exported_path = first_est.export_saved_model(export_dir_base,
                                                 serving_input_receiver_fn)

    # Test that we can pass either warm_start_from as an external checkpoint
    # or an exported SavedModel.
    est = estimator.EstimatorV2(
        model_fn=_make_model_fn(30.), warm_start_from=exported_path)
    # Prediction here is set to 1 + the value of the Variable that is
    # warm-started from the SavedModel of the first model (3.), as opposed to
    # the initialization in the new model_fn (30.).
    self.assertEqual(4., next(est.predict(dummy_input_fn)))

    est = estimator.EstimatorV2(
        model_fn=_make_model_fn(40.), warm_start_from=first_est.model_dir)
    # Prediction here is set to 1 + the value of the Variable that is
    # warm-started from a checkpoint of the first model (3.), as opposed to
    # the initialization in the new model_fn (40.).
    self.assertEqual(4., next(est.predict(dummy_input_fn)))

    # Also test that we can use WarmStartSettings with an exported SavedModel's
    # variables dir
    warm_start_settings = estimator.WarmStartSettings(
        ckpt_to_initialize_from=os.path.join(
            exported_path, tf.compat.as_bytes('variables')))
    est = estimator.EstimatorV2(
        model_fn=_make_model_fn(50.), warm_start_from=warm_start_settings)
    # Prediction here is set to 1 + the value of the Variable that is
    # warm-started from a checkpoint of the first model (3.), as opposed to
    # the initialization in the new model_fn (40.).
    self.assertEqual(4., next(est.predict(dummy_input_fn)))

  def test_no_trained_model_invalid_checkpoint_path(self):
    est = estimator.EstimatorV2(model_fn=model_fn_global_step_incrementer)
    with self.assertRaises(ValueError):
      next(
          est.predict(
              dummy_input_fn,
              checkpoint_path=tf.train.latest_checkpoint('fakedir')))

  def test_tensor_predictions(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=tf.constant([[10.]]))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    self.assertEqual(10., next(est.predict(dummy_input_fn)))

  def test_predictionhooks_are_used(self):
    hook = tf.test.mock.MagicMock(
        wraps=tf.train.SessionRunHook(), spec=tf.train.SessionRunHook)

    def _model_fn_hooks(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=tf.constant([[10.]]),
          prediction_hooks=[hook])

    est = estimator.EstimatorV2(model_fn=_model_fn_hooks)
    est.train(dummy_input_fn, steps=1)
    self.assertFalse(hook.begin.called)
    next(est.predict(dummy_input_fn))
    self.assertTrue(hook.begin.called)

  def test_warn_if_no_queue_runner(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=tf.constant([[10.]]))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    with tf.test.mock.patch.object(logging, 'warning') as mock_log:
      next(est.predict(dummy_input_fn))
      self.assertRegexpMatches(
          str(mock_log.call_args),
          'Input graph does not.*contain a QueueRunner.')

  def test_skip_warn_if_dataset_returns_features(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=tf.constant([[10.]]))

    def _input_fn():
      dataset = tf.data.Dataset.from_tensors([1])
      iterator = tf.data.make_one_shot_iterator(dataset)
      return iterator.get_next()

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    with tf.test.mock.patch.object(logging, 'warning') as mock_log:
      next(est.predict(_input_fn))
      # The warning should not have keyword QueueRunner.
      self.assertRegexpMatches(str(mock_log.call_args), '^((?!QueueRunner).)*$')

  def test_skip_warn_if_dataset_returns_features_dict(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=tf.constant([[10.]]))

    def _input_fn():
      dataset = tf.data.Dataset.from_tensors([1])
      iterator = tf.data.make_one_shot_iterator(dataset)
      features = {'age': iterator.get_next()}
      return features

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    with tf.test.mock.patch.object(logging, 'warning') as mock_log:
      next(est.predict(_input_fn))
      # The warning should not have keyword QueueRunner.
      self.assertRegexpMatches(str(mock_log.call_args), '^((?!QueueRunner).)*$')

  def test_input_fn_can_return_just_features(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=tf.constant([[10.]]))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)

    def _only_features():
      return {'x': tf.constant([[0.]])}

    self.assertEqual([10.], next(est.predict(_only_features)))

  def test_batch_size_mismatch(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions={
              'y1': tf.constant([[10.]]),
              'y2': tf.constant([[12.], [13]])
          })

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    with self.assertRaisesRegexp(ValueError,
                                 'Batch length of predictions should be same'):
      next(est.predict(dummy_input_fn))

  def test_iterate_batches(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions={
              # First dim is different but the prediction should still work
              'y1': tf.zeros(shape=[3]),
              'y2': tf.zeros(shape=[5, 3])
          })

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)

    predictions = next(est.predict(dummy_input_fn, yield_single_examples=False))
    self.assertAllEqual(predictions['y1'].shape, [3])
    self.assertAllEqual(predictions['y2'].shape, [5, 3])

  def test_predict_keys_defined_for_tensor(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=tf.constant([[10.]]))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    with self.assertRaisesRegexp(
        ValueError,
        'predict_keys argument is not valid in case of non-dict predictions'):
      next(est.predict(dummy_input_fn, predict_keys=['y']))

  def test_predict_keys_does_not_exists(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions={
              'y1': tf.constant([[10.]]),
              'y2': tf.constant([[12.]])
          })

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    with self.assertRaisesRegexp(ValueError,
                                 'Expected to run at least one output from'):
      next(est.predict(dummy_input_fn, predict_keys=['y3']))

  def test_return_given_predict_keys(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions={
              'y1': tf.constant([[10.]]),
              'y2': tf.constant([[12.]])
          })

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    results = next(est.predict(dummy_input_fn, predict_keys=['y1']))
    self.assertIn('y1', results)
    self.assertNotIn('y2', results)

  def test_yield_rows_of_tensor(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=tf.constant([[10.], [12.]]))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    results = est.predict(dummy_input_fn)
    self.assertEqual([10.], next(results))
    self.assertEqual([12.], next(results))

  def test_yield_rows_of_dict(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions={
              'y1': tf.constant([[10.], [12]]),
              'y2': tf.constant([[0.], [2.]])
          })

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    results = est.predict(dummy_input_fn)
    self.assertDictEqual({'y1': [10.], 'y2': [0.]}, next(results))
    self.assertDictEqual({'y1': [12.], 'y2': [2.]}, next(results))

  def test_hooks_should_be_session_run_hook(self):
    est = estimator.EstimatorV2(model_fn=model_fn_global_step_incrementer)
    est.train(dummy_input_fn, steps=1)
    with self.assertRaisesRegexp(TypeError, 'must be a SessionRunHook'):
      next(est.predict(dummy_input_fn, hooks=['NotAHook']))

  def test_hooks_are_used(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=tf.constant([[10.], [12.]]))

    step_counter_hook = _StepCounterHook()
    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    results = est.predict(dummy_input_fn, hooks=[step_counter_hook])
    self.assertEqual(0, step_counter_hook.steps)  # not called yet
    next(results)
    self.assertEqual(1, step_counter_hook.steps)  # first call
    next(results)
    self.assertEqual(1, step_counter_hook.steps)  # it's in same batch
    next(results)
    self.assertEqual(2, step_counter_hook.steps)  # next batch

  def test_predict_from_old_model_dir(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      v = tf.Variable([[16.]], name='weight')
      prediction = v * 2
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=prediction)

    est1 = estimator.EstimatorV2(model_fn=_model_fn)
    est1.train(dummy_input_fn, steps=1)
    est2 = estimator.EstimatorV2(model_fn=_model_fn, model_dir=est1.model_dir)
    self.assertEqual([32.], next(est2.predict(dummy_input_fn)))

  def test_predict_from_checkpoint_path(self):

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      v = tf.Variable([[16.]], name='weight')
      prediction = v * 2
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=prediction)

    est1 = estimator.EstimatorV2(model_fn=_model_fn)
    est1.train(dummy_input_fn, steps=1)
    est2 = estimator.EstimatorV2(model_fn=_model_fn, model_dir=est1.model_dir)
    self.assertEqual([32.],
                     next(
                         est2.predict(
                             dummy_input_fn,
                             checkpoint_path=est2.latest_checkpoint())))

  def test_scaffold_is_used(self):

    def _model_fn_scaffold(features, labels, mode):
      _, _ = features, labels
      tf.Variable(1., name='weight')
      self.mock_saver = get_mock_saver()
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          predictions=tf.constant([[1.]]),
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          scaffold=tf.train.Scaffold(saver=self.mock_saver))

    est = estimator.EstimatorV2(model_fn=_model_fn_scaffold)
    est.train(dummy_input_fn, steps=1)
    next(est.predict(dummy_input_fn))
    self.assertTrue(self.mock_saver.restore.called)

  def test_features_labels_mode(self):
    given_features = {'test-features': [[1], [1]]}
    given_labels = {'test-labels': [[1], [1]]}

    def _input_fn():
      return given_features, given_labels

    def _model_fn(features, labels, mode):
      self.features, self.labels, self.mode = features, labels, mode
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=tf.constant([[0.]]))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(_input_fn, steps=1)
    next(est.predict(_input_fn))
    self.assertEqual(given_features, self.features)
    self.assertIsNone(self.labels)
    self.assertEqual(ModeKeys.PREDICT, self.mode)

  def test_graph_initialization_global_step_and_random_seed(self):
    expected_random_seed = run_config.RunConfig().tf_random_seed

    def _model_fn(features, labels, mode):
      _, _, _ = features, labels, mode
      self.assertIsNotNone(tf.train.get_global_step())
      self.assertEqual(expected_random_seed, tf.get_default_graph().seed)
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=tf.constant([[0.]]))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    next(est.predict(dummy_input_fn))


def _model_fn_for_export_tests(features, labels, mode):
  _, _ = features, labels
  tf.Variable(1., name='weight')
  scores = tf.constant([3.])
  classes = tf.constant(['wumpus'])
  update_global_step = tf.assign_add(tf.train.get_global_step(), 1)
  with tf.control_dependencies([update_global_step]):
    train_op = tf.constant(2.)
  return model_fn_lib.EstimatorSpec(
      mode,
      predictions=tf.constant(10.),
      loss=tf.constant(1.),
      train_op=train_op,
      export_outputs={'test': export_lib.ClassificationOutput(scores, classes)})


def _x_y_input_fn():
  return ({
      'x': tf.constant([[1], [1]], name='feature_x'),
      'y': tf.constant([[2], [2]], name='feature_y')
  }, tf.constant([[1], [1]], name='truth'))


def _model_fn_with_x_y(features, labels, mode):
  _ = labels
  tf.Variable(1., name='weight')
  scores = tf.constant([3.])
  classes = tf.constant(['wumpus'])
  if mode == ModeKeys.PREDICT:
    tf.Variable(36., name='name_collision')
    return model_fn_lib.EstimatorSpec(
        mode,
        predictions=tf.constant(10.),
        export_outputs={
            'test': export_lib.ClassificationOutput(scores, classes)
        })
  else:
    prefix = 'eval_' if mode == ModeKeys.EVAL else ''

    multiplied = tf.math.multiply(
        features['x'], features['y'], name='{}multiplied'.format(prefix))
    mean = tf.keras.metrics.Mean(name='{}mean'.format(prefix))
    mean.update_state(features['x'] - features['y'])
    eval_metrics = {
        'mean1':
            mean,
        'mean2':
            tf.metrics.mean(
                features['x'] - features['y'], name='{}mean'.format(prefix))
    }
    tf.Variable(1., name='later_var')
    tf.Variable(3., name='name_collision')
    return model_fn_lib.EstimatorSpec(
        mode,
        predictions=multiplied,
        loss=tf.constant(1.),
        train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
        eval_metric_ops=eval_metrics)


def _model_fn_with_saveables_for_export_tests(features, labels, mode):
  _, _ = features, labels
  table = saver_test_utils.CheckpointedOp(name='v2')
  update_global_step = tf.assign_add(tf.train.get_global_step(), 1)
  with tf.control_dependencies([update_global_step]):
    train_op = table.insert('k1', 30.0)
  prediction = table.lookup('k1', 0.0)
  return model_fn_lib.EstimatorSpec(
      mode,
      predictions=prediction,
      loss=tf.constant(1.),
      train_op=train_op,
      export_outputs={
          'test': export_lib.PredictOutput({'prediction': prediction})
      })


def _get_serving_input_receiver_fn():
  feature_spec = {
      'x': tf.io.VarLenFeature(dtype=tf.dtypes.int64),
      'y': tf.io.VarLenFeature(dtype=tf.dtypes.int64)
  }
  return export_lib.build_parsing_serving_input_receiver_fn(feature_spec)


def _get_supervised_input_receiver_fn():
  return export_lib.build_supervised_input_receiver_fn_from_input_fn(
      _x_y_input_fn)


_VOCAB_FILE_CONTENT = 'emerson\nlake\npalmer\n'
_EXTRA_FILE_CONTENT = 'kermit\npiggy\nralph\n'


@test_util.run_v1_only('b/119219961')
class EstimatorExportTest(tf.test.TestCase):

  def test_export_saved_model_proto_roundtrip_raw_receiver(self):
    tmpdir = tempfile.mkdtemp()
    est = estimator.EstimatorV2(model_fn=_model_fn_for_export_tests)
    est.train(input_fn=dummy_input_fn, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('export'))
    serving_input_receiver_fn = _get_serving_input_receiver_fn()
    export_dir = est.export_saved_model(export_dir_base,
                                        serving_input_receiver_fn)

    # Check that all the files are in the right places.
    self.assertTrue(tf.gfile.Exists(export_dir_base))
    self._validate_exported_files(export_dir)

    # Restore, to validate that the export was well-formed.
    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        tf.saved_model.load(sess, [tf.saved_model.SERVING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('input_example_tensor' in graph_ops)
        self.assertTrue('ParseExample/ParseExampleV2' in graph_ops)
        self.assertTrue('weight' in graph_ops)

  def test_export_saved_model_train(self):
    self._test_export_saved_model_for_mode(_get_supervised_input_receiver_fn(),
                                           ModeKeys.TRAIN)

  def test_export_saved_model_eval(self):
    self._test_export_saved_model_for_mode(_get_supervised_input_receiver_fn(),
                                           ModeKeys.EVAL)

  def test_export_saved_model_predict(self):
    self._test_export_saved_model_for_mode(_get_serving_input_receiver_fn(),
                                           ModeKeys.PREDICT)

  def _test_export_saved_model_for_mode(self, input_receiver_fn, mode):
    tmpdir = tempfile.mkdtemp()
    est = estimator.EstimatorV2(model_fn=_model_fn_for_export_tests)
    est.train(input_fn=_x_y_input_fn, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('export'))
    export_dir = est.export_saved_model(
        export_dir_base, input_receiver_fn, experimental_mode=mode)

    # Check that all the files are in the right places.
    self.assertTrue(tf.gfile.Exists(export_dir_base))
    self._validate_exported_files(export_dir)

    # Restore, to validate that the export was well-formed.
    tag_set = export_lib.EXPORT_TAG_MAP[mode]
    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        tf.saved_model.load(sess, tag_set, export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertFalse('name_collision_1' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # Clean up.
    tf.gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_proto_roundtrip_receiver_map(self):
    input_receiver_fn_map = {ModeKeys.PREDICT: _get_serving_input_receiver_fn()}
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        tf.saved_model.load(sess, [tf.saved_model.SERVING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('input_example_tensor' in graph_ops)
        self.assertTrue('ParseExample/ParseExampleV2' in graph_ops)
        self.assertFalse('feature_x' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # Clean up.
    tf.gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_proto_roundtrip_train_only(self):
    input_receiver_fn_map = {
        ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        tf.saved_model.load(sess, [tf.saved_model.TRAINING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('multiplied' in graph_ops)
        self.assertTrue('mean/update_op' in graph_ops)
        self.assertFalse('eval_multiplied' in graph_ops)
        self.assertTrue('feature_x' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # Clean up.
    tf.gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_proto_roundtrip_eval_only(self):
    input_receiver_fn_map = {ModeKeys.EVAL: _get_supervised_input_receiver_fn()}
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        tf.saved_model.load(sess, [tag_constants.EVAL], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('eval_multiplied' in graph_ops)
        self.assertTrue('eval_mean/value' in graph_ops)
        self.assertFalse('multiplied' in graph_ops)
        self.assertTrue('feature_x' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # Clean up.
    tf.gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_proto_roundtrip_no_serving(self):
    input_receiver_fn_map = {
        ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
        ModeKeys.EVAL: _get_supervised_input_receiver_fn()
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        tf.saved_model.load(sess, [tf.saved_model.TRAINING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('multiplied' in graph_ops)
        self.assertFalse('eval_multiplied' in graph_ops)
        self.assertTrue('feature_x' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        tf.saved_model.load(sess, [tag_constants.EVAL], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('eval_multiplied' in graph_ops)
        self.assertFalse('multiplied' in graph_ops)
        self.assertTrue('feature_x' in graph_ops)
        self.assertTrue('feature_y' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # Clean up.
    tf.gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_proto_roundtrip_three_defs(self):
    input_receiver_fn_map = {
        ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
        ModeKeys.EVAL: _get_supervised_input_receiver_fn(),
        ModeKeys.PREDICT: _get_serving_input_receiver_fn()
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    # Restore, to validate that the export was well-formed.
    for tag_set in export_lib.EXPORT_TAG_MAP.values():
      with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:
          tf.saved_model.load(sess, tag_set, export_dir)
          graph_ops = [x.name for x in graph.get_operations()]
          self.assertTrue('global_step/Assign' in graph_ops)
          self.assertTrue('global_step/Initializer/zeros' in graph_ops)
          self.assertTrue('weight' in graph_ops)

    # Clean up.
    tf.gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_proto_roundtrip_all_vars(self):
    input_receiver_fn_map = {
        ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
        ModeKeys.PREDICT: _get_serving_input_receiver_fn()
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        tf.saved_model.load(sess, [tf.saved_model.TRAINING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('later_var' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        tf.saved_model.load(sess, [tf.saved_model.SERVING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertFalse('later_var' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # Clean up.
    tf.gfile.DeleteRecursively(tmpdir)

  def test_export_all_saved_models_name_collision(self):
    input_receiver_fn_map = {
        ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
        ModeKeys.PREDICT: _get_serving_input_receiver_fn()
    }
    export_dir, tmpdir = self._test_export_all_saved_models(
        input_receiver_fn_map)

    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        tf.saved_model.load(sess, [tf.saved_model.TRAINING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('name_collision' in graph_ops)
        self.assertFalse('name_collision_1' in graph_ops)
        collection_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.assertEqual(3, collection_vars[-1].eval())

    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        tf.saved_model.load(sess, [tf.saved_model.SERVING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('name_collision' in graph_ops)
        self.assertFalse('name_collision_1' in graph_ops)
        collection_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # This is a non-obvious detail: when we load the estimator spec
        # for predict, name_collision gets set to 36. However, we then restore
        # from checkpoint, which should overwrite that var and make it the 3
        # from training. In practice, this would not be a good way to write
        # a model_fn, but leaving this check in for now to ensure consistency
        # with what would happen given our current order of spec, then
        # checkpoint.
        self.assertEqual(3, collection_vars[-1].eval())

    # Clean up.
    tf.gfile.DeleteRecursively(tmpdir)

  def _test_export_all_saved_models(self, input_receiver_fn_map):
    tmpdir = tempfile.mkdtemp()
    est = estimator.EstimatorV2(model_fn=_model_fn_with_x_y)
    est.train(input_fn=_x_y_input_fn, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('export'))
    export_dir = est.experimental_export_all_saved_models(
        export_dir_base, input_receiver_fn_map)

    # Check that all the files are in the right places.
    self.assertTrue(tf.gfile.Exists(export_dir_base))

    self._validate_exported_files(export_dir)

    return export_dir, tmpdir

  def _validate_exported_files(self, export_dir):
    self.assertTrue(tf.gfile.Exists(export_dir))
    self.assertTrue(
        tf.gfile.Exists(
            os.path.join(
                tf.compat.as_bytes(export_dir),
                tf.compat.as_bytes('saved_model.pb'))))
    self.assertTrue(
        tf.gfile.Exists(
            os.path.join(
                tf.compat.as_bytes(export_dir),
                tf.compat.as_bytes('variables'))))
    self.assertTrue(
        tf.gfile.Exists(
            os.path.join(
                tf.compat.as_bytes(export_dir),
                tf.compat.as_bytes('variables/variables.index'))))
    self.assertTrue(
        tf.gfile.Exists(
            os.path.join(
                tf.compat.as_bytes(export_dir),
                tf.compat.as_bytes('variables/variables.data-00000-of-00001'))))

  def test_export_all_saved_models_var_not_found(self):
    input_receiver_fn_map = {
        ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
        ModeKeys.EVAL: _get_supervised_input_receiver_fn(),
        ModeKeys.PREDICT: _get_serving_input_receiver_fn()
    }

    def _model_fn_with_predict_only_vars(features, labels, mode):
      _, _ = features, labels
      if mode == ModeKeys.PREDICT:
        tf.Variable(1., name='only_in_predict')
      else:
        tf.Variable(1., name='otherwise')

      prediction = tf.constant(1.)
      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=prediction,
          loss=tf.constant(1.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          export_outputs={
              'test': export_lib.PredictOutput({'prediction': prediction})
          })

    tmpdir = tempfile.mkdtemp()
    est = estimator.EstimatorV2(model_fn=_model_fn_with_predict_only_vars)
    est.train(input_fn=_x_y_input_fn, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('export'))

    err_regex = r'Could not load all requested variables[\w\W]*infer'
    with self.assertRaisesRegexp(ValueError, err_regex):
      est.experimental_export_all_saved_models(export_dir_base,
                                               input_receiver_fn_map)

  def test_export_all_saved_models_metric_operation(self):
    """Ensures metrics ops.Operations can be expoerted (b/109740581)."""

    def _model_fn(features, labels, mode):
      del features, labels  # Unused
      metric_obj = tf.keras.metrics.Mean()
      metric_obj.update_state(tf.constant([0]))
      eval_metrics = {
          'metrics1': (tf.constant([0]), tf.no_op()),
          'metrics2': metric_obj,
      }
      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=tf.constant(10.),
          loss=tf.constant(1.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          eval_metric_ops=eval_metrics)

    tmpdir = tempfile.mkdtemp()
    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(input_fn=dummy_input_fn, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir),
        tf.compat.as_bytes('metric_operation_export'))

    input_receiver_fn_map = {ModeKeys.EVAL: _get_supervised_input_receiver_fn()}

    export_dir = est.experimental_export_all_saved_models(
        export_dir_base, input_receiver_fn_map)

    # Restore, to validate that the export was well-formed.
    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        meta_graph = tf.saved_model.load(sess, [tag_constants.EVAL], export_dir)
        sig_outputs = meta_graph.signature_def[ModeKeys.EVAL].outputs
        self.assertTrue(sig_outputs['metrics1/update_op'].name.startswith(
            'metric_op_wrapper'))
        self.assertTrue(sig_outputs['metrics2/update_op'].name.startswith(
            'metric_op_wrapper'))

  def test_export_saved_model_with_saveables_proto_roundtrip(self):
    tmpdir = tempfile.mkdtemp()
    est = estimator.EstimatorV2(
        model_fn=_model_fn_with_saveables_for_export_tests)
    est.train(input_fn=dummy_input_fn, steps=1)
    feature_spec = {
        'x': tf.io.VarLenFeature(dtype=tf.dtypes.int64),
        'y': tf.io.VarLenFeature(dtype=tf.dtypes.int64)
    }
    serving_input_receiver_fn = (
        export_lib.build_parsing_serving_input_receiver_fn(feature_spec))

    # Perform the export.
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('export'))
    export_dir = est.export_saved_model(export_dir_base,
                                        serving_input_receiver_fn)

    # Check that all the files are in the right places.
    self.assertTrue(tf.gfile.Exists(export_dir_base))
    self.assertTrue(tf.gfile.Exists(export_dir))
    self.assertTrue(
        tf.gfile.Exists(
            os.path.join(
                tf.compat.as_bytes(export_dir),
                tf.compat.as_bytes('saved_model.pb'))))
    self.assertTrue(
        tf.gfile.Exists(
            os.path.join(
                tf.compat.as_bytes(export_dir),
                tf.compat.as_bytes('variables'))))
    self.assertTrue(
        tf.gfile.Exists(
            os.path.join(
                tf.compat.as_bytes(export_dir),
                tf.compat.as_bytes('variables/variables.index'))))
    self.assertTrue(
        tf.gfile.Exists(
            os.path.join(
                tf.compat.as_bytes(export_dir),
                tf.compat.as_bytes('variables/variables.data-00000-of-00001'))))

    # Restore, to validate that the export was well-formed.
    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        tf.saved_model.load(sess, [tf.saved_model.SERVING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('input_example_tensor' in graph_ops)
        self.assertTrue('ParseExample/ParseExampleV2' in graph_ops)
        # The original saver is used to restore variables
        self.assertTrue('save/LookupTableImportV2' in graph_ops)

    # Clean up.
    tf.gfile.DeleteRecursively(tmpdir)

  def test_export_saved_model_assets(self):
    tmpdir = tempfile.mkdtemp()
    est = estimator.EstimatorV2(model_fn=_model_fn_for_export_tests)
    est.train(input_fn=dummy_input_fn, steps=1)
    feature_spec = {
        'x': tf.io.VarLenFeature(dtype=tf.dtypes.int64),
        'y': tf.io.VarLenFeature(dtype=tf.dtypes.int64)
    }
    serving_input_receiver_fn = (
        export_lib.build_parsing_serving_input_receiver_fn(feature_spec))

    # Create a fake asset.
    vocab_file_name = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('my_vocab_file'))
    vocab_file = tf.io.gfile.GFile(vocab_file_name, mode='w')
    vocab_file.write(_VOCAB_FILE_CONTENT)
    vocab_file.close()

    # hack in an op that uses the asset, in order to test asset export.
    # this is not actually valid, of course.
    def serving_input_receiver_with_asset_fn():
      features, receiver_tensor, _ = serving_input_receiver_fn()
      filename = ops.convert_to_tensor(
          vocab_file_name, tf.dtypes.string, name='asset_filepath')
      tf.add_to_collection(tf.compat.v1.GraphKeys.ASSET_FILEPATHS, filename)
      features['bogus_filename'] = filename

      return export_lib.ServingInputReceiver(features, receiver_tensor)

    # Perform the export.
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('export'))
    export_dir = est.export_saved_model(export_dir_base,
                                        serving_input_receiver_with_asset_fn)

    # Check that the asset files are in the right places.
    expected_vocab_file_name = os.path.join(
        tf.compat.as_bytes(export_dir),
        tf.compat.as_bytes('assets/my_vocab_file'))
    self.assertTrue(
        tf.gfile.Exists(
            os.path.join(
                tf.compat.as_bytes(export_dir), tf.compat.as_bytes('assets'))))
    self.assertTrue(tf.gfile.Exists(expected_vocab_file_name))
    self.assertEqual(
        tf.compat.as_bytes(_VOCAB_FILE_CONTENT),
        tf.compat.as_bytes(tf.io.gfile.GFile(expected_vocab_file_name).read()))

    # Restore, to validate that the export was well-formed.
    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        tf.saved_model.load(sess, [tf.saved_model.SERVING], export_dir)
        assets = [
            x.eval() for x in graph.get_collection(tf.GraphKeys.ASSET_FILEPATHS)
        ]
        self.assertItemsEqual([vocab_file_name], assets)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('input_example_tensor' in graph_ops)
        self.assertTrue('ParseExample/ParseExampleV2' in graph_ops)
        self.assertTrue('asset_filepath' in graph_ops)
        self.assertTrue('weight' in graph_ops)

    # cleanup
    tf.gfile.DeleteRecursively(tmpdir)

  def test_export_saved_model_extra_assets(self):
    tmpdir = tempfile.mkdtemp()
    est = estimator.EstimatorV2(model_fn=_model_fn_for_export_tests)
    est.train(input_fn=dummy_input_fn, steps=1)
    feature_spec = {
        'x': tf.io.VarLenFeature(dtype=tf.dtypes.int64),
        'y': tf.io.VarLenFeature(dtype=tf.dtypes.int64)
    }
    serving_input_receiver_fn = (
        export_lib.build_parsing_serving_input_receiver_fn(feature_spec))

    # Create a fake asset.
    extra_file_name = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('my_extra_file'))
    extra_file = tf.io.gfile.GFile(extra_file_name, mode='w')
    extra_file.write(_EXTRA_FILE_CONTENT)
    extra_file.close()

    # Perform the export.
    assets_extra = {'some/sub/directory/my_extra_file': extra_file_name}
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('export'))
    export_dir = est.export_saved_model(
        export_dir_base, serving_input_receiver_fn, assets_extra=assets_extra)

    # Check that the asset files are in the right places.
    expected_extra_path = os.path.join(
        tf.compat.as_bytes(export_dir),
        tf.compat.as_bytes('assets.extra/some/sub/directory/my_extra_file'))
    self.assertTrue(
        tf.gfile.Exists(
            os.path.join(
                tf.compat.as_bytes(export_dir),
                tf.compat.as_bytes('assets.extra'))))
    self.assertTrue(tf.gfile.Exists(expected_extra_path))
    self.assertEqual(
        tf.compat.as_bytes(_EXTRA_FILE_CONTENT),
        tf.compat.as_bytes(tf.io.gfile.GFile(expected_extra_path).read()))

    # cleanup
    tf.gfile.DeleteRecursively(tmpdir)

  def test_export_saved_model_tensor_features(self):
    """Test that models accepting a single raw Tensor can be exported.

    See https://github.com/tensorflow/tensorflow/issues/11674

    If the model_fn and receiver_fn accept raw tensors rather than dictionaries
    as input, export_saved_model should be okay with that, too.

    """

    tmpdir = tempfile.mkdtemp()

    def _input_fn_tensor_features():
      t = tf.constant([1, 2, 3], dtype=tf.dtypes.float32, shape=[1, 3])
      return (t, None)

    def _model_fn_tensor_features(features, labels, mode):
      _ = labels
      prediction = tf.linalg.matmul(features, features, transpose_b=True)

      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=prediction,
          loss=tf.constant(1.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          export_outputs={
              'test': export_lib.PredictOutput({'prediction': prediction})
          })

    def _serving_input_receiver_fn():
      feat = tf.placeholder(dtype=tf.dtypes.float32)
      return export_lib.TensorServingInputReceiver(
          features=feat, receiver_tensors=feat)

    est = estimator.EstimatorV2(model_fn=_model_fn_tensor_features)
    est.train(input_fn=_input_fn_tensor_features, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('export'))
    export_dir = est.export_saved_model(export_dir_base,
                                        _serving_input_receiver_fn)

    # Restore, to validate that the export was well-formed.
    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        tf.saved_model.load(sess, [tf.saved_model.SERVING], export_dir)
        graph_ops = [x.name.lower() for x in graph.get_operations()]
        self.assertTrue('const' in graph_ops)
        self.assertTrue('matmul' in graph_ops)

    # Clean up.
    tf.gfile.DeleteRecursively(tmpdir)

  def test_export_saved_model_int_feature_keys(self):
    """Test that the `features` dict can contain int keys."""
    tmpdir = tempfile.mkdtemp()

    def _input_fn_with_int_keys():
      features = {
          'string_key': tf.constant([1], dtype=tf.dtypes.float32),
          42: tf.constant([43], dtype=tf.dtypes.float32),
      }
      return (features, None)

    def _model_fn_with_int_keys(features, labels, mode):
      _ = labels
      prediction = tf.math.maximum(features['string_key'], features[42])

      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=prediction,
          loss=tf.constant(1.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          export_outputs={
              'test': export_lib.PredictOutput({'prediction': prediction})
          })

    def _serving_input_receiver_fn():
      features = {
          'string_key': tf.placeholder(dtype=tf.dtypes.float32),
          42: tf.placeholder(dtype=tf.dtypes.float32, name='42_placeholder'),
      }
      # int is only allowed in the `features` dict, not the `receiver_tensors`.
      receiver_tensors = {
          'string_key': features['string_key'],
          '42_key': features[42],
      }
      return export_lib.ServingInputReceiver(
          features=features, receiver_tensors=receiver_tensors)

    est = estimator.EstimatorV2(model_fn=_model_fn_with_int_keys)
    est.train(input_fn=_input_fn_with_int_keys, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('export'))
    export_dir = est.export_saved_model(export_dir_base,
                                        _serving_input_receiver_fn)

    # Restore, to validate that the export was well-formed.
    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        meta_graph_def = tf.saved_model.load(sess, [tf.saved_model.SERVING],
                                             export_dir)
        graph_ops = [x.name.lower() for x in graph.get_operations()]
        self.assertTrue('maximum' in graph_ops)
        self.assertTrue('42_placeholder' in graph_ops)
        self.assertTrue(
            '42_key' in meta_graph_def.signature_def['serving_default'].inputs)

    # Clean up.
    tf.gfile.DeleteRecursively(tmpdir)

  def test_scaffold_is_used_for_saver(self):
    tmpdir = tempfile.mkdtemp()

    def _model_fn_scaffold(features, labels, mode):
      _, _ = features, labels
      tf.Variable(1., name='weight')
      self.mock_saver = get_mock_saver()
      scores = tf.constant([3.])
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          predictions=tf.constant([[1.]]),
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          scaffold=tf.train.Scaffold(saver=self.mock_saver),
          export_outputs={'test': export_lib.ClassificationOutput(scores)})

    est = estimator.EstimatorV2(model_fn=_model_fn_scaffold)
    est.train(dummy_input_fn, steps=1)
    feature_spec = {
        'x': tf.io.VarLenFeature(dtype=tf.dtypes.int64),
        'y': tf.io.VarLenFeature(dtype=tf.dtypes.int64)
    }
    serving_input_receiver_fn = (
        export_lib.build_parsing_serving_input_receiver_fn(feature_spec))

    # Perform the export.
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('export'))
    est.export_saved_model(export_dir_base, serving_input_receiver_fn)

    self.assertTrue(self.mock_saver.restore.called)
    self.assertTrue(self.mock_saver.export_meta_graph.called)
    self.assertTrue(self.mock_saver.save.called)

  def test_scaffold_is_used_for_saver_multiple_modes(self):
    tmpdir = tempfile.mkdtemp()
    savers = {'predict_saver': None, 'train_saver': None}

    def _model_fn_scaffold(features, labels, mode):
      _, _ = features, labels
      tf.Variable(1., name='weight')

      scores = tf.constant([3.])
      if mode == ModeKeys.PREDICT:
        savers['predict_saver'] = get_mock_saver()
        scaffold = tf.train.Scaffold(saver=savers['predict_saver'])
      elif mode == ModeKeys.TRAIN:
        savers['train_saver'] = get_mock_saver()
        scaffold = tf.train.Scaffold(saver=savers['train_saver'])
      else:
        scaffold = tf.train.Scaffold()
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          predictions=tf.constant([[1.]]),
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          scaffold=scaffold,
          export_outputs={'test': export_lib.ClassificationOutput(scores)})

    est = estimator.EstimatorV2(model_fn=_model_fn_scaffold)
    est.train(dummy_input_fn, steps=1)
    input_receiver_fn_map = {
        ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
        ModeKeys.EVAL: _get_supervised_input_receiver_fn(),
        ModeKeys.PREDICT: _get_serving_input_receiver_fn()
    }

    # Perform the export.
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('export'))
    est.experimental_export_all_saved_models(export_dir_base,
                                             input_receiver_fn_map)

    self.assertTrue(savers['train_saver'].restore.called)
    self.assertEqual(savers['train_saver'].export_meta_graph.call_count, 1)
    self.assertEqual(savers['train_saver'].save.call_count, 1)

    self.assertTrue(savers['predict_saver'].restore.called)
    self.assertEqual(savers['predict_saver'].export_meta_graph.call_count, 1)
    self.assertEqual(savers['predict_saver'].save.call_count, 0)

  def test_scaffold_is_used_for_local_init(self):
    tmpdir = tempfile.mkdtemp()

    def _model_fn_scaffold(features, labels, mode):
      _, _ = features, labels
      my_int = tf.Variable(
          1, name='my_int', collections=[tf.GraphKeys.LOCAL_VARIABLES])
      _ = training.get_or_create_steps_per_run_variable()
      scores = tf.constant([3.])
      with tf.control_dependencies([
          tf.initializers.local_variables(),
          tf.initializers.tables_initializer()
      ]):
        assign_op = tf.assign(my_int, 12345)

      # local_initSop must be an Operation, not a Tensor.
      custom_local_init_op = tf.group(assign_op)
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          predictions=tf.constant([[1.]]),
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          scaffold=tf.train.Scaffold(local_init_op=custom_local_init_op),
          export_outputs={'test': export_lib.ClassificationOutput(scores)})

    est = estimator.EstimatorV2(model_fn=_model_fn_scaffold)
    est.train(dummy_input_fn, steps=1)
    feature_spec = {
        'x': tf.io.VarLenFeature(dtype=tf.dtypes.int64),
        'y': tf.io.VarLenFeature(dtype=tf.dtypes.int64)
    }
    serving_input_receiver_fn = (
        export_lib.build_parsing_serving_input_receiver_fn(feature_spec))

    # Perform the export.
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('export'))
    export_dir = est.export_saved_model(export_dir_base,
                                        serving_input_receiver_fn)

    # Restore, to validate that the custom local_init_op runs.
    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        tf.saved_model.load(sess, [tf.saved_model.SERVING], export_dir)
        my_int = graph.get_tensor_by_name('my_int:0')
        my_int_value = sess.run(my_int)
        self.assertEqual(12345, my_int_value)

  def test_scaffold_is_used_for_local_init_multiple_modes(self):
    tmpdir = tempfile.mkdtemp()

    def _model_fn_scaffold(features, labels, mode):
      _, _ = features, labels
      my_int = tf.Variable(
          1, name='my_int', collections=[tf.GraphKeys.LOCAL_VARIABLES])
      scores = tf.constant([3.])
      with tf.control_dependencies([
          tf.initializers.local_variables(),
          tf.initializers.tables_initializer()
      ]):
        assign_op = tf.assign(my_int, 12345)

      custom_local_init_op = None
      if mode == ModeKeys.PREDICT:
        # local_initSop must be an Operation, not a Tensor.
        custom_local_init_op = tf.group(assign_op)

      return model_fn_lib.EstimatorSpec(
          mode=mode,
          predictions=tf.constant([[1.]]),
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          scaffold=tf.train.Scaffold(local_init_op=custom_local_init_op),
          export_outputs={'test': export_lib.ClassificationOutput(scores)})

    est = estimator.EstimatorV2(model_fn=_model_fn_scaffold)
    est.train(dummy_input_fn, steps=1)
    input_receiver_fn_map = {
        ModeKeys.TRAIN: _get_supervised_input_receiver_fn(),
        ModeKeys.EVAL: _get_supervised_input_receiver_fn(),
        ModeKeys.PREDICT: _get_serving_input_receiver_fn()
    }

    # Perform the export.
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('export'))
    export_dir = est.experimental_export_all_saved_models(
        export_dir_base, input_receiver_fn_map)

    # Restore, to validate that the custom local_init_op runs.
    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        tf.saved_model.load(sess, [tf.saved_model.SERVING], export_dir)
        my_int = graph.get_tensor_by_name('my_int:0')
        my_int_value = sess.run(my_int)
        self.assertEqual(12345, my_int_value)
    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        tf.saved_model.load(sess, [tf.saved_model.TRAINING], export_dir)
        my_int = graph.get_tensor_by_name('my_int:0')
        my_int_value = sess.run(my_int)
        self.assertEqual(1, my_int_value)

  def test_features_labels_mode(self):
    given_features = {'test-features': tf.constant([[1], [1]])}

    def serving_input_receiver_fn():
      return export_lib.ServingInputReceiver(
          given_features, tf.placeholder(dtype=tf.dtypes.string))

    def _model_fn(features, labels, mode):
      self.features, self.labels, self.mode = features, labels, mode
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=tf.constant([[0.]]),
          export_outputs={
              'test': export_lib.ClassificationOutput(tf.constant([[0.]]))
          })

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    est.export_saved_model(tempfile.mkdtemp(), serving_input_receiver_fn)
    self.assertEqual(given_features, self.features)
    self.assertIsNone(self.labels)
    self.assertEqual(ModeKeys.PREDICT, self.mode)

  def test_graph_initialization_global_step_and_random_seed(self):
    expected_random_seed = run_config.RunConfig().tf_random_seed

    def _model_fn(features, labels, mode):
      _, _, _ = features, labels, mode
      self.assertIsNotNone(tf.train.get_global_step())
      self.assertEqual(expected_random_seed, tf.get_default_graph().seed)
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=tf.constant(0.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1),
          predictions=tf.constant([[0.]]),
          export_outputs={
              'test': export_lib.ClassificationOutput(tf.constant([[0.]]))
          })

    def serving_input_receiver_fn():
      return export_lib.ServingInputReceiver(
          {'test-features': tf.constant([[1], [1]])},
          tf.placeholder(dtype=tf.dtypes.string))

    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(dummy_input_fn, steps=1)
    est.export_saved_model(tempfile.mkdtemp(), serving_input_receiver_fn)

  def test_export_saved_model_respects_soft_placement(self):

    def model_fn_with_a_gpu_op_but_no_kernel(features, labels, mode):
      _, _ = features, labels
      table = saver_test_utils.CheckpointedOp(name='v2')

      update_global_step = tf.assign_add(tf.train.get_global_step(), 1)
      with tf.control_dependencies([update_global_step]):
        train_op = table.insert('k1', 30.0)

      #  In this test, there are no GPUs available.  The goal is to verify that
      #  export_saved_model executes nevertheless.
      with tf.device('/gpu:0'):
        string_op = tf.strings.as_string(update_global_step)

      with tf.control_dependencies([string_op]):
        prediction = table.lookup('k1', 0.0)

      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=prediction,
          loss=tf.constant(1.),
          train_op=train_op,
          export_outputs={
              'test': export_lib.PredictOutput({'prediction': prediction})
          })

    tmpdir = tempfile.mkdtemp()
    est = estimator.EstimatorV2(model_fn=model_fn_with_a_gpu_op_but_no_kernel)
    est.train(input_fn=dummy_input_fn, steps=1)
    feature_spec = {
        'x': tf.io.VarLenFeature(dtype=tf.dtypes.int64),
        'y': tf.io.VarLenFeature(dtype=tf.dtypes.int64)
    }
    serving_input_receiver_fn = (
        export_lib.build_parsing_serving_input_receiver_fn(feature_spec))
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('export'))

    export_dir = est.export_saved_model(export_dir_base,
                                        serving_input_receiver_fn)

    # At this point, if export_saved_model executed with
    # allow_soft_placement=True, then the GPU-assigned operation was silently
    # placed on the CPU.  Otherwise, an exception would have been raised
    # related to the fact that the requested GPU device isn't available.

    # Expectations below assume that export_saved_model has completed normally.
    self.assertTrue(tf.gfile.Exists(export_dir_base))
    self.assertTrue(tf.gfile.Exists(export_dir))
    self.assertTrue(
        tf.gfile.Exists(
            os.path.join(
                tf.compat.as_bytes(export_dir),
                tf.compat.as_bytes('saved_model.pb'))))
    self.assertTrue(
        tf.gfile.Exists(
            os.path.join(
                tf.compat.as_bytes(export_dir),
                tf.compat.as_bytes('variables'))))
    self.assertTrue(
        tf.gfile.Exists(
            os.path.join(
                tf.compat.as_bytes(export_dir),
                tf.compat.as_bytes('variables/variables.index'))))
    self.assertTrue(
        tf.gfile.Exists(
            os.path.join(
                tf.compat.as_bytes(export_dir),
                tf.compat.as_bytes('variables/variables.data-00000-of-00001'))))

    tf.gfile.DeleteRecursively(tmpdir)

  def _validate_strip_default_attrs(self, estimator_cls, export_fn,
                                    attributes_stripped):
    """Validate estimator export correctly strips/leaves default attributes.

    Args:
      estimator_cls: `Estimator` or `EstimatorV2`
      export_fn: a function that takes in an estimator and export arguments, and
        exports the estimator.
      attributes_stripped: whether to attributes are expected to be stripped in
        the MetaGraphDef.
    """
    est = estimator_cls(model_fn=_model_fn_for_export_tests)
    est.train(input_fn=dummy_input_fn, steps=1)
    feature_spec = {
        'x': tf.io.VarLenFeature(dtype=tf.dtypes.int64),
        'y': tf.io.VarLenFeature(dtype=tf.dtypes.int64)
    }
    serving_input_receiver_fn = (
        export_lib.build_parsing_serving_input_receiver_fn(feature_spec))

    # Perform the export, and obtain the MetaGraphDefs
    tmpdir = tempfile.mkdtemp()
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('export'))

    export_dir = export_fn(est, export_dir_base, serving_input_receiver_fn)
    saved_model_pb = loader_impl._parse_saved_model(export_dir)
    self.assertIsNotNone(saved_model_pb)
    meta_graph_def = [
        x for x in saved_model_pb.meta_graphs
        if x.meta_info_def.tags == [tf.saved_model.SERVING]
    ][0]

    # "weight" node in graph is a "Variable" Op with 2 default valued attrs.
    #   o "container"    : "".
    #   o "shared_name"  : "".

    # When default attributes are not stripped, the "weight" node should have
    # attributes "container" and "shared_name". When default attributes are
    # stripped, the node should not have these attributes.
    node_def = test_util.get_node_def_from_graph('weight',
                                                 meta_graph_def.graph_def)
    self.assertEqual(attributes_stripped, 'container' not in node_def.attr)
    self.assertEqual(attributes_stripped, 'shared_name' not in node_def.attr)

    # Clean up.
    tf.gfile.DeleteRecursively(tmpdir)

  def test_export_saved_model_proto_strip_default_attrs(self):
    # Test deprecated export_savedmodel to ensure that V1 behavior is consistent
    self._validate_strip_default_attrs(
        estimator.Estimator,
        lambda e, *args: e.export_savedmodel(*args, strip_default_attrs=True),
        True)
    self._validate_strip_default_attrs(
        estimator.Estimator,
        lambda e, *args: e.export_savedmodel(*args, strip_default_attrs=False),
        False)

    # Make sure that export_saved_model strips the default attributes.
    self._validate_strip_default_attrs(
        estimator.Estimator, lambda e, *args: e.export_saved_model(*args), True)
    self._validate_strip_default_attrs(
        estimator.EstimatorV2, lambda e, *args: e.export_saved_model(*args),
        True)

  def test_export_saved_model_no_export_outputs(self):
    """Ensure that an EstimatorSpec without outputs defined can be exported."""

    def _model_fn(features, labels, mode):
      _, _ = features, labels
      tf.Variable(1., name='weight')
      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=tf.constant(10.),
          loss=tf.constant(1.),
          train_op=tf.assign_add(tf.compat.v1.train.get_global_step(), 1))

    tmpdir = tempfile.mkdtemp()
    est = estimator.EstimatorV2(model_fn=_model_fn)
    est.train(input_fn=dummy_input_fn, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        tf.compat.as_bytes(tmpdir), tf.compat.as_bytes('no_export_outputs'))
    export_dir = est.export_saved_model(export_dir_base,
                                        _get_serving_input_receiver_fn())

    # Check that all the files are in the right places.
    self.assertTrue(tf.gfile.Exists(export_dir_base))
    self._validate_exported_files(export_dir)

    # Restore, to validate that the export was well-formed.
    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        meta_graph = tf.saved_model.load(sess, [tf.saved_model.SERVING],
                                         export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertTrue('weight' in graph_ops)

        sig_def = meta_graph.signature_def
        self.assertEqual(len(sig_def), 1)
        sig_outputs = sig_def[
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs
        self.assertEqual(sig_outputs['output'].name, 'Const:0')

  def test_export_from_warm_start(self):

    def _make_model_fn(x):

      def _variable_creating_model_fn(features, labels, mode):
        _, _ = features, labels
        tf.get_variable('x', initializer=x)
        global_step = tf.train.get_global_step()
        return model_fn_lib.EstimatorSpec(
            mode,
            predictions=tf.constant(1.),
            loss=tf.constant(1.),
            train_op=tf.assign_add(global_step, 1))

      return _variable_creating_model_fn

    est = estimator.EstimatorV2(model_fn=_make_model_fn(42.))
    est.train(dummy_input_fn, steps=10)

    warm_started_est = estimator.EstimatorV2(
        model_fn=_make_model_fn(36.), warm_start_from=est.model_dir)
    saved_model_dir = warm_started_est.export_saved_model(
        tempfile.mkdtemp(), _get_serving_input_receiver_fn())
    variable_dir = saved_model_utils.get_variables_path(saved_model_dir)
    self.assertEqual(42., tf.train.load_variable(variable_dir, 'x'))

  def test_export_saved_model_symbol_deprecated(self):
    est = estimator.EstimatorV2(model_fn=_model_fn_for_export_tests)
    with self.assertRaisesRegexp(AttributeError,
                                 'Please use `export_saved_model`'):
      est.export_savedmodel


class EstimatorHookOrderingTest(tf.test.TestCase):

  def testCustomHooksAreCalledBeforeNanTensorHook(self):

    def nan_making_model_fn(mode, features, labels):
      """A graph that generates NaN's for testing."""
      del features, labels

      global_step = tf.Variable(0, dtype=tf.dtypes.int64, name='global_step')
      inc_global_step = tf.assign_add(global_step, 1)
      nan_const = tf.constant(np.nan, dtype=tf.dtypes.float32)
      loss = tf.cond(inc_global_step > 1, lambda: nan_const, lambda: 1.0)

      return model_fn_lib.EstimatorSpec(
          mode=mode,
          predictions=global_step.read_value(),
          loss=loss,
          train_op=inc_global_step)

    def empty_input_fn():
      return dict(), None

    class AfterRunCountingHook(tf.train.SessionRunHook):
      """Hooks that counts the number of times after_run() is called."""

      def __init__(self):
        self.after_run_count = 0

      def after_run(self, run_context, run_values):
        del run_context, run_values
        self.after_run_count += 1

    test_hook = AfterRunCountingHook()
    est = estimator.EstimatorV2(model_fn=nan_making_model_fn)
    with self.assertRaises(tf.train.NanLossDuringTrainingError):
      est.train(input_fn=empty_input_fn, steps=2, hooks=[test_hook])
    self.assertEqual(2, test_hook.after_run_count)


class EstimatorIntegrationTest(tf.test.TestCase):

  def test_complete_flow_with_a_simple_linear_model(self):

    def _model_fn(features, labels, mode):
      predictions = tf.layers.dense(
          features['x'], 1, kernel_initializer=tf.initializers.zeros())
      export_outputs = {'predictions': export_lib.RegressionOutput(predictions)}

      if mode == ModeKeys.PREDICT:
        return model_fn_lib.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)

      loss = tf.keras.losses.MeanSquaredError()(labels, predictions)
      train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(
          loss, tf.train.get_global_step())
      mean = tf.keras.metrics.Mean()
      mean.update_state(loss)
      eval_metric_ops = {
          'absolute_error': tf.metrics.mean_absolute_error(labels, predictions),
          'mean': mean,
      }

      return model_fn_lib.EstimatorSpec(
          mode,
          predictions=predictions,
          loss=loss,
          train_op=train_op,
          eval_metric_ops=eval_metric_ops,
          export_outputs=export_outputs)

    est = estimator.EstimatorV2(model_fn=_model_fn)
    data = np.linspace(0., 1., 100, dtype=np.float32).reshape(-1, 1)

    # TRAIN
    # learn y = x
    train_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, y=data, batch_size=50, num_epochs=None, shuffle=True)
    est.train(train_input_fn, steps=200)

    # EVALUATE
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, y=data, batch_size=50, num_epochs=1, shuffle=True)
    scores = est.evaluate(eval_input_fn)
    self.assertEqual(200, scores['global_step'])
    self.assertGreater(0.1, scores['absolute_error'])
    self.assertAlmostEqual(4.4e-14, scores['mean'], places=2)

    # PREDICT
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data}, y=None, batch_size=10, num_epochs=1, shuffle=False)
    predictions = list(est.predict(predict_input_fn))
    self.assertAllClose(data, predictions, atol=0.01)

    # EXPORT
    feature_spec = {'x': tf.io.FixedLenFeature([1], tf.dtypes.float32)}
    serving_input_receiver_fn = (
        export_lib.build_parsing_serving_input_receiver_fn(feature_spec))
    export_dir = est.export_saved_model(tempfile.mkdtemp(),
                                        serving_input_receiver_fn)
    self.assertTrue(tf.gfile.Exists(export_dir))


class EstimatorInputContextTest(tf.test.TestCase):

  def test_with_input_fn(self):
    total_batch_size = 10
    num_shards = 2

    def _input_with_context(input_context):
      batch_size = total_batch_size // num_shards
      self.assertEqual('DummyInputContext', input_context.name)
      self.assertEqual(batch_size, input_context.batch_size)
      return tf.data.Dataset.from_tensors(([1.], [2.]))

    def _input_without_context():
      return tf.data.Dataset.from_tensors(([1.], [2.]))

    class DummyInputContext(object):

      def __init__(self, n_shards, total_bs):
        self._name = 'DummyInputContext'
        self._num_shards = n_shards
        self._total_batch_size = total_bs

      @property
      def name(self):
        return self._name

      @property
      def batch_size(self):
        return self._total_batch_size // self._num_shards

    # This class is the mock for DistributionStrategy. It only overrides
    # the make_input_fn_iterator method.
    class DummyDistributionStrategy(object):

      def __init__(self, n_shards):
        self._num_shards = n_shards

      def make_input_fn_iterator(self, input_fn):
        input_context = DummyInputContext(num_shards, total_batch_size)
        return input_fn(input_context)

    distribution = DummyDistributionStrategy(num_shards)
    est = estimator.EstimatorV2(model_fn=dummy_model_fn)
    # We only test the `input_fn` instead of calling `Estimator.train`
    est._get_iterator_from_input_fn(_input_with_context, None, distribution)  # pylint: disable=protected-access
    est._get_iterator_from_input_fn(_input_without_context, None, distribution)  # pylint: disable=protected-access


if __name__ == '__main__':
  tf.test.main()
