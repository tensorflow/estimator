# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TPUEstimator export related functionalities."""

from absl import flags
from absl.testing import parameterized
import numpy as np
import os
import tempfile
import tensorflow.compat.v1 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.example import example_pb2
from tensorflow.python import data as dataset_lib
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import training
from tensorflow.python.util import compat
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.tpu import tpu_config
from tensorflow_estimator.python.estimator.tpu import tpu_estimator
from tensorflow_estimator.python.estimator.util import tf_keras_v1
# pylint: enable=g-direct-tensorflow-import

flags.DEFINE_integer('test_num_shards', 8, 'number of replicas to test')


FLAGS = flags.FLAGS

_TRAIN = model_fn_lib.ModeKeys.TRAIN
_EVAL = model_fn_lib.ModeKeys.EVAL
_PREDICT = model_fn_lib.ModeKeys.PREDICT

_PER_HOST = 'per_host_sharding'
_PER_SHARD = 'per_shard_sharding'
_UNSHARDED = 'unsharded'
_INPUT_PIPELINE_WITH_QUEUE_RUNNER = (
    'Input pipeline contains one or more QueueRunners')


def dense_computation(features):
  return tf_keras_v1.__internal__.legacy.layers.dense(
      features['x'], 1, kernel_initializer=init_ops.zeros_initializer())


def get_model_fn(export_tpu_tensor=True, export_cpu_tensor=False,
                 tpu_estimator_spec=True):

  def model_fn(features, labels, mode, params):
    del params
    loss = None
    train_op = None
    predictions = dense_computation(features)
    export_outputs = None
    if mode != _PREDICT:
      loss = losses.mean_squared_error(labels, predictions)
      optimizer = tf.tpu.CrossShardOptimizer(
          training.GradientDescentOptimizer(learning_rate=0.5))
      train_op = optimizer.minimize(loss, training.get_global_step())
    else:
      if export_tpu_tensor:
        key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        export_outputs = {
            key: export_lib.PredictOutput({
                'prediction': predictions
            })
        }
      else:
        export_outputs = {}

      if export_cpu_tensor:

        def host_call(predictions):
          return string_ops.as_string(predictions, name='classes')

        classes = tf.tpu.outside_compilation(host_call, predictions)
        classification_output = export_lib.ClassificationOutput(
            classes=classes)
        export_outputs['classification'] = classification_output

    if tpu_estimator_spec:
      spec_type = tpu_estimator.TPUEstimatorSpec
    else:
      spec_type = model_fn_lib.EstimatorSpec

    return spec_type(
        mode,
        loss=loss,
        train_op=train_op,
        predictions={'predictions': predictions},
        export_outputs=export_outputs)

  return model_fn


def dummy_input_fn_with_dataset(batch_size, repeat=True, x=None):
  if x is None:
    x = np.random.normal(size=[batch_size, 1]).astype(np.float32)
  labels = [[2.0]] * batch_size

  dataset1 = dataset_lib.Dataset.from_tensor_slices(x)
  dataset2 = dataset_lib.Dataset.from_tensor_slices(labels)
  dataset = dataset_lib.Dataset.zip((dataset1, dataset2))
  if repeat:
    dataset = dataset.repeat()
  dataset = dataset.batch(batch_size, drop_remainder=True)

  def _map(x, y):
    return {'x': x}, y

  return dataset.map(_map)


def dummy_input_fn(batch_size, repeat=True):
  dataset = dummy_input_fn_with_dataset(batch_size, repeat)
  iterator = dataset_ops.make_one_shot_iterator(dataset)
  return iterator.get_next()


def create_run_config(iterations_per_loop, **kwargs):
  if 'num_shards' not in kwargs:
    kwargs['num_shards'] = FLAGS.test_num_shards
  return tpu_config.RunConfig(
      master='',
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=iterations_per_loop, **kwargs),
  )


class TPUEstimatorExportTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    feature_spec = {'x': parsing_ops.FixedLenFeature([1], dtypes.float32)}
    self._serving_input_receiver_fn = (
        export_lib.build_parsing_serving_input_receiver_fn(feature_spec))

    feature_spec = {
        'x':
            array_ops.placeholder(dtype=dtypes.float32, shape=(2, 1), name='x'),
    }
    label_spec = array_ops.placeholder(
        dtype=dtypes.float32, shape=(1, 1), name='truth')
    self._supervised_input_receiver_fn = (
        export_lib.build_raw_supervised_input_receiver_fn(
            feature_spec, label_spec))

  @parameterized.parameters(
      (True, False, False),
      (True, True, False),
      (False, True, False),
      (True, False, True),
      (True, True, True),
      (False, True, True))
  def test_export_tpu_savedmodel_e2e(self, export_tpu_tensor, export_cpu_tensor,
                                     use_export_mode_v2):
    tmpdir = tempfile.mkdtemp()

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    model_fn = get_model_fn(export_tpu_tensor, export_cpu_tensor)
    run_config = create_run_config(iterations_per_loop=4)
    if use_export_mode_v2:
      export_api_version = tpu_estimator.ExportSavedModelApiVersion.V2

      batch_config = tpu_estimator.BatchConfig(
          num_batch_threads=1,
          max_batch_size=1,
          batch_timeout_micros=100,
          allowed_batch_sizes=[1])

      def tpu_model_fn(features, labels, mode, params):
        if mode == _PREDICT and params['use_tpu']:
          return tpu_estimator.model_fn_inference_on_tpu(
              model_fn, features, labels, mode, params, batch_config)
        else:
          return model_fn(features, labels, mode, params)

      est_model_fn = tpu_model_fn
    else:
      export_api_version = tpu_estimator.ExportSavedModelApiVersion.V1
      est_model_fn = model_fn
    est = tpu_estimator.TPUEstimator(
        model_fn=est_model_fn,
        config=run_config,
        train_batch_size=16,
        export_to_tpu=True,
        export_saved_model_api_version=export_api_version)
    est.train(_input_fn, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    export_dir = est.export_saved_model(export_dir_base,
                                        self._serving_input_receiver_fn)

    self._validate_export(export_dir_base, export_dir, export_tpu_tensor,
                          export_cpu_tensor)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def _validate_export(self, export_dir_base, export_dir, export_tpu_tensor,
                       export_cpu_tensor):
    # Check that all the files are in the right places.
    self.assertTrue(gfile.Exists(export_dir_base))
    self.assertTrue(gfile.Exists(export_dir))
    self.assertTrue(
        gfile.Exists(
            os.path.join(
                compat.as_bytes(export_dir),
                compat.as_bytes('saved_model.pb'))))
    self.assertTrue(
        gfile.Exists(
            os.path.join(
                compat.as_bytes(export_dir), compat.as_bytes('variables'))))
    self.assertTrue(
        gfile.Exists(
            os.path.join(
                compat.as_bytes(export_dir),
                compat.as_bytes('variables/variables.index'))))
    self.assertTrue(
        gfile.Exists(
            os.path.join(
                compat.as_bytes(export_dir),
                compat.as_bytes('variables/variables.data-00000-of-00001'))))

    def session_run():
      example = example_pb2.Example()
      example.features.feature['x'].float_list.value.append(1)

      tensor_name_prediction = None
      tensor_name_classes = None
      if export_tpu_tensor:
        key_prediction = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        tensor_name_prediction = (
            meta_graph.signature_def[key_prediction].
            outputs['prediction'].name)
        tensor_name_input = (meta_graph.signature_def[key_prediction].
                             inputs['examples'].name)

      if export_cpu_tensor:
        key_classification = 'classification'
        tensor_name_classes = (meta_graph.signature_def[key_classification].
                               outputs['classes'].name)
        tensor_name_input = (meta_graph.signature_def[key_classification].
                             inputs['inputs'].name)

      if export_tpu_tensor:
        sess.run(
            tensor_name_prediction,
            feed_dict={tensor_name_input: [example.SerializeToString()]})
      if export_cpu_tensor:
        sess.run(
            tensor_name_classes,
            feed_dict={tensor_name_input: [example.SerializeToString()]})
      if export_cpu_tensor and export_tpu_tensor:
        sess.run(
            [tensor_name_prediction, tensor_name_classes],
            feed_dict={tensor_name_input: [example.SerializeToString()]})

    # Restore, to validate that the export was well-formed.
    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        meta_graph = loader.load(
            sess, [tag_constants.SERVING, tag_constants.TPU], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertIn('input_example_tensor', graph_ops)
        self.assertIn('ParseExample/ParseExampleV2', graph_ops)
        self.assertNotIn('dense/kernel/GuaranteeConst', graph_ops)

        sess.run(tf.tpu.initialize_system())
        session_run()

    # Restore, to validate that the export was well-formed.
    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        meta_graph = loader.load(sess, [tag_constants.SERVING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertIn('input_example_tensor', graph_ops)
        self.assertIn('ParseExample/ParseExampleV2', graph_ops)
        self.assertIn('dense/kernel', graph_ops)
        # GuaranteeConst ops won't be present in the CPU-only graph.
        self.assertNotIn('dense/kernel/GuaranteeConst', graph_ops)

        session_run()

  def test_export_tpu_savedmodel_export_to_cpu_false(self):
    # Test that when `export_to_cpu` is `False`, CPU metagraph is not exported.
    tmpdir = tempfile.mkdtemp()

    model_fn = get_model_fn(export_tpu_tensor=True,
                            export_cpu_tensor=True)
    run_config = create_run_config(iterations_per_loop=4)

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    est = tpu_estimator.TPUEstimator(
        model_fn=model_fn, config=run_config, train_batch_size=16,
        export_to_tpu=True, export_to_cpu=False)
    est.train(_input_fn, steps=1)

    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export_no_tpu'))
    export_dir = est.export_saved_model(export_dir_base,
                                        self._serving_input_receiver_fn)
    saved_model = loader_impl.parse_saved_model(export_dir)
    self.assertLen(saved_model.meta_graphs, 1)
    tags = set(saved_model.meta_graphs[0].meta_info_def.tags)
    self.assertEqual(tags, set([tag_constants.SERVING, tag_constants.TPU]))

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_tpu_savedmodel_export_to_tpu_false(self):
    # Test that when `export_to_tpu` is `False`, TPU metagraph is not exported.
    tmpdir = tempfile.mkdtemp()

    model_fn = get_model_fn(export_tpu_tensor=True,
                            export_cpu_tensor=True)
    run_config = create_run_config(iterations_per_loop=4)

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    est = tpu_estimator.TPUEstimator(
        model_fn=model_fn, config=run_config, train_batch_size=16,
        export_to_tpu=False)
    est.train(_input_fn, steps=1)

    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export_no_tpu'))
    export_dir = est.export_saved_model(export_dir_base,
                                        self._serving_input_receiver_fn)
    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        with self.assertRaisesRegex(
            RuntimeError,
            'MetaGraphDef associated with tags \'serve\', \'tpu\' could not be '
            'found in SavedModel.'):
          loader.load(
              sess, [tag_constants.SERVING, tag_constants.TPU], export_dir)
        loader.load(
            sess, [tag_constants.SERVING], export_dir)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_tpu_savedmodel_export_to_tpu_false_eval(self):
    # Test exporting CPU evaulation graph when `export_to_tpu` is `False`.
    tmpdir = tempfile.mkdtemp()
    mode = model_fn_lib.ModeKeys.EVAL

    model_fn = get_model_fn(export_tpu_tensor=True, export_cpu_tensor=True)
    run_config = create_run_config(iterations_per_loop=4)

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    est = tpu_estimator.TPUEstimator(
        model_fn=model_fn,
        config=run_config,
        train_batch_size=16,
        export_to_tpu=False)
    est.train(_input_fn, steps=1)

    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export_no_tpu_eval'))
    export_dir = est.export_saved_model(
        export_dir_base, self._supervised_input_receiver_fn,
        experimental_mode=mode)

    # Check that all the files are in the right places.
    self.assertTrue(gfile.Exists(export_dir_base))

    # Restore, to validate that the export was well-formed.
    tag_set = export_lib.EXPORT_TAG_MAP[mode]
    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        loader.load(sess, tag_set, export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertIn('dense/kernel', graph_ops)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_export_estimator_savedmodel(self):
    export_tpu_tensor = True
    export_cpu_tensor = False

    tmpdir = tempfile.mkdtemp()

    def _input_fn(params):
      del params
      # Estimator does not pass `batch_size` to `input_fn`.
      return dummy_input_fn(batch_size=1)

    model_fn = get_model_fn(export_tpu_tensor=export_tpu_tensor,
                            export_cpu_tensor=export_cpu_tensor,
                            tpu_estimator_spec=False)
    est = estimator_lib.Estimator(model_fn=model_fn)
    est.train(_input_fn, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    export_dir = tpu_estimator.export_estimator_savedmodel(
        est,
        export_dir_base,
        self._serving_input_receiver_fn)

    self._validate_export(export_dir_base, export_dir, export_tpu_tensor,
                          export_cpu_tensor)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def test_regression_output_tensors_roundtrip(self):
    value = array_ops.placeholder(dtypes.float32, 1, name='value')
    regression_output = export_lib.RegressionOutput(value)
    self.assertSequenceEqual(
        [value],
        tpu_estimator._export_output_to_tensors(regression_output))

    value_new = array_ops.placeholder(dtypes.float32, 1, name='value_new')
    regression_output_new = (
        tpu_estimator._clone_export_output_with_tensors(
            regression_output, [value_new]
        )
    )
    self.assertEqual(value_new, regression_output_new.value)

  def test_predict_output_tensors_roundtrip(self):
    value1 = array_ops.placeholder(dtypes.float32, 1, name='value1')
    value2 = array_ops.placeholder(dtypes.float32, 1, name='value2')
    predict_output = export_lib.PredictOutput({
        'value1': value1,
        'value2': value2
    })
    export_output_tensors = tpu_estimator._export_output_to_tensors(
        predict_output)
    self.assertSameElements([value1, value2], export_output_tensors)
    self.assertLen(export_output_tensors, 2)

    tensors_new = [
        array_ops.identity(t, name=t.name.split(':')[0] + '_new')
        for t in export_output_tensors
    ]
    predict_output_new = tpu_estimator._clone_export_output_with_tensors(
        predict_output, tensors_new)
    outputs = predict_output_new.outputs
    self.assertLen(outputs, 2)
    self.assertEqual(outputs['value1'].name, 'value1_new:0')
    self.assertEqual(outputs['value2'].name, 'value2_new:0')

  def test_classification_output_tensors_roundtrip_classes_only(self):
    classes = array_ops.placeholder(dtypes.string, 1, name='classes')
    classification_output = export_lib.ClassificationOutput(
        classes=classes)

    classification_output_tensors = (tpu_estimator.
                                     _export_output_to_tensors(
                                         classification_output))
    self.assertEqual(classification_output_tensors, [None, classes])

    classes_new = array_ops.placeholder(dtypes.string, 1, name='classes_new')
    classification_output_new = (tpu_estimator.
                                 _clone_export_output_with_tensors(
                                     classification_output,
                                     [None, classes_new]))
    self.assertEqual(classification_output_new.classes, classes_new)

  def test_classification_output_tensors_roundtrip_scores_only(self):
    scores = array_ops.placeholder(dtypes.float32, 1, name='scores')
    classification_output = export_lib.ClassificationOutput(
        scores=scores)

    classification_output_tensors = (tpu_estimator.
                                     _export_output_to_tensors(
                                         classification_output))
    self.assertEqual(classification_output_tensors, [scores, None])

    scores_new = array_ops.placeholder(dtypes.float32, 1, name='scores_new')
    classification_output_new = (tpu_estimator.
                                 _clone_export_output_with_tensors(
                                     classification_output, [scores_new, None]))
    self.assertEqual(classification_output_new.scores, scores_new)

  def test_classification_output_tensors_roundtrip_classify_both(self):
    classes = array_ops.placeholder(dtypes.string, 1, name='classes')
    scores = array_ops.placeholder(dtypes.float32, 1, name='scores')
    classification_output = export_lib.ClassificationOutput(
        scores, classes)

    classification_output_tensors = (tpu_estimator.
                                     _export_output_to_tensors(
                                         classification_output))
    self.assertSequenceEqual(classification_output_tensors, [scores, classes])

    classes_new = array_ops.placeholder(dtypes.string, 1, name='classes_new')
    scores_new = array_ops.placeholder(dtypes.float32, 1, name='scores_new')
    classification_output_new = (tpu_estimator.
                                 _clone_export_output_with_tensors(
                                     classification_output,
                                     [scores_new, classes_new]))
    self.assertEqual(classification_output_new.classes, classes_new)
    self.assertEqual(classification_output_new.scores, scores_new)


def get_model_fn_v2():

  def model_fn(features, labels, mode, params):
    loss = None
    train_op = None
    export_outputs = None

    # This could be some pre-processing on CPU like calls to input layer with
    # embedding columns.
    x2 = features['x'] * 2

    def computation(input_tensor):
      return tf_keras_v1.__internal__.legacy.layers.dense(
          input_tensor, 1, kernel_initializer=init_ops.zeros_initializer())

    if mode != _PREDICT:
      predictions = computation(x2)
      loss = losses.mean_squared_error(labels, predictions)
      optimizer = tf.tpu.CrossShardOptimizer(
          training.GradientDescentOptimizer(learning_rate=0.5))
      train_op = optimizer.minimize(loss, training.get_global_step())
    else:
      inputs = [x2]
      if params['use_tpu']:
        predictions = array_ops.identity(
            tpu_estimator.inference_on_tpu(
                computation, inputs, num_batch_threads=1, max_batch_size=2,
                batch_timeout_micros=100),
            name='predictions')
      else:
        predictions = array_ops.identity(
            computation(*inputs), name='predictions')
      key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
      export_outputs = {
          key: export_lib.PredictOutput({'prediction': predictions})
      }

      classes = string_ops.as_string(predictions, name='classes')
      classification_output = export_lib.ClassificationOutput(classes=classes)
      export_outputs['classification'] = classification_output

    return tpu_estimator.TPUEstimatorSpec(
        mode,
        loss=loss,
        train_op=train_op,
        predictions={'predictions': predictions},
        export_outputs=export_outputs)

  return model_fn


class TPUEstimatorExportV2Test(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    feature_spec = {'x': parsing_ops.FixedLenFeature([1], dtypes.float32)}
    self._serving_input_receiver_fn = (
        export_lib.build_parsing_serving_input_receiver_fn(feature_spec))

  def test_export_tpu_savedmodel_e2e(self):
    tmpdir = tempfile.mkdtemp()

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    model_fn = get_model_fn_v2()
    run_config = create_run_config(iterations_per_loop=4)
    est = tpu_estimator.TPUEstimator(
        model_fn=model_fn,
        config=run_config,
        train_batch_size=16,
        export_to_tpu=True,
        export_saved_model_api_version=tpu_estimator.ExportSavedModelApiVersion
        .V2)
    est.train(_input_fn, steps=1)

    # Perform the export.
    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export'))
    export_dir = est.export_saved_model(export_dir_base,
                                        self._serving_input_receiver_fn)

    self._validate_export(export_dir_base, export_dir)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)

  def _validate_export(self, export_dir_base, export_dir):
    # Check that all the files are in the right places.
    self.assertTrue(gfile.Exists(export_dir_base))
    self.assertTrue(gfile.Exists(export_dir))
    self.assertTrue(
        gfile.Exists(
            os.path.join(
                compat.as_bytes(export_dir),
                compat.as_bytes('saved_model.pb'))))
    self.assertTrue(
        gfile.Exists(
            os.path.join(
                compat.as_bytes(export_dir), compat.as_bytes('variables'))))
    self.assertTrue(
        gfile.Exists(
            os.path.join(
                compat.as_bytes(export_dir),
                compat.as_bytes('variables/variables.index'))))
    self.assertTrue(
        gfile.Exists(
            os.path.join(
                compat.as_bytes(export_dir),
                compat.as_bytes('variables/variables.data-00000-of-00001'))))

    def session_run():
      example = example_pb2.Example()
      example.features.feature['x'].float_list.value.append(1)

      tensor_name_prediction = None
      tensor_name_classes = None
      key_prediction = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
      tensor_name_prediction = (
          meta_graph.signature_def[key_prediction].outputs['prediction'].name)
      key_classification = 'classification'
      tensor_name_classes = (
          meta_graph.signature_def[key_classification].outputs['classes'].name)

      sess.run(
          tensor_name_prediction,
          feed_dict={'input_example_tensor:0': [example.SerializeToString()]})
      sess.run(
          tensor_name_classes,
          feed_dict={'input_example_tensor:0': [example.SerializeToString()]})
      sess.run(
          [tensor_name_prediction, tensor_name_classes],
          feed_dict={'input_example_tensor:0': [example.SerializeToString()]})

    # Restore, to validate that the export was well-formed.
    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        meta_graph = loader.load(sess,
                                 [tag_constants.SERVING, tag_constants.TPU],
                                 export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertIn('input_example_tensor', graph_ops)
        self.assertIn('ParseExample/ParseExampleV2', graph_ops)
        self.assertNotIn('dense/kernel/GuaranteeConst', graph_ops)
        self.assertIn('batch/BatchFunction', graph_ops)

        sess.run(tf.tpu.initialize_system())
        session_run()

    # Restore, to validate that the export was well-formed.
    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        meta_graph = loader.load(sess, [tag_constants.SERVING], export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        self.assertIn('input_example_tensor', graph_ops)
        self.assertIn('ParseExample/ParseExampleV2', graph_ops)
        self.assertIn('dense/kernel', graph_ops)
        # GuaranteeConst ops won't be present in the CPU-only graph.
        self.assertNotIn('dense/kernel/GuaranteeConst', graph_ops)

        session_run()

  def test_export_tpu_savedmodel_export_to_tpu_false(self):
    # Test that when `export_to_tpu` is `False`, TPU metagraph is not exported.
    tmpdir = tempfile.mkdtemp()

    model_fn = get_model_fn_v2()
    run_config = create_run_config(iterations_per_loop=4)

    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])

    est = tpu_estimator.TPUEstimator(
        model_fn=model_fn,
        config=run_config,
        train_batch_size=16,
        export_to_tpu=False,
        export_saved_model_api_version=tpu_estimator.ExportSavedModelApiVersion
        .V2)
    est.train(_input_fn, steps=1)

    export_dir_base = os.path.join(
        compat.as_bytes(tmpdir), compat.as_bytes('export_no_tpu'))
    export_dir = est.export_saved_model(export_dir_base,
                                        self._serving_input_receiver_fn)
    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        with self.assertRaisesRegex(
            RuntimeError,
            'MetaGraphDef associated with tags \'serve\', \'tpu\' could not be '
            'found in SavedModel.'):
          loader.load(sess, [tag_constants.SERVING, tag_constants.TPU],
                      export_dir)
        loader.load(sess, [tag_constants.SERVING], export_dir)

    # Clean up.
    gfile.DeleteRecursively(tmpdir)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  test.main()
