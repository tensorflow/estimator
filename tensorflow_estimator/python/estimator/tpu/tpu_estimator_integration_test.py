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
"""Tests for TPUEstimator."""

import contextlib
import tempfile
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2

from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.export import export
from tensorflow_estimator.python.estimator.export import export_output
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
      features['x'], 1, kernel_initializer=tf.zeros_initializer())


def model_fn_global_step_incrementer(features, labels, mode, params):
  del params
  loss = None
  train_op = None
  predictions = dense_computation(features)
  if mode != _PREDICT:
    loss = tf.losses.mean_squared_error(labels, predictions)
    optimizer = tf.tpu.CrossShardOptimizer(
        tf.train.GradientDescentOptimizer(learning_rate=0.5))
    train_op = optimizer.minimize(loss, tf.train.get_global_step())
  return tpu_estimator.TPUEstimatorSpec(
      mode,
      loss=loss,
      train_op=train_op,
      predictions={'predictions': predictions},
      export_outputs={
          'test': export_output.PredictOutput({
              'prediction': predictions
          })
      })


def dummy_input_fn_with_dataset(batch_size, repeat=True, x=None):
  if x is None:
    x = np.random.normal(size=[batch_size, 1]).astype(np.float32)
  labels = [[2.0]] * batch_size

  dataset1 = tf.data.Dataset.from_tensor_slices(x)
  dataset2 = tf.data.Dataset.from_tensor_slices(labels)
  dataset = tf.data.Dataset.zip((dataset1, dataset2))
  if repeat:
    dataset = dataset.repeat()
  dataset = dataset.batch(batch_size, drop_remainder=True)

  def _map(x, y):
    return {'x': x}, y

  return dataset.map(_map)


def dummy_input_fn(batch_size, repeat=True):
  dataset = dummy_input_fn_with_dataset(batch_size, repeat)
  iterator = dataset.make_one_shot_iterator()
  return iterator.get_next()


def create_run_config(iterations_per_loop, **kwargs):
  return tpu_config.RunConfig(
      master='',
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=iterations_per_loop,
          num_shards=FLAGS.test_num_shards,
          **kwargs),
  )


class TPUEstimatorIntegrationTest(tf.test.TestCase):

  def setUp(self):
    self._recorded_input_fn_invoke_metadata = {
        _TRAIN: {'called_count': 0, 'batch_size': None},
        _EVAL: {'called_count': 0, 'batch_size': None},
        _PREDICT: {'called_count': 0, 'batch_size': None}
    }
    self._data = np.linspace(0., 1., 100, dtype=np.float32).reshape(-1, 1)
    self._export_mode = False

  @contextlib.contextmanager
  def export_mode(self):
    """Enable the export mode for model_fn."""
    # Inside the model_fn, the test will check the batch size passed via params.
    # However, export mode should not have that. It is infeasible for model_fn
    # to distinguish the predict vs export mode today. So, this contextmanager
    # helps the model_fn to do that.
    self._export_mode = True
    yield
    self._export_mode = False

  def assertInputFnCalledCountAndBatch(self, expected_called_count,
                                       expected_batch_size):
    real_called_count = {k: v['called_count'] for k, v in
                         self._recorded_input_fn_invoke_metadata.items()}
    real_batch_size = {k: v['batch_size'] for k, v in
                       self._recorded_input_fn_invoke_metadata.items()}
    self.assertEqual(expected_called_count, real_called_count)
    self.assertEqual(expected_batch_size, real_batch_size)

  def _generate_expected_batch_size_and_called_count(
      self,
      num_shards,
      train_batch_size,
      eval_batch_size,
      predict_batch_size,
      train_sharding_policy=_UNSHARDED,
      eval_sharding_policy=_UNSHARDED,
      predict_sharding_policy=None):

    expected_batch_size_for_model_fn = {}
    expected_batch_size_for_input_fn = {}
    expected_called_count_for_input_fn = {}

    if train_sharding_policy == _PER_SHARD:
      self.assertEqual(0, train_batch_size % num_shards)
      expected_batch_size_for_model_fn[_TRAIN] = train_batch_size // num_shards
      expected_batch_size_for_input_fn[_TRAIN] = train_batch_size // num_shards
      expected_called_count_for_input_fn[_TRAIN] = num_shards
    elif train_sharding_policy == _PER_HOST:
      self.assertEqual(0, train_batch_size % num_shards)
      expected_batch_size_for_model_fn[_TRAIN] = train_batch_size // num_shards
      expected_batch_size_for_input_fn[_TRAIN] = train_batch_size
      expected_called_count_for_input_fn[_TRAIN] = 1
    else:
      expected_batch_size_for_model_fn[_TRAIN] = train_batch_size
      expected_batch_size_for_input_fn[_TRAIN] = train_batch_size
      expected_called_count_for_input_fn[_TRAIN] = 1

    if eval_sharding_policy == _PER_HOST:
      self.assertEqual(0, train_batch_size % num_shards)
      expected_batch_size_for_model_fn[_EVAL] = eval_batch_size // num_shards
      expected_batch_size_for_input_fn[_EVAL] = eval_batch_size
      expected_called_count_for_input_fn[_EVAL] = 1
    else:
      expected_batch_size_for_model_fn[_EVAL] = eval_batch_size
      expected_batch_size_for_input_fn[_EVAL] = eval_batch_size
      expected_called_count_for_input_fn[_EVAL] = 1

    if predict_sharding_policy is None:
      # On CPU.
      expected_batch_size_for_model_fn[_PREDICT] = predict_batch_size
      expected_batch_size_for_input_fn[_PREDICT] = predict_batch_size
      expected_called_count_for_input_fn[_PREDICT] = 1
    else:
      expected_batch_size_for_model_fn[_PREDICT] = (
          predict_batch_size // num_shards)
      expected_batch_size_for_input_fn[_PREDICT] = predict_batch_size
      expected_called_count_for_input_fn[_PREDICT] = 1

    return (expected_batch_size_for_model_fn, expected_batch_size_for_input_fn,
            expected_called_count_for_input_fn)

  def _wrap_input_fn_with_batch_size(self, batch_size, input_fn):
    def _input_fn(params):
      self.assertNotIn('batch_size', params)
      params['batch_size'] = batch_size
      return input_fn(params)
    return _input_fn

  def _make_input_fn(self, mode, repeat=False, take=None):
    metadata = self._recorded_input_fn_invoke_metadata[mode]
    def _input_fn(params):
      metadata['called_count'] += 1
      batch_size = params['batch_size']

      if metadata['batch_size'] is None:
        metadata['batch_size'] = batch_size
      else:
        self.assertEqual(batch_size, metadata['batch_size'])

      dataset1 = tf.data.Dataset.from_tensor_slices(self._data)
      dataset2 = tf.data.Dataset.from_tensor_slices(self._data)
      dataset = tf.data.Dataset.zip((dataset1, dataset2))

      if repeat:
        dataset = dataset.repeat()

      dataset = dataset.batch(batch_size)

      if take:
        dataset = dataset.take(take)

      def _map_fn(x, y):
        x.set_shape([batch_size, 1])
        y.set_shape([batch_size, 1])
        return {'x': x}, y

      dataset = dataset.map(_map_fn)
      return dataset

    return _input_fn

  def _make_model_fn(self, batch_size_dict, use_tpu_estimator_spec=False):

    def _create_estimator_spec(mode, loss=None, predictions=None,
                               export_outputs=None, eval_metrics=None,
                               train_op=None):
      if use_tpu_estimator_spec:
        return tpu_estimator.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            predictions=predictions,
            export_outputs=export_outputs,
            eval_metrics=eval_metrics)
      else:
        return model_fn_lib.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            predictions=predictions,
            export_outputs=export_outputs,
            eval_metric_ops=(eval_metrics[0](*eval_metrics[1]) if eval_metrics
                             else None))

    def _model_fn(features, labels, mode, params):
      if not self._export_mode:
        # Always check batch size in params
        self.assertEqual(batch_size_dict[mode], params['batch_size'])
      else:
        self.assertNotIn('batch_size', params)

      # Check the input feeds correct shape for train and eval. When eval on CPU
      # or predict, it is allowed to have dynamic shape. So, here only validates
      # the fully known shape (which covers the TPU train).
      if features['x'].shape.is_fully_defined():
        self.assertEqual(batch_size_dict[mode], features['x'].shape[0])

      predictions = tf_keras_v1.__internal__.legacy.layers.dense(
          features['x'], 1,
          kernel_initializer=tf.ones_initializer())
      export_outputs = {
          'predictions': export_output.RegressionOutput(predictions)
      }

      if mode == _PREDICT:
        return _create_estimator_spec(
            mode=mode,
            predictions={'predictions': predictions},
            export_outputs=export_outputs)

      loss = tf.losses.mean_squared_error(labels, predictions)

      optimizer = tf.tpu.CrossShardOptimizer(
          tf.train.GradientDescentOptimizer(learning_rate=0.5))
      train_op = optimizer.minimize(loss,
                                    global_step=tf.train.get_global_step())

      eval_metrics = (
          lambda labels, predictions: {  # pylint: disable=g-long-lambda
              'absolute_error': tf.metrics.mean_absolute_error(
                  labels, predictions)},
          [labels, predictions])
      return _create_estimator_spec(
          mode=mode,
          loss=loss,
          predictions={'predictions': predictions},
          export_outputs=export_outputs,
          train_op=train_op,
          eval_metrics=eval_metrics)
    return _model_fn

  def _test_identity_savedmodel(self, export_dir):
    with tf.Graph().as_default() as graph:
      with tf.Session(graph=graph) as sess:
        metagraph_def = tf.saved_model.loader.load(sess, [tf.saved_model.SERVING], export_dir)
        fetch = metagraph_def.signature_def['predictions'].outputs['outputs']
        feed = metagraph_def.signature_def['predictions'].inputs['inputs']
        for x in self._data:
          example = example_pb2.Example(
              features=feature_pb2.Features(
                  feature={
                      'x':
                          feature_pb2.Feature(
                              float_list=feature_pb2.FloatList(
                                  value=np.ravel(x)))
                  })).SerializeToString()
          y = sess.run(fetch.name, feed_dict={feed.name: [example]})
          self.assertAlmostEqual(y, x[0], delta=0.01)

  def test_complete_flow_with_per_core_input(self):
    # Choose the train_batch_size divisible by 2 and 8 (common shards in test
    # env) and batch_size for eval and predict prime number.
    train_batch_size = 16
    eval_batch_size = 16
    predict_batch_size = 8

    run_config = create_run_config(iterations_per_loop=4,
                                   per_host_input_for_training=False)
    num_shards = run_config.tpu_config.num_shards

    (expected_batch_size_for_model_fn, expected_batch_size_for_input_fn,
     expected_called_count_for_input_fn) = (
         self._generate_expected_batch_size_and_called_count(
             num_shards,
             train_batch_size,
             eval_batch_size,
             predict_batch_size,
             train_sharding_policy=_PER_SHARD,
             eval_sharding_policy=_PER_HOST,
             predict_sharding_policy=_PER_HOST))

    est = tpu_estimator.TPUEstimator(
        model_fn=self._make_model_fn(
            expected_batch_size_for_model_fn, use_tpu_estimator_spec=True),
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size)

    # TRAIN
    # learn y = x
    # Note: Gradients are all zero. Just testing execution.
    def _input_fn(params):
      dataset = self._make_input_fn(mode=_TRAIN, repeat=True)(params)
      return tf.data.make_one_shot_iterator(dataset).get_next()

    train_input_fn = _input_fn
    est.train(train_input_fn, steps=7)

    # EVALUTE
    scores = est.evaluate(self._make_input_fn(mode=_EVAL), steps=6)
    self.assertEqual(7, scores['global_step'])
    self.assertGreater(0.1, scores['absolute_error'])

    # PREDICT
    predict_input_fn = self._make_input_fn(mode=_PREDICT, take=2)
    predictions = [x['predictions'] for x in est.predict(predict_input_fn)]
    self.assertAllClose(
        self._data[:predict_batch_size * 2], predictions, atol=0.01)

    # Verify all input_fn invoke recorded metadata.
    self.assertInputFnCalledCountAndBatch(
        expected_called_count_for_input_fn, expected_batch_size_for_input_fn)

    # EXPORT
    feature_spec = {'x': tf.io.FixedLenFeature([1], tf.float32)}
    serving_input_receiver_fn = (
        export.build_parsing_serving_input_receiver_fn(feature_spec))
    with self.export_mode():
      export_dir = est.export_saved_model(
          tempfile.mkdtemp(dir=self.get_temp_dir()), serving_input_receiver_fn)
    self.assertTrue(tf.gfile.Exists(export_dir))
    self._test_identity_savedmodel(export_dir)

  def test_complete_flow_with_per_host_input(self):
    # Choose the train_batch_size divisible by 2 and 8 (common shards in test
    # env) and batch_size for eval and predict prime number.
    train_batch_size = 16
    eval_batch_size = 16
    predict_batch_size = 16

    run_config = create_run_config(
        iterations_per_loop=4, per_host_input_for_training=True)
    num_shards = run_config.tpu_config.num_shards

    (expected_batch_size_for_model_fn, expected_batch_size_for_input_fn,
     expected_called_count_for_input_fn) = (
         self._generate_expected_batch_size_and_called_count(
             num_shards,
             train_batch_size,
             eval_batch_size,
             predict_batch_size,
             train_sharding_policy=_PER_HOST,
             eval_sharding_policy=_PER_HOST,
             predict_sharding_policy=_PER_HOST))

    est = tpu_estimator.TPUEstimator(
        model_fn=self._make_model_fn(
            expected_batch_size_for_model_fn, use_tpu_estimator_spec=True),
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size)

    # TRAIN
    # learn y = x
    # Note: Gradients are all zero. Just testing execution.
    train_input_fn = self._make_input_fn(mode=_TRAIN, repeat=True)
    est.train(train_input_fn, steps=7)

    # EVALUTE
    scores = est.evaluate(self._make_input_fn(mode=_EVAL), steps=6)
    self.assertEqual(7, scores['global_step'])
    self.assertGreater(0.1, scores['absolute_error'])

    # PREDICT
    predict_input_fn = self._make_input_fn(mode=_PREDICT, take=2)
    predictions = [x['predictions'] for x in est.predict(predict_input_fn)]
    self.assertAllClose(
        self._data[:predict_batch_size * 2], predictions, atol=0.01)

    # Verify all input_fn invoke recorded metadata.
    self.assertInputFnCalledCountAndBatch(
        expected_called_count_for_input_fn, expected_batch_size_for_input_fn)

    # EXPORT
    feature_spec = {'x': tf.io.FixedLenFeature([1], tf.float32)}
    serving_input_receiver_fn = (
        export.build_parsing_serving_input_receiver_fn(feature_spec))
    with self.export_mode():
      export_dir = est.export_saved_model(
          tempfile.mkdtemp(dir=self.get_temp_dir()), serving_input_receiver_fn)
    self.assertTrue(tf.gfile.Exists(export_dir))
    self._test_identity_savedmodel(export_dir)

  def test_complete_flow_with_eval_on_tpu(self):
    # Choose the train_batch_size divisible by 2 and 8 (common shards in test
    # env) and batch_size for eval and predict prime number.
    train_batch_size = 16
    eval_batch_size = 8
    predict_batch_size = 8

    run_config = create_run_config(iterations_per_loop=4)
    num_shards = run_config.tpu_config.num_shards

    (expected_batch_size_for_model_fn, expected_batch_size_for_input_fn,
     expected_called_count_for_input_fn) = (
         self._generate_expected_batch_size_and_called_count(
             num_shards,
             train_batch_size,
             eval_batch_size,
             predict_batch_size,
             train_sharding_policy=_PER_HOST,
             eval_sharding_policy=_PER_HOST,
             predict_sharding_policy=_PER_HOST))

    est = tpu_estimator.TPUEstimator(
        model_fn=self._make_model_fn(
            expected_batch_size_for_model_fn, use_tpu_estimator_spec=True),
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size)

    # TRAIN
    # learn y = x
    # Note: Gradients are all zero. Just testing execution.
    train_input_fn = self._make_input_fn(mode=_TRAIN, repeat=True)
    est.train(train_input_fn, steps=7)

    # EVALUTE
    eval_input_fn = self._make_input_fn(mode=_EVAL, repeat=False)
    scores = est.evaluate(eval_input_fn, steps=2)
    self.assertEqual(7, scores['global_step'])
    self.assertGreater(0.1, scores['absolute_error'])

    # PREDICT
    predict_input_fn = self._make_input_fn(mode=_PREDICT, take=2)
    predictions = [x['predictions'] for x in est.predict(predict_input_fn)]
    self.assertAllClose(
        self._data[:predict_batch_size * 2], predictions, atol=0.01)

    # Verify all input_fn invoke recorded metadata.
    self.assertInputFnCalledCountAndBatch(
        expected_called_count_for_input_fn, expected_batch_size_for_input_fn)

    # EXPORT
    feature_spec = {'x': tf.io.FixedLenFeature([1], tf.float32)}
    serving_input_receiver_fn = (
        export.build_parsing_serving_input_receiver_fn(feature_spec))
    with self.export_mode():
      export_dir = est.export_saved_model(
          tempfile.mkdtemp(dir=self.get_temp_dir()), serving_input_receiver_fn)
    self.assertTrue(tf.gfile.Exists(export_dir))
    self._test_identity_savedmodel(export_dir)

  def test_complete_flow_with_no_tpu(self):
    # Choose the train_batch_size divisible by 2 and 8 (common shards in test
    # env) and batch_size for eval and predict prime number.
    train_batch_size = 16
    eval_batch_size = 8
    predict_batch_size = 1

    run_config = create_run_config(iterations_per_loop=4)
    num_shards = run_config.tpu_config.num_shards

    (expected_batch_size_for_model_fn, expected_batch_size_for_input_fn,
     expected_called_count_for_input_fn) = (
         self._generate_expected_batch_size_and_called_count(
             num_shards, train_batch_size, eval_batch_size, predict_batch_size,
             train_sharding_policy=_UNSHARDED,
             eval_sharding_policy=_UNSHARDED))

    est = tpu_estimator.TPUEstimator(
        model_fn=self._make_model_fn(
            expected_batch_size_for_model_fn, use_tpu_estimator_spec=True),
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size,
        use_tpu=False)

    # TRAIN
    # learn y = x
    # Note: Gradients are all zero. Just testing execution.
    train_input_fn = self._make_input_fn(mode=_TRAIN, repeat=True)
    est.train(train_input_fn, steps=7)

    # EVALUTE
    eval_input_fn = self._make_input_fn(mode=_EVAL)
    scores = est.evaluate(eval_input_fn, steps=2)
    self.assertEqual(7, scores['global_step'])
    self.assertGreater(0.1, scores['absolute_error'])

    # PREDICT
    predict_input_fn = self._make_input_fn(mode=_PREDICT)
    predictions = [x['predictions'] for x in est.predict(predict_input_fn)]
    self.assertAllClose(self._data, predictions, atol=0.01)

    # Verify all input_fn invoke recorded metadata.
    self.assertInputFnCalledCountAndBatch(
        expected_called_count_for_input_fn, expected_batch_size_for_input_fn)

    # EXPORT
    feature_spec = {'x': tf.io.FixedLenFeature([1], tf.float32)}
    serving_input_receiver_fn = (
        export.build_parsing_serving_input_receiver_fn(feature_spec))
    with self.export_mode():
      export_dir = est.export_saved_model(
          tempfile.mkdtemp(dir=self.get_temp_dir()), serving_input_receiver_fn)
    self.assertTrue(tf.gfile.Exists(export_dir))
    self._test_identity_savedmodel(export_dir)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
