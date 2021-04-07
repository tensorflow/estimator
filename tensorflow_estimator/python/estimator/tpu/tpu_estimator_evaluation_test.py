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
"""Tests for TPUEstimator evaluation related functionalities."""

from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow.python.training import evaluation

from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.export import export_output as export_output_lib
from tensorflow_estimator.python.estimator.tpu import tpu_config
from tensorflow_estimator.python.estimator.tpu import tpu_estimator
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
  return tf.compat.v1.layers.dense(
      features['x'], 1, kernel_initializer=tf.compat.v1.zeros_initializer())


def get_model_fn(export_tpu_tensor=True, export_cpu_tensor=False,
                 tpu_estimator_spec=True):

  def model_fn(features, labels, mode, params):
    del params
    loss = None
    train_op = None
    predictions = dense_computation(features)
    export_outputs = None
    if mode != _PREDICT:
      loss = tf.compat.v1.losses.mean_squared_error(labels, predictions)
      optimizer = tf.compat.v1.tpu.CrossShardOptimizer(
          tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.5))
      train_op = optimizer.minimize(loss, tf.compat.v1.train.get_global_step())
    else:
      if export_tpu_tensor:
        key = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        export_outputs = {
            key: export_output_lib.PredictOutput({
                'prediction': predictions
            })
        }
      else:
        export_outputs = {}

      if export_cpu_tensor:

        def host_call(predictions):
          classes = tf.as_string(predictions, name='classes')
          classification_output = export_output_lib.ClassificationOutput(
              classes=classes)
          export_outputs['classification'] = classification_output

        tf.compat.v1.tpu.outside_compilation(host_call, predictions)

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

  dataset1 = tf.compat.v1.data.Dataset.from_tensor_slices(x)
  dataset2 = tf.compat.v1.data.Dataset.from_tensor_slices(labels)
  dataset = tf.compat.v1.data.Dataset.zip((dataset1, dataset2))
  if repeat:
    dataset = dataset.repeat()
  dataset = dataset.batch(batch_size, drop_remainder=True)

  def _map(x, y):
    return {'x': x}, y

  return dataset.map(_map)


def dummy_input_fn(batch_size, repeat=True):
  dataset = dummy_input_fn_with_dataset(batch_size, repeat)
  iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
  return iterator.get_next()


def create_run_config(iterations_per_loop, **kwargs):
  return tpu_config.RunConfig(
      master='',
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=iterations_per_loop,
          num_shards=FLAGS.test_num_shards,
          **kwargs),
  )


class TPUEstimatorEvaluationTest(tf.test.TestCase):

  def _create_input_fn(self):
    def _input_fn(params):
      return dummy_input_fn(params['batch_size'])
    return _input_fn

  def _create_head(self, mode, loss, eval_metrics):
    """Creates a head returning `TPUEstimatorSpec` based on mode."""
    if mode == _EVAL:
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode, eval_metrics=eval_metrics, loss=loss)
    # Train
    optimizer = tf.compat.v1.tpu.CrossShardOptimizer(
        tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.5))
    train_op = optimizer.minimize(
        loss, global_step=tf.compat.v1.train.get_global_step())
    return tpu_estimator.TPUEstimatorSpec(
        mode=mode, train_op=train_op, loss=loss)

  def _create_head_with_eval_metric_ops(self, mode, loss, eval_metric_ops):
    """Creates a head returning `TPUEstimatorSpec` based on mode.

    This version contains eval that will not run on TPUs, where eval_metric_ops
    has not been split into a metrics_fn that runs on CPUs. The intent is to
    test the entire eval (model_fn forward pass) and metrics output on CPU.

    Args:
      mode: The mode such as TRAIN, EVAL.
      loss: Training loss `Tensor`. Must be either scalar, or with shape `[1]`.
      eval_metric_ops: Dict of metric results keyed by name.

    Returns:
      An EstimatorSpec for EVAL or TPUEstimatorSpec otherwise.
    """
    if mode == _EVAL:
      return model_fn_lib.EstimatorSpec(
          mode=mode, eval_metric_ops=eval_metric_ops, loss=loss)
    # Train
    optimizer = tf.compat.v1.tpu.CrossShardOptimizer(
        tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.5))
    train_op = optimizer.minimize(
        loss, global_step=tf.compat.v1.train.get_global_step())
    return tpu_estimator.TPUEstimatorSpec(
        mode=mode, train_op=train_op, loss=loss)

  def _metric_fn_on_cpu(self, labels, predictions):
    return {
        'mse': tf.compat.v1.metrics.mean_absolute_error(labels, predictions),
    }

  def _model_fn_without_eval_metrics(self, features, labels, mode, params):
    del params  # unused.
    predictions = tf.compat.v1.layers.dense(
        features['x'], 1, kernel_initializer=tf.compat.v1.zeros_initializer())
    loss = tf.compat.v1.losses.mean_squared_error(labels, predictions)

    return self._create_head(mode, loss, None)

  def _model_fn_with_eval_tensor_list(self, features, labels, mode, params):
    del params  # unused.
    predictions = tf.compat.v1.layers.dense(
        features['x'], 1, kernel_initializer=tf.compat.v1.zeros_initializer())
    loss = tf.compat.v1.losses.mean_squared_error(labels, predictions)

    return self._create_head(
        mode, loss,
        eval_metrics=(self._metric_fn_on_cpu, [labels, predictions]))

  def _model_fn_with_eval_dict(self, features, labels, mode, params):
    del params  # unused.
    predictions = tf.compat.v1.layers.dense(
        features['x'], 1, kernel_initializer=tf.compat.v1.zeros_initializer())
    loss = tf.compat.v1.losses.mean_squared_error(labels, predictions)

    return self._create_head(
        mode, loss,
        eval_metrics=(self._metric_fn_on_cpu, {
            'labels': labels,
            'predictions': predictions}))

  def _model_fn_with_eval_metric_ops(self, features, labels, mode, params):
    del params  # unused.
    predictions = tf.compat.v1.layers.dense(
        features['x'], 1, kernel_initializer=tf.compat.v1.zeros_initializer())
    loss = tf.compat.v1.losses.mean_squared_error(labels, predictions)

    eval_metric_ops = self._metric_fn_on_cpu(labels, predictions)
    return self._create_head_with_eval_metric_ops(
        mode, loss, eval_metric_ops)

  def _test_eval_steps(self, model_fn, expected_eval_steps, iterations):

    run_config = create_run_config(iterations_per_loop=iterations)
    est = tpu_estimator.TPUEstimator(
        model_fn=model_fn,
        config=run_config,
        train_batch_size=16,
        eval_batch_size=16)

    est.train(self._create_input_fn(), steps=1)

    class _EvalStepCheckHook(tf.compat.v1.train.SessionRunHook):
      """Check eval step counter after one session.run.

      As the evaluation sets the eval iterations as the eval steps, the
      after_run should be invoked only once.
      """

      def __init__(self, iterations_per_loop, test_case):
        """Constructs the run hook."""
        self._iterations = iterations_per_loop
        self._invoked = False
        self._test_case = test_case

      def before_run(self, run_context):
        del run_context
        # For eval on TPU, the hook should be run only once.
        self._test_case.assertFalse(self._invoked)

      def after_run(self, run_context, run_values):
        # To avoid race condition between the eval step read and increment in
        # evaluation graph, we read the value explicitly here.
        eval_steps = run_context.session.run(
            evaluation._get_or_create_eval_step())
        self._test_case.assertEqual(expected_eval_steps, eval_steps)
        self._test_case.assertFalse(self._invoked)
        self._invoked = True

    est.evaluate(self._create_input_fn(),
                 steps=expected_eval_steps,
                 hooks=[_EvalStepCheckHook(iterations, self)])

  def test_no_eval_metrics(self):
    run_config = create_run_config(iterations_per_loop=2)
    est = tpu_estimator.TPUEstimator(
        model_fn=self._model_fn_without_eval_metrics,
        config=run_config,
        train_batch_size=16,
        eval_batch_size=16)

    est.train(self._create_input_fn(), steps=1)
    est.evaluate(self._create_input_fn(), steps=1)

  def test_eval_steps_not_effected_by_training_iterations(self):
    self._test_eval_steps(
        model_fn=self._model_fn_with_eval_tensor_list,
        expected_eval_steps=2,
        iterations=4)
    self._test_eval_steps(
        model_fn=self._model_fn_with_eval_tensor_list,
        expected_eval_steps=6,
        iterations=4)

  def test_eval_steps_with_no_eval_metrics(self):
    self._test_eval_steps(
        model_fn=self._model_fn_without_eval_metrics,
        expected_eval_steps=6,
        iterations=1)

  def test_eval_metrics_with_tensor_list(self):
    run_config = create_run_config(iterations_per_loop=2)
    est = tpu_estimator.TPUEstimator(
        model_fn=self._model_fn_with_eval_tensor_list,
        config=run_config,
        train_batch_size=16,
        eval_batch_size=16)

    est.train(self._create_input_fn(), steps=1)
    est.evaluate(self._create_input_fn(), steps=1)

  def test_eval_batch_size_with_non_divisible_num_shards_broadcast_mode(self):
    run_config = create_run_config(
        iterations_per_loop=2,
        per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST)
    est = tpu_estimator.TPUEstimator(
        model_fn=self._model_fn_with_eval_tensor_list,
        config=run_config,
        train_batch_size=7,
        eval_batch_size=7)

    est.train(self._create_input_fn(), steps=1)
    est.evaluate(self._create_input_fn(), steps=1)

  def test_eval_metrics_with_tensor_list_on_cpu(self):
    run_config = create_run_config(iterations_per_loop=2)
    est = tpu_estimator.TPUEstimator(
        model_fn=self._model_fn_with_eval_tensor_list,
        config=run_config,
        train_batch_size=16,
        eval_batch_size=16,
        use_tpu=False)

    est.train(self._create_input_fn(), steps=1)
    est.evaluate(self._create_input_fn(), steps=1)

  def test_eval_metrics_with_dict(self):
    run_config = create_run_config(iterations_per_loop=2)
    est = tpu_estimator.TPUEstimator(
        model_fn=self._model_fn_with_eval_dict,
        config=run_config,
        train_batch_size=16,
        eval_batch_size=16)

    est.train(self._create_input_fn(), steps=1)
    est.evaluate(self._create_input_fn(), steps=1)

  def test_eval_metrics_with_dict_on_cpu(self):
    run_config = create_run_config(iterations_per_loop=2)
    est = tpu_estimator.TPUEstimator(
        model_fn=self._model_fn_with_eval_dict,
        config=run_config,
        train_batch_size=16,
        eval_batch_size=16,
        use_tpu=False)

    est.train(self._create_input_fn(), steps=1)
    est.evaluate(self._create_input_fn(), steps=1)

  def test_eval_metrics_ops_cpu_training(self):
    run_config = create_run_config(iterations_per_loop=2)
    est = tpu_estimator.TPUEstimator(
        model_fn=self._model_fn_with_eval_metric_ops,
        config=run_config,
        train_batch_size=16,
        eval_batch_size=16,
        use_tpu=False,
        eval_on_tpu=False)

    est.train(self._create_input_fn(), steps=1)
    est.evaluate(self._create_input_fn(), steps=1)

  def test_eval_metrics_ops_cpu_training_warning(self):
    run_config = create_run_config(iterations_per_loop=2)
    est = tpu_estimator.TPUEstimator(
        model_fn=self._model_fn_with_eval_metric_ops,
        config=run_config,
        train_batch_size=16,
        eval_batch_size=16,
        use_tpu=False,
        # eval_on_tpu is ignored if use_tpu is False
        eval_on_tpu=True)

    est.train(self._create_input_fn(), steps=1)
    est.evaluate(self._create_input_fn(), steps=1)

  def test_eval_metrics_ops_tpu_training(self):
    run_config = create_run_config(iterations_per_loop=2)
    est = tpu_estimator.TPUEstimator(
        model_fn=self._model_fn_with_eval_metric_ops,
        config=run_config,
        train_batch_size=16,
        eval_batch_size=16,
        use_tpu=True,
        eval_on_tpu=False)

    est.train(self._create_input_fn(), steps=1)
    est.evaluate(self._create_input_fn(), steps=1)

  def test_eval_metrics_ops_tpu_training_failure(self):
    run_config = create_run_config(iterations_per_loop=2)
    est = tpu_estimator.TPUEstimator(
        model_fn=self._model_fn_with_eval_metric_ops,
        config=run_config,
        train_batch_size=16,
        eval_batch_size=16,
        use_tpu=True,
        # Generates an error on eval, because model_fn(mode=EVAL)
        # has not been split into an eval_metrics_fn.
        eval_on_tpu=True)

    est.train(self._create_input_fn(), steps=1)
    with self.assertRaisesRegex(
        RuntimeError, 'TPU evaluation must have type`TPUEstimatorSpec`'):
      est.evaluate(self._create_input_fn(), steps=1)

  def test_error_out_if_steps_is_float(self):
    with self.assertRaisesRegex(TypeError, 'must be int'):
      run_config = create_run_config(iterations_per_loop=2)
      est = tpu_estimator.TPUEstimator(
          model_fn=self._model_fn_with_eval_dict,
          config=run_config,
          train_batch_size=16,
          eval_batch_size=16,
          use_tpu=True)
      est.evaluate(self._create_input_fn(), steps=12.3)

  def test_error_out_if_steps_is_invalid(self):
    with self.assertRaisesRegex(ValueError, 'must be positive'):
      run_config = create_run_config(iterations_per_loop=2)
      est = tpu_estimator.TPUEstimator(
          model_fn=self._model_fn_with_eval_dict,
          config=run_config,
          train_batch_size=16,
          eval_batch_size=16,
          use_tpu=True)
      est.evaluate(self._create_input_fn(), steps=-321)


if __name__ == '__main__':
  tf.test.main()
