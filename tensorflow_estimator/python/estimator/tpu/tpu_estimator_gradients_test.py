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
"""Tests to check gradients of TPUEstimator + TPU Embeddings."""

import math
import tempfile
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_estimator.python.estimator.util import tf_keras_v1
from tensorflow_estimator.python.estimator.tpu import tpu_config
from tensorflow_estimator.python.estimator.tpu import tpu_estimator

flags.DEFINE_integer('test_num_shards', 2, 'number of replicas to test')

FLAGS = flags.FLAGS

LEARNING_RATE = 0.12
HIDDEN_LAYER_SIZE = 20
KERNEL_INIT_VALUE = 0.1
BIAS_INIT_VALUE = 0.2
ADADGRAD_INIT_VALUE = 0.1
BUCKET_SIZE = 8
EMBEDDING_DIM = 3
KEY_NAME = 'x'
GRAD_MULTIPLIER = 1000.

BIAS_VAR = 'dense/bias:0'
CPU_EMBEDDING_VAR = 'dense_features/x_embedding/embedding_weights:0'
CPU_EMBEDDING_ACCUM_VAR = 'dense_features/x_embedding/embedding_weights/Adagrad:0'
TPU_EMBEDDING_VAR = 'dense_features/x_embedding/embedding_weights/part_0:0'
TPU_EMBEDDING_ACCUM_VAR = 'dense_features/x_embedding/embedding_weights/Adagrad/part_0:0'

# This test must be running with "--xla_jf_conv_full_precision=true",
DEFAULT_TOL = 1e-6


def create_model_fn(feature_columns, optimizer_type='adagrad'):

  def model_fn(features, labels, mode, params):
    del params

    dense_features = tf_keras_v1.layers.DenseFeatures(feature_columns)
    input_layer = dense_features(features)
    hidden_layer = tf_keras_v1.__internal__.legacy.layers.dense(
        input_layer,
        HIDDEN_LAYER_SIZE,
        kernel_initializer=tf.constant_initializer(KERNEL_INIT_VALUE),
        bias_initializer=tf.constant_initializer(BIAS_INIT_VALUE))

    last_layer = tf.reduce_sum(hidden_layer, axis=1)

    logits = tf.reshape(last_layer, [-1])
    labels = tf.reshape(labels, [-1])
    losses = tf.square(labels - logits)

    # Use reduce_mean to match the CrossShardOptimizer reduction.
    loss = tf.reduce_mean(losses)
    if optimizer_type == 'adagrad':
      optimizer = tf.train.AdagradOptimizer(
          LEARNING_RATE, initial_accumulator_value=ADADGRAD_INIT_VALUE)
    elif optimizer_type == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    else:
      raise ValueError('{} is not supported.'.format(optimizer_type))
    # Default reduction=tf.losses.Reduction.MEAN
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tpu_estimator.TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  return model_fn


def get_estimator(use_tpu,
                  output_dir,
                  feature_columns,
                  batch_size,
                  optimizer_type='adagrad',
                  grad_multiplier_fn=None):
  run_config = tpu_config.RunConfig(
      master='',
      model_dir=output_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False),
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=1,
          num_shards=FLAGS.test_num_shards,
          per_host_input_for_training=(
              tpu_config.InputPipelineConfig.PER_HOST_V2)),
      save_checkpoints_steps=1)

  if optimizer_type == 'adagrad':
    optimization_parameters = tpu_estimator.AdagradParameters(
        LEARNING_RATE,
        ADADGRAD_INIT_VALUE,
        use_gradient_accumulation=False)
  elif optimizer_type == 'sgd':
    optimization_parameters = tpu_estimator.StochasticGradientDescentParameters(
        LEARNING_RATE)

  estimator = tpu_estimator.TPUEstimator(
      model_fn=create_model_fn(feature_columns, optimizer_type),
      use_tpu=use_tpu,
      config=run_config,
      train_batch_size=batch_size,
      eval_batch_size=batch_size,
      embedding_config_spec=tpu_estimator.EmbeddingConfigSpec(
          feature_columns=feature_columns,
          optimization_parameters=optimization_parameters,
          experimental_gradient_multiplier_fn=grad_multiplier_fn))
  return estimator


def get_feature_columns():
  initializer = tf.zeros_initializer()

  column = tf.feature_column.categorical_column_with_identity(
      key=KEY_NAME, num_buckets=BUCKET_SIZE)
  embedding_fc = tf.tpu.experimental.embedding_column(
      column,
      dimension=EMBEDDING_DIM,
      combiner='mean',
      initializer=initializer)

  all_fc = [embedding_fc]
  return all_fc


class _EmbeddingVariableHook(tf.train.SessionRunHook):
  """A hook to record the embedding variable."""

  def __init__(self, use_tpu, include_slot_vars=True):
    self._use_tpu = use_tpu
    self._include_slot_vars = include_slot_vars

  def _set_bias_var(self):
    self._bias_var = [v for v in tf.trainable_variables() if v.name == BIAS_VAR]

  def begin(self):
    search_var = TPU_EMBEDDING_VAR if self._use_tpu else CPU_EMBEDDING_VAR
    self._var = [v for v in tf.global_variables() if v.name == search_var][0]
    if self._include_slot_vars:
      search_accum_var = TPU_EMBEDDING_ACCUM_VAR if self._use_tpu else CPU_EMBEDDING_ACCUM_VAR
      self._slot_var = [
          v for v in tf.global_variables() if v.name == search_accum_var
      ][0]
    self._set_bias_var()

    self.bias_values = []
    self.var_values = []
    self.slot_var_values = []

  def after_create_session(self, session, coord):
    del coord
    self.bias_values.append(session.run(self._bias_var))
    self.var_values.append(session.run(self._var))
    if self._include_slot_vars:
      self.slot_var_values.append(session.run(self._slot_var))

  def after_run(self, run_context, run_values):
    self.bias_values.append(run_context.session.run(self._bias_var))
    self.var_values.append(run_context.session.run(self._var))
    if self._include_slot_vars:
      self.slot_var_values.append(run_context.session.run(self._slot_var))


def get_activation_gradients(label):
  """Gets the sample gradient w.r.t activation according to the model_fn."""
  # The sample loss is (label - logits)**2, where
  #     logits = \sum_j^HIDDEN_LAYER_SIZE (
  #         \sum_i^EMBEDDING_DIM w_i * kernel + bias)
  #
  # Note kernel and bias are both constant initializer in this test.
  #
  # So, gradients of loss w.r.t w_i is
  #    grads = 2 * (label - logits) gradients( logits w.r.t. w_i)
  #          = 2 * (label - logits) (-1 * HIDDEN_LAYER_SIZE * kernel)
  #
  # Given the weights are zero initializer,
  #    grads = - 2 HIDDEN_LAYER_SIZE * kernel (label - HIDDEN_LAYER_SIZE * bias)

  return -2 * HIDDEN_LAYER_SIZE * KERNEL_INIT_VALUE * (
      label - HIDDEN_LAYER_SIZE * BIAS_INIT_VALUE)


def get_embedding_update(gradient, previous_accum_inc=0.0):
  """Gets the embedding update according to Adagrad.

  Args:
    gradient: the embedding gradient.
    previous_accum_inc: The previous total accumulator increment (in addition to
        the initialize value)

  Returns:
    the value to apply gradient.
  """
  return -LEARNING_RATE * (
      gradient /
      math.sqrt(ADADGRAD_INIT_VALUE + previous_accum_inc + gradient**2))


def dense_to_sparse(dense_tensor, out_type, ignore_value=-1):
  indices = tf.where(
      tf.not_equal(dense_tensor,
                   tf.constant(ignore_value, dense_tensor.dtype)))
  values = tf.gather_nd(dense_tensor, indices)
  shape = tf.shape(dense_tensor, out_type=out_type)
  return tf.SparseTensor(indices, values, shape)


class TPUEstimatorGradientsSimpleTest(tf.test.TestCase):
  """Test gradients for different Ids in global batch.

  In all examples examined by this test, in one global batch, each embedding ID
  appears only once. So, we can expect the embedding variable and accumulate
  variable will be same after one CPU training and TPU training.

  For more complicated example, each ID can appear multiple times in one core
  mini-batch and across multiple cores, see
  TPUEstimatorGradientsWithIdCollisionTest.
  """

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def _input_fn(self, params):
    # This input_fn returns a tuple of sparse tensor and a dense tensor in
    # sequence.
    # sample 0:  sparse tensor value [0]  dense (target) [1]
    # sample 1:  sparse tensor value [1]  dense (target) [2]
    # sample 2:  sparse tensor value [2]  dense (target) [3]
    # ...
    batch_size = params['batch_size']

    ds = tf.data.Dataset.range(8)

    def _map_fn(index):
      index = tf.reshape(index, [1])
      dense_tensor = tf.cast(index + 1, tf.float32)
      return ({KEY_NAME: dense_to_sparse(index, tf.int64)}, dense_tensor)

    ds = ds.map(_map_fn)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

  def test_input_fn(self):
    ds = self._input_fn({'batch_size': 1})
    gn = ds.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
      for i in range(8):
        features, dense_tensor = sess.run(gn)
        sparse_tensor = features[KEY_NAME]

        self.assertAllEqual([[0, 0]], sparse_tensor.indices)
        self.assertAllEqual([i], sparse_tensor.values)
        self.assertAllEqual([[i + 1]], dense_tensor)

    tf.reset_default_graph()

    ds = self._input_fn({'batch_size': 2})
    gn = ds.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
      for i in range(4):
        features, dense_tensor = sess.run(gn)
        sparse_tensor = features[KEY_NAME]
        self.assertAllEqual([[0, 0], [1, 0]], sparse_tensor.indices)
        self.assertAllEqual([i * 2, i * 2 + 1], sparse_tensor.values)
        self.assertAllEqual([[i * 2 + 1], [i * 2 + 2]], dense_tensor)

  def assert_embedding_variables(self,
                                 gradients_for_embedding,
                                 hand_calculated_embedding_values,
                                 values_in_hook,
                                 tol=DEFAULT_TOL):
    """Assert the embedding variables after training one step."""

    expected_embedding_var_values = []
    # Before training, all zeros (zeros_initializer)
    expected_embedding_var_values.append(np.zeros((BUCKET_SIZE, EMBEDDING_DIM)))

    after_training_var_values = np.zeros((BUCKET_SIZE, EMBEDDING_DIM))
    embedding_row_value_after_one_step = [
        get_embedding_update(g) for g in gradients_for_embedding
    ]

    for i in range(len(embedding_row_value_after_one_step)):
      after_training_var_values[i][:] = embedding_row_value_after_one_step[i]
    expected_embedding_var_values.append(after_training_var_values)

    # Check against hand calculated value.
    self.assertAllClose(hand_calculated_embedding_values,
                        embedding_row_value_after_one_step)
    # Check against the value recorded during training.
    self.assertAllClose(
        expected_embedding_var_values, values_in_hook, atol=tol, rtol=tol)

  def assert_embedding_slot_variables(self, gradients_for_embedding,
                                      hand_calculated_embedding_slot_values,
                                      values_in_hook, tol):
    """Assert the embedding slot variables after training one step."""

    expected_embedding_slot_var_values = []
    # Before training, all same (ADADGRAD_INIT_VALUE)
    expected_embedding_slot_var_values.append(
        np.ones((BUCKET_SIZE, EMBEDDING_DIM)) * ADADGRAD_INIT_VALUE)

    after_training_slot_var_values = np.zeros((BUCKET_SIZE, EMBEDDING_DIM))
    accumulator_sum = [
        ADADGRAD_INIT_VALUE + g * g for g in gradients_for_embedding
    ]
    for i in range(len(accumulator_sum)):
      after_training_slot_var_values[i][:] = accumulator_sum[i]

    expected_embedding_slot_var_values.append(after_training_slot_var_values)

    # Check against hand calculated value.
    self.assertAllClose(hand_calculated_embedding_slot_values, accumulator_sum)
    # Check against the value recorded during training.
    self.assertAllClose(
        expected_embedding_slot_var_values, values_in_hook, atol=tol, rtol=tol)

  def test_one_sample_per_core(self):
    use_tpu = True
    per_core_batch_size = 1
    num_shards = FLAGS.test_num_shards
    batch_size = num_shards * per_core_batch_size

    hook = _EmbeddingVariableHook(use_tpu=use_tpu)

    estimator = get_estimator(use_tpu, self._model_dir, get_feature_columns(),
                              batch_size)
    estimator.train(self._input_fn, steps=1, hooks=[hook])

    # After training one step, the core 0 gets one sample with ID 0, and core 1
    # gets one sample with ID 1. So, all other IDs' embedding vars remain as
    # zeros.
    gradients_for_embedding = [
        get_activation_gradients(label=1),
        get_activation_gradients(label=2)
    ]
    gradients_for_embedding += [0] * (BUCKET_SIZE - batch_size)
    # Scale the gradients by 1/num_shards as CrossShardOptimizer scales the
    # loss for MEAN reduction.
    gradients_for_embedding = np.array(gradients_for_embedding) / num_shards

    hand_calculated_embedding_values = [0] * BUCKET_SIZE
    # Gradients are 6.0 and 4.0. the embedding value should
    #    - LEARNING_RATE* x / math.sqrt(ADADGRAD_INIT_VALUE + x*x)
    hand_calculated_embedding_values[:2] = [
        -0.1198336797537491, -0.11962674870701442
    ]

    self.assert_embedding_variables(
        gradients_for_embedding=gradients_for_embedding,
        hand_calculated_embedding_values=hand_calculated_embedding_values,
        values_in_hook=hook.var_values,
        tol=DEFAULT_TOL)

    hand_calculated_embedding_slot_values = [ADADGRAD_INIT_VALUE] * BUCKET_SIZE
    hand_calculated_embedding_slot_values[0] += 6.0**2
    hand_calculated_embedding_slot_values[1] += 4.0**2

    self.assert_embedding_slot_variables(
        gradients_for_embedding=gradients_for_embedding,
        hand_calculated_embedding_slot_values=(
            hand_calculated_embedding_slot_values),
        values_in_hook=hook.slot_var_values,
        tol=DEFAULT_TOL)

  def test_one_sample_per_core_tpu_vs_cpu(self):
    use_tpu = True
    per_core_batch_size = 1
    num_shards = FLAGS.test_num_shards
    batch_size = num_shards * per_core_batch_size

    # TPU
    tpu_hook = _EmbeddingVariableHook(use_tpu=use_tpu)
    estimator = get_estimator(use_tpu, self._model_dir + '_tpu',
                              get_feature_columns(), batch_size)
    estimator.train(self._input_fn, steps=1, hooks=[tpu_hook])

    # CPU
    use_tpu = False
    cpu_hook = _EmbeddingVariableHook(use_tpu=use_tpu)

    cpu_estimator = get_estimator(use_tpu, self._model_dir + '_cpu',
                                  get_feature_columns(), batch_size)
    cpu_estimator.train(self._input_fn, steps=1, hooks=[cpu_hook])

    tol = DEFAULT_TOL
    self.assertAllClose(
        tpu_hook.var_values, cpu_hook.var_values, atol=tol, rtol=tol)
    self.assertAllClose(
        tpu_hook.slot_var_values, cpu_hook.slot_var_values, atol=tol, rtol=tol)

    # Also check dense.
    self.assertAllClose(
        tpu_hook.bias_values, cpu_hook.bias_values, atol=tol, rtol=tol)

  def test_multi_samples_per_core(self):
    use_tpu = True
    per_core_batch_size = 2
    num_shards = FLAGS.test_num_shards
    batch_size = num_shards * per_core_batch_size

    hook = _EmbeddingVariableHook(use_tpu=use_tpu)

    estimator = get_estimator(use_tpu, self._model_dir, get_feature_columns(),
                              batch_size)
    estimator.train(self._input_fn, steps=1, hooks=[hook])

    # After training one step, the core 0 gets two samples with ID 0 and 1. For
    # core 1 gets two samples with ID 2 and 3. So, all other IDs' embedding vars
    # remain as zeros.
    gradients_for_embedding = [
        get_activation_gradients(label=1),
        get_activation_gradients(label=2),
        get_activation_gradients(label=3),
        get_activation_gradients(label=4)
    ]
    gradients_for_embedding += [0] * (BUCKET_SIZE - batch_size)
    gradients_for_embedding = np.array(gradients_for_embedding)
    # Scale the gradients by 1/ per_core_batch_size, as for each core the loss
    # is mean loss.
    gradients_for_embedding /= per_core_batch_size
    # Further scale the gradients by 1/num_shards as CrossShardOptimizer scales
    # the loss for MEAN reduction.
    gradients_for_embedding /= num_shards

    # Gradients are 3.0, 2.0, 1.0, and 0.0. the embedding value should
    #    - LEARNING_RATE* x / math.sqrt(ADADGRAD_INIT_VALUE + x*x)
    hand_calculated_embedding_values = [0] * BUCKET_SIZE
    hand_calculated_embedding_values[:2] = [-0.119338837, -0.118527551]
    hand_calculated_embedding_values[2:4] = [-0.1144155107094, 0]

    self.assert_embedding_variables(
        gradients_for_embedding=gradients_for_embedding,
        hand_calculated_embedding_values=hand_calculated_embedding_values,
        values_in_hook=hook.var_values,
        tol=DEFAULT_TOL)

    hand_calculated_embedding_slot_values = [ADADGRAD_INIT_VALUE] * BUCKET_SIZE
    hand_calculated_embedding_slot_values[0] += 3.0**2
    hand_calculated_embedding_slot_values[1] += 2.0**2
    hand_calculated_embedding_slot_values[2] += 1.0**2

    self.assert_embedding_slot_variables(
        gradients_for_embedding=gradients_for_embedding,
        hand_calculated_embedding_slot_values=(
            hand_calculated_embedding_slot_values),
        values_in_hook=hook.slot_var_values,
        tol=DEFAULT_TOL)

  def test_multi_samples_per_core_tpu_vs_cpu(self):
    use_tpu = True
    per_core_batch_size = 2
    num_shards = FLAGS.test_num_shards
    batch_size = num_shards * per_core_batch_size

    # TPU
    tpu_hook = _EmbeddingVariableHook(use_tpu=use_tpu)
    estimator = get_estimator(use_tpu, self._model_dir + '_tpu',
                              get_feature_columns(), batch_size)
    estimator.train(self._input_fn, steps=1, hooks=[tpu_hook])

    # CPU
    use_tpu = False
    cpu_hook = _EmbeddingVariableHook(use_tpu=use_tpu)
    cpu_estimator = get_estimator(use_tpu, self._model_dir + '_cpu',
                                  get_feature_columns(), batch_size)
    cpu_estimator.train(self._input_fn, steps=1, hooks=[cpu_hook])

    tol = DEFAULT_TOL
    self.assertAllClose(
        tpu_hook.var_values, cpu_hook.var_values, atol=tol, rtol=tol)
    self.assertAllClose(
        tpu_hook.slot_var_values,
        cpu_hook.slot_var_values,
        atol=tol,
        rtol=tol)

    # Also check dense.
    self.assertAllClose(
        tpu_hook.bias_values, cpu_hook.bias_values, atol=tol, rtol=tol)


class TPUEstimatorGradientsWithIdCollisionTest(tf.test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def _input_fn(self, params):
    # This input_fn is expected to be called twice each having a batch_size 2.
    # The first output will be
    #   label = [1, 2]
    #   sparse inputs: SarseTensorValue(
    #       indices=array([[0, 0], [0, 1],
    #                      [1, 0], [1, 1]]),
    #       values=array([0, 1,
    #                     1, 2]),
    #       dense_shape=array([2, 2]))
    #
    # The second output will be
    #   label = [3, 4]
    #   sparse inputs: SarseTensorValue(
    #       indices=array([[0, 0], [0, 1],
    #                      [1, 0], [1, 1]]),
    #       values=array([1, 2,
    #                     2, 3]),
    #       dense_shape=array([2, 2]))
    #
    # So, each sample has two ids. Each core gets two samples, which share some
    # ids. And different cores share ids also.
    batch_size = params['batch_size']
    self.assertTrue(batch_size == 2 or batch_size == 4)

    ds = tf.data.Dataset.range(8)

    def _map_fn(index):
      x = tf.floordiv(index, 2)
      y = tf.floormod(index, 2)

      label = tf.cast(index + 1, tf.float32)
      label = tf.reshape(label, [1])

      target_dense = tf.stack([x + y, x + y + 1])
      return ({KEY_NAME: dense_to_sparse(target_dense, tf.int64)}, label)

    ds = ds.map(_map_fn)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

  def test_input_fn(self):
    ds = self._input_fn({'batch_size': 2})
    gn = ds.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
      # First call
      features, label = sess.run(gn)
      sparse_tensor = features[KEY_NAME]
      self.assertAllEqual([[0, 0], [0, 1], [1, 0], [1, 1]],
                          sparse_tensor.indices)
      self.assertAllEqual([
          0,
          1,
          1,
          2,
      ], sparse_tensor.values)
      self.assertAllEqual([[1], [2]], label)

      # second call
      features, label = sess.run(gn)
      sparse_tensor = features[KEY_NAME]
      self.assertAllEqual([[0, 0], [0, 1], [1, 0], [1, 1]],
                          sparse_tensor.indices)
      self.assertAllEqual([
          1,
          2,
          2,
          3,
      ], sparse_tensor.values)
      self.assertAllEqual([[3], [4]], label)

  def test_adagrad_opt_embedding_variables_on_tpu(self):
    use_tpu = True
    per_core_batch_size = 2
    num_shards = FLAGS.test_num_shards
    batch_size = num_shards * per_core_batch_size

    hook = _EmbeddingVariableHook(use_tpu=use_tpu)

    estimator = get_estimator(use_tpu, self._model_dir, get_feature_columns(),
                              batch_size)
    estimator.train(self._input_fn, steps=1, hooks=[hook])

    final_step = 1
    tol = DEFAULT_TOL

    # In this parcticular example, the gradient w.r.t. each activation is not
    # gradient w.r.t. embedding due to the combiner.
    unscaled_gradient_for_activation = [
        get_activation_gradients(label=1),
        get_activation_gradients(label=2),
        get_activation_gradients(label=3),
        get_activation_gradients(label=4),
    ]
    self.assertAllEqual([12., 8., 4.0, 0.0], unscaled_gradient_for_activation)

    # Due to reduce_mean and 1/num_shards scaling, the embeddings gradients
    # are 3.0, 2.0, 1.0, 0.0 as num of samples per core is 2 and
    # num_shards (number of cores) is 2.

    # Now calcuates the gradients for embedding vars and accumulator for each
    # var.
    #
    # Note the IDs for each core are

    # Core 0 sample 0  IDs: [0  1]
    # Core 0 sample 1  IDs: [1  2]

    # Core 1 sample 0  IDs: [1  2]
    # Core 1 sample 1  IDs: [2  3]

    # For embedding ID 0, it appears only in the first sample of the first core.
    # So, its gradient is 3.0 / 2, where 1/2 is due to the mean combiner.
    gradient_for_id_0 = 1.5
    accumuator_for_id_0 = gradient_for_id_0**2 + ADADGRAD_INIT_VALUE

    self.assertAllClose(
        accumuator_for_id_0,
        hook.slot_var_values[final_step][0][0],
        atol=tol, rtol=tol)

    # embedding_update = - LR * g / (init_accm + g**2)
    gradient_update_for_id_0 = get_embedding_update(gradient_for_id_0)
    self.assertAllClose(
        gradient_update_for_id_0,
        hook.var_values[final_step][0][0],
        rtol=tol, atol=tol)

    # Similarly, for embedding ID 3, it appears only in the second sample of the
    # second core.  So, its gradient is 0.0 / 2, where 1/2 is due to the mean
    # combiner.
    gradient_for_id_3 = 0
    accumuator_for_id_3 = gradient_for_id_3**2 + ADADGRAD_INIT_VALUE

    self.assertAllClose(
        accumuator_for_id_3,
        hook.slot_var_values[final_step][3][0],
        atol=tol, rtol=tol)

    # embedding_update = - LR * g / (init_accm + g**2)
    gradient_update_for_id_3 = get_embedding_update(gradient_for_id_3)
    self.assertAllClose(
        gradient_update_for_id_3,
        hook.var_values[final_step][3][0],
        rtol=tol, atol=tol)

    # For embedding ID 2, it appears in
    #  - second sample of first core
    #  - first sample of second core
    #  - second sample of second core
    #
    # Note that the gradients of the second activation of the second core is 0.
    # So, equivalent, it is same as
    #
    #  - second sample of first core -> gradient = 0.5 * 2.0 = 1.0
    #  - first sample of second core -> gradient = 0.5 * 1.0 = 0.5
    gradient_for_id_2_in_core_0 = 1.0
    gradient_for_id_2_in_core_1 = 0.5
    accumuator_for_id_2 = (
        ADADGRAD_INIT_VALUE + gradient_for_id_2_in_core_0**2 +
        gradient_for_id_2_in_core_1**2)

    self.assertAllClose(
        accumuator_for_id_2,
        hook.slot_var_values[final_step][2][0],
        atol=tol, rtol=tol)

    # embedding_update = (
    #    - LR * g1 / (init_accum + g1**2)
    #    - LR * g2 / (init_accum + g1**2 + g2**2)
    gradient_update_for_id_2_after_apply_core_0 = get_embedding_update(
        gradient_for_id_2_in_core_0)
    accum_inc = gradient_for_id_2_in_core_0**2

    gradient_update_for_id_2_after_apply_core_1 = get_embedding_update(
        gradient_for_id_2_in_core_1, previous_accum_inc=accum_inc)

    embedding_update_for_id_2 = (
        gradient_update_for_id_2_after_apply_core_0 +
        gradient_update_for_id_2_after_apply_core_1)

    self.assertAllClose(
        embedding_update_for_id_2,
        hook.var_values[final_step][2][0],
        rtol=tol, atol=tol)

    # For embedding ID 1, it appears in
    #  - first sample of first core
    #  - second sample of first core
    #  - first sample of second core
    #
    # So, the gradient for each sample
    #
    #  - first sample of first core -> gradient = 0.5 * 3.0 = 1.5
    #  - second sample of first core -> gradient = 0.5 * 2.0 = 1.0
    #  - first sample of second core -> gradient = 0.5 * 1.0 = 0.5
    #
    # Baracore combines the gradients in single core and then applies them core
    # by core.
    gradient_for_id_1_in_core_0 = 1.5 + 1.0
    gradient_for_id_1_in_core_1 = 0.5
    accumuator_for_id_1 = (
        ADADGRAD_INIT_VALUE + gradient_for_id_1_in_core_0**2 +
        gradient_for_id_1_in_core_1**2)

    self.assertAllClose(
        accumuator_for_id_1,
        hook.slot_var_values[final_step][1][0],
        atol=tol, rtol=tol)

    # ID 1 resides on Core 1, so the updates from Core 1 are applied first.
    # embedding_update = (
    #    - LR * g1 / (init_accum + g1**2)
    #    - LR * g2 / (init_accum + g1**2 + g2**2)
    gradient_update_for_id_1_after_apply_core_1 = get_embedding_update(
        gradient_for_id_1_in_core_1)
    accum_inc = gradient_for_id_1_in_core_1**2

    gradient_update_for_id_1_after_apply_core_0 = get_embedding_update(
        gradient_for_id_1_in_core_0, previous_accum_inc=accum_inc)

    embedding_update_for_id_1 = (
        gradient_update_for_id_1_after_apply_core_0 +
        gradient_update_for_id_1_after_apply_core_1)

    self.assertAllClose(
        embedding_update_for_id_1,
        hook.var_values[final_step][1][0],
        rtol=tol, atol=tol)

  def test_adagrad_opt_embedding_variables_on_cpu(self):
    use_tpu = False
    per_core_batch_size = 2
    num_shards = FLAGS.test_num_shards
    batch_size = num_shards * per_core_batch_size

    hook = _EmbeddingVariableHook(use_tpu=use_tpu)

    estimator = get_estimator(use_tpu, self._model_dir, get_feature_columns(),
                              batch_size)
    estimator.train(self._input_fn, steps=1, hooks=[hook])

    final_step = 1
    tol = DEFAULT_TOL

    # In this CPU example, the gradients for embedding rows are same as the
    # above: not only the sample loss and gradient, but also the scaling. The
    # only difference is CPU combines all gradients in one update; while TPU
    # updates the gradients core by core.
    #
    # For ID 0: gradient = 0.5 * 3.0 = 1.5
    # For ID 1: gradient = 0.5 * 3.0 + 0.5 * 2.0 + 0.5 * 1.0 = 3.0
    # For ID 2: gradient = 0.5 * 2.0 + 0.5 * 1.0 = 1.5
    # For ID 3: gradient = 0.5 * 0.0 = 0.0

    gradients_for_embedding = np.array([1.5, 3.0, 1.5, 0])

    # Check accumulator after one step.
    for index, gradient in enumerate(gradients_for_embedding):
      accumulator = ADADGRAD_INIT_VALUE + gradient**2
      self.assertAllClose(
          accumulator,
          hook.slot_var_values[final_step][index][0],
          atol=tol, rtol=tol)

    # Check embedding value after one step.
    for index, gradient in enumerate(gradients_for_embedding):
      embedding_update = get_embedding_update(gradient)
      self.assertAllClose(
          embedding_update,
          hook.var_values[final_step][index][0],
          atol=tol, rtol=tol)

  def test_sgd_opt_embedding_variables_on_cpu(self):
    use_tpu = False
    per_core_batch_size = 2
    num_shards = FLAGS.test_num_shards
    batch_size = num_shards * per_core_batch_size

    hook = _EmbeddingVariableHook(use_tpu=use_tpu, include_slot_vars=False)

    estimator = get_estimator(
        use_tpu,
        self._model_dir,
        get_feature_columns(),
        batch_size,
        optimizer_type='sgd')
    estimator.train(self._input_fn, steps=1, hooks=[hook])

    final_step = 1
    tol = DEFAULT_TOL

    # In this CPU example, the gradients for embedding rows are same as the
    # above: not only the sample loss and gradient, but also the scaling. The
    # only difference is CPU combines all gradients in one update; while TPU
    # updates the gradients core by core.
    #
    # For ID 0: gradient = 0.5 * 3.0 = 1.5
    # For ID 1: gradient = 0.5 * 3.0 + 0.5 * 2.0 + 0.5 * 1.0 = 3.0
    # For ID 2: gradient = 0.5 * 2.0 + 0.5 * 1.0 = 1.5
    # For ID 3: gradient = 0.5 * 0.0 = 0.0

    gradients_for_embedding = np.array([1.5, 3.0, 1.5, 0])
    # SGD has simple update rule, w += - lr * g
    embedding_update = [LEARNING_RATE * (-g) for g in gradients_for_embedding]

    # Check embedding value after one step.
    for index in range(len(gradients_for_embedding)):
      self.assertAllClose(
          embedding_update[index],
          hook.var_values[final_step][index][0],
          atol=tol, rtol=tol)

  def test_sgd_opt_embedding_variables_cpu_vs_tpu(self):
    # For sgd, cpu and tpu should agree.
    per_core_batch_size = 2
    num_shards = FLAGS.test_num_shards
    batch_size = num_shards * per_core_batch_size

    use_tpu = False
    cpu_hook = _EmbeddingVariableHook(use_tpu=use_tpu, include_slot_vars=False)
    cpu_estimator = get_estimator(
        use_tpu,
        self._model_dir + '_cpu',
        get_feature_columns(),
        batch_size,
        optimizer_type='sgd')
    cpu_estimator.train(self._input_fn, steps=1, hooks=[cpu_hook])

    use_tpu = True
    tpu_hook = _EmbeddingVariableHook(use_tpu=use_tpu, include_slot_vars=False)
    estimator = get_estimator(
        use_tpu,
        self._model_dir + '_tpu',
        get_feature_columns(),
        batch_size,
        optimizer_type='sgd')
    estimator.train(self._input_fn, steps=1, hooks=[tpu_hook])

    tol = DEFAULT_TOL
    self.assertAllClose(
        cpu_hook.var_values, tpu_hook.var_values, atol=tol, rtol=tol)

    self.assertAllClose(
        cpu_hook.bias_values,
        tpu_hook.bias_values,
        atol=tol, rtol=tol)

    # Test gradient multiplier.
    def grad_multiplier_fn(global_step):
      # First global step is 0.
      return tf.cast(global_step + 1, tf.float32) * GRAD_MULTIPLIER

    tpu_hook2 = _EmbeddingVariableHook(use_tpu=use_tpu, include_slot_vars=False)
    estimator2 = get_estimator(
        use_tpu,
        self._model_dir + '_tpu_grad_multiplier',
        get_feature_columns(),
        batch_size,
        optimizer_type='sgd',
        grad_multiplier_fn=grad_multiplier_fn)
    estimator2.train(self._input_fn, steps=1, hooks=[tpu_hook2])

    tol = DEFAULT_TOL
    self.assertAllClose([v * GRAD_MULTIPLIER for v in cpu_hook.var_values],
                        tpu_hook2.var_values,
                        atol=tol * GRAD_MULTIPLIER,
                        rtol=tol * GRAD_MULTIPLIER)

    self.assertAllClose(
        cpu_hook.bias_values, tpu_hook2.bias_values, atol=tol, rtol=tol)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
