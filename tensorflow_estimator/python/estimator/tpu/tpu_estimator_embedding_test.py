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

from absl import flags
from absl.testing import parameterized
import itertools
import os
import tempfile

import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python import data as dataset_lib
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.layers import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.tpu import feature_column_v2 as tpu_fc_v2
from tensorflow.python.tpu import tpu_embedding
from tensorflow.python.tpu import tpu_optimizer
from tensorflow.python.training import training
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.export import export
from tensorflow_estimator.python.estimator.export import export_output as export_output_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import tpu_config
from tensorflow_estimator.python.estimator.tpu import tpu_estimator

flags.DEFINE_integer('test_num_shards', 8, 'number of replicas to test')

FLAGS = flags.FLAGS

_TRAIN = model_fn_lib.ModeKeys.TRAIN
_EVAL = model_fn_lib.ModeKeys.EVAL
_PREDICT = model_fn_lib.ModeKeys.PREDICT
_PER_HOST_V1 = tpu_config.InputPipelineConfig.PER_HOST_V1
_PER_HOST_V2 = tpu_config.InputPipelineConfig.PER_HOST_V2

# Constant used for tests that uses categorical_column with vocabulary
_VOCAB_EMBEDDING_DIM = 10
_VOCAB_SIZE = 4
_VOCAB_NUM_BUCKETS = 5


def dense_computation(features):
  return layers.dense(
      features['x'], 1, kernel_initializer=init_ops.zeros_initializer())


def create_run_config(iterations_per_loop, **kwargs):
  return tpu_config.RunConfig(
      master='',
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=iterations_per_loop,
          num_shards=FLAGS.test_num_shards,
          **kwargs),
  )


class TPUEstimatorFeatureColumnTestBase(test.TestCase):

  def setUp(self):
    self._old_value = tpu_estimator._WRAP_INPUT_FN_INTO_WHILE_LOOP

    feature_spec = {
        'x': parsing_ops.SparseFeature(['ix0', 'ix1'], 'val',
                                       dtypes.int64, [1, 100]),
        'y': parsing_ops.SparseFeature(['ix0', 'ix1'], 'val',
                                       dtypes.int64, [1, 100])}
    self._serving_input_receiver_fn = (
        export.build_parsing_serving_input_receiver_fn(feature_spec))
    super().setUp()

  def tearDown(self):
    tpu_estimator._WRAP_INPUT_FN_INTO_WHILE_LOOP = self._old_value
    super().tearDown()

  def _create_estimator_with_feature_columns(self,
                                             feature_columns,
                                             numeric_check=False,
                                             use_cpu=False,
                                             input_method=_PER_HOST_V2):
    """Returns TPUEstimator which uses `feature_columns` in model_fn."""

    def _model_fn(features, labels, mode, params):
      """Creates simple TF model using feature_columns to create input layer."""
      del params
      sequence_columns, non_sequence_columns = (
          tpu_fc_v2.split_sequence_columns_v2(feature_columns))
      if sequence_columns:
        sequence_layer = fc_lib.SequenceFeatures(sequence_columns)
        sequence_features, sequence_lengths = sequence_layer(features)
        sequence_lengths = math_ops.cast(sequence_lengths, dtypes.float32)
      if non_sequence_columns:
        dense_layer = fc_lib.DenseFeatures(non_sequence_columns)
        input_layer = dense_layer(features)
      if numeric_check:
        # Make predictions the same as input_layer. This is used in some tests
        # where we set the labels to be the same as input_layer, which forces
        # the loss to be zero.
        if sequence_columns:
          # For sequence columns, we return the sequence lengths, so that we can
          # verify that these have been correctly calculated.
          predictions = array_ops.concat(sequence_lengths, -1)
        else:
          predictions = array_ops.identity(input_layer)
      else:
        if sequence_columns:
          # At this point we know that all the sequence features have the same
          # max sequence length. To get the total number of entries, so we can
          # reshape, we need total embedding dimension * max_sequence_length
          sequence_entries_per_batch = (
              sequence_features.shape[-1] *
              sequence_columns[0].get_max_sequence_length())
          flattened = array_ops.reshape(
              sequence_features, [-1, sequence_entries_per_batch])
          sequence_lengths = array_ops.expand_dims(sequence_lengths, -1)
          input_layer = array_ops.concat(
              [input_layer, flattened, sequence_lengths], -1)
        predictions = layers.dense(
            input_layer, 1, kernel_initializer=init_ops.zeros_initializer())

      loss = None
      train_op = None
      eval_metrics = None
      export_outputs = None
      if mode == model_fn_lib.ModeKeys.TRAIN:
        loss = losses.mean_squared_error(labels, predictions)
        optimizer = training.AdagradOptimizer(learning_rate=0.5)
        optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)
        train_op = optimizer.minimize(
            loss, global_step=training.get_global_step())
      elif mode == model_fn_lib.ModeKeys.EVAL:
        loss = losses.mean_squared_error(labels, predictions)

        def metric_fn_on_cpu(labels, predictions):
          return {
              'mse': metrics_lib.mean_absolute_error(labels, predictions),
          }

        eval_metrics = (metric_fn_on_cpu, [labels, predictions])

      else:
        export_outputs = {'prediction':
                          export_output_lib.PredictOutput(
                              {'prediction': predictions})}
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          train_op=train_op,
          loss=loss,
          eval_metrics=eval_metrics,
          export_outputs=export_outputs,
          predictions=predictions)

    run_config = create_run_config(
        iterations_per_loop=2,
        per_host_input_for_training=input_method)
    embedding_config_spec = tpu_estimator.EmbeddingConfigSpec(
        feature_columns=feature_columns,
        optimization_parameters=tpu_estimator.AdagradParameters(
            learning_rate=.01, initial_accumulator=0.1),
    )
    return tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        config=run_config,
        train_batch_size=8,
        eval_batch_size=8,
        use_tpu=not use_cpu,
        embedding_config_spec=embedding_config_spec,
        export_to_tpu=True)


class TPUEstimatorFeatureColumnTest(TPUEstimatorFeatureColumnTestBase,
                                    parameterized.TestCase):

  def test_get_tpu_embedding_config_from_feature_columns(self):
    feature_a = 'a'
    feature_b = 'b'  # shared
    feature_c = 'c'  # shared, weighted
    feature_d = 'd'  # weighted
    feature_e = 'e'  # sequence
    feature_f = 'f'  # shared sequence
    feature_g = 'g'  # shared sequence
    feature_h = 'h'  # shared non-sequence

    categorical_column_a = fc_lib.categorical_column_with_identity(
        key=feature_a, num_buckets=3)
    categorical_column_b = fc_lib.categorical_column_with_identity(
        key=feature_b, num_buckets=6)
    categorical_column_c = fc_lib.categorical_column_with_identity(
        key=feature_c, num_buckets=6)
    weight_feature_key_c = 'c_weight'
    weighted_column_c = fc_lib.weighted_categorical_column(
        categorical_column=categorical_column_c,
        weight_feature_key=weight_feature_key_c)
    categorical_column_d = fc_lib.categorical_column_with_identity(
        key=feature_d, num_buckets=3)
    weight_feature_key_d = 'd_weight'
    weighted_column_d = fc_lib.weighted_categorical_column(
        categorical_column=categorical_column_d,
        weight_feature_key=weight_feature_key_d)
    sequence_categorical_column_e = (
        fc_lib.sequence_categorical_column_with_identity(
            key=feature_e, num_buckets=7))
    sequence_categorical_column_f = (
        fc_lib.sequence_categorical_column_with_identity(
            key=feature_f, num_buckets=4))
    sequence_categorical_column_g = (
        fc_lib.sequence_categorical_column_with_identity(
            key=feature_g, num_buckets=4))
    categorical_column_h = (
        fc_lib.categorical_column_with_identity(
            key=feature_h, num_buckets=4))

    table_a = 'tbl_a'
    table_bc = 'tbl_b_c_weighted_by_c_weight_shared_embedding'
    table_e = 'tbl_e'
    table_fgh = 'tbl_f_g_h_shared_embedding'
    embedding_dimension_a = 2
    embedding_dimension_bc = 5
    embedding_dimension_d = 2
    embedding_dimension_e = 3
    embedding_dimension_fgh = 4
    column_a = tpu_fc_v2.embedding_column_v2(
        categorical_column_a,
        dimension=embedding_dimension_a,
        combiner='mean',
        initializer=lambda: 'my_initializer_a')
    column_b, column_c = tpu_fc_v2.shared_embedding_columns_v2(
        [categorical_column_b, weighted_column_c],
        dimension=embedding_dimension_bc,
        combiner='sqrtn',
        initializer=lambda: 'my_initializer_b_c')
    column_d = tpu_fc_v2.embedding_column_v2(
        weighted_column_d,
        dimension=embedding_dimension_d,
        combiner='mean',
        initializer=lambda: 'my_initializer_d')
    sequence_column_e = tpu_fc_v2.embedding_column_v2(
        sequence_categorical_column_e,
        max_sequence_length=3,
        dimension=embedding_dimension_e,
        initializer=lambda: 'my_initializer_e')
    sequence_column_f, sequence_column_g, column_h = (
        tpu_fc_v2.shared_embedding_columns_v2(
            [sequence_categorical_column_f, sequence_categorical_column_g,
             categorical_column_h],
            max_sequence_lengths=[2, 1, 0],
            dimension=embedding_dimension_fgh,
            initializer=lambda: 'my_initializer_f_g_h'))

    table_to_config, feature_to_config = (
        _tpu_estimator_embedding.get_configs_from_feature_columns(
            [column_a, column_b, column_c, column_d, sequence_column_e,
             sequence_column_f, sequence_column_g, column_h]))

    self.assertEqual(feature_to_config[feature_a].table_id, table_a)
    self.assertEqual(feature_to_config[feature_b].table_id, table_bc)
    self.assertEqual(feature_to_config[feature_e].table_id, table_e)
    self.assertEqual(feature_to_config[feature_f].table_id, table_fgh)
    self.assertEqual(feature_to_config[feature_e].max_sequence_length, 3)
    self.assertEqual(feature_to_config[feature_f].max_sequence_length, 2)
    self.assertEqual(feature_to_config[feature_g].max_sequence_length, 1)
    self.assertEqual(feature_to_config[feature_h].max_sequence_length, 0)
    self.assertEqual(table_to_config[table_a].vocabulary_size, 3)
    self.assertEqual(table_to_config[table_bc].vocabulary_size, 6)
    self.assertEqual(table_to_config[table_e].vocabulary_size, 7)
    self.assertEqual(table_to_config[table_fgh].vocabulary_size, 4)
    self.assertEqual(table_to_config[table_a].dimension, embedding_dimension_a)
    self.assertEqual(table_to_config[table_bc].dimension,
                     embedding_dimension_bc)
    self.assertEqual(table_to_config[table_e].dimension, embedding_dimension_e)
    self.assertEqual(table_to_config[table_fgh].dimension,
                     embedding_dimension_fgh)
    self.assertEqual(table_to_config[table_a].combiner, 'mean')
    self.assertEqual(table_to_config[table_bc].combiner, 'sqrtn')
    self.assertEqual(table_to_config[table_a].initializer(), 'my_initializer_a')
    self.assertEqual(table_to_config[table_bc].initializer(),
                     'my_initializer_b_c')
    self.assertEqual(table_to_config[table_e].initializer(), 'my_initializer_e')
    self.assertEqual(table_to_config[table_fgh].initializer(),
                     'my_initializer_f_g_h')

    self.assertEqual(feature_to_config[feature_a].weight_key, None)
    self.assertEqual(feature_to_config[feature_b].weight_key, None)
    self.assertEqual(feature_to_config[feature_c].weight_key,
                     weight_feature_key_c)
    self.assertEqual(feature_to_config[feature_d].weight_key,
                     weight_feature_key_d)

  def _create_estimator_with_config_dicts(self,
                                          feature_to_config_dict,
                                          table_to_config_dict,
                                          use_cpu=False,
                                          partition_strategy='div',
                                          input_method=_PER_HOST_V2):
    """Returns TPUEstimator which uses `feature_columns` in model_fn."""

    def _model_fn(features, labels, mode, params):
      """Creates simple TF model using feature_columns to create input layer."""
      del params
      input_features = []
      for feature in features:
        if len(features[feature].shape) == 1:
          input_features.append(array_ops.expand_dims(features[feature], -1))
        elif len(features[feature].shape) > 2:
          input_features.append(
              array_ops.reshape(features[feature],
                                [features[feature].shape[0], -1]))
        else:
          input_features.append(features[feature])
        input_features = [
            math_ops.cast(feature, dtypes.float32)
            for feature in input_features]
        input_layer = array_ops.concat(input_features, -1)
        predictions = layers.dense(
            input_layer, 1, kernel_initializer=init_ops.zeros_initializer())

      loss = None
      train_op = None
      eval_metrics = None
      export_outputs = None
      if mode == model_fn_lib.ModeKeys.TRAIN:
        loss = losses.mean_squared_error(labels, predictions)
        optimizer = training.GradientDescentOptimizer(learning_rate=0.5)
        optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)
        train_op = optimizer.minimize(
            loss, global_step=training.get_global_step())
      elif mode == model_fn_lib.ModeKeys.EVAL:
        loss = losses.mean_squared_error(labels, predictions)

        def metric_fn_on_cpu(labels, predictions):
          return {
              'mse': metrics_lib.mean_absolute_error(labels, predictions),
          }

        eval_metrics = (metric_fn_on_cpu, [labels, predictions])

      else:
        export_outputs = {'prediction':
                          export_output_lib.PredictOutput(
                              {'prediction': predictions})}
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          train_op=train_op,
          loss=loss,
          eval_metrics=eval_metrics,
          export_outputs=export_outputs,
          predictions=predictions)

    run_config = create_run_config(
        iterations_per_loop=2,
        per_host_input_for_training=input_method)
    embedding_config_spec = tpu_estimator.EmbeddingConfigSpec(
        table_to_config_dict=table_to_config_dict,
        feature_to_config_dict=feature_to_config_dict,
        optimization_parameters=tpu_estimator.AdagradParameters(
            learning_rate=.01, initial_accumulator=0.1),
        partition_strategy=partition_strategy,
    )
    return tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        config=run_config,
        train_batch_size=8,
        eval_batch_size=8,
        use_tpu=not use_cpu,
        embedding_config_spec=embedding_config_spec,
        export_to_tpu=True)

  def _get_vocab_feature_columns(self,
                                 embedding_initializer=None,
                                 is_vocabulary_file=True):
    """Return feature columns for categorical_column_with_vocabulary_x tests."""
    if is_vocabulary_file:
      vocab_file = os.path.join(tempfile.mkdtemp(), 'vocab')
      with open(vocab_file, 'w') as f:
        f.write('\n'.join([str(i) for i in range(_VOCAB_SIZE)]))
      vocab_column = fc_lib.categorical_column_with_vocabulary_file(
          key='x',
          vocabulary_file=vocab_file,
          vocabulary_size=_VOCAB_SIZE,
          num_oov_buckets=_VOCAB_NUM_BUCKETS - _VOCAB_SIZE)
    else:
      vocab_list = [str(i) for i in range(_VOCAB_SIZE)]
      vocab_column = fc_lib.categorical_column_with_vocabulary_list(
          key='x',
          vocabulary_list=vocab_list,
          num_oov_buckets=_VOCAB_NUM_BUCKETS - _VOCAB_SIZE)

    feature_columns = [
        tpu_fc_v2.embedding_column_v2(
            categorical_column=vocab_column,
            dimension=_VOCAB_EMBEDDING_DIM,
            initializer=embedding_initializer),
    ]
    return set(feature_columns)

  def _get_vocab_input_fn_and_feature_columns(self,
                                              numeric_check=False,
                                              is_vocabulary_file=True,
                                              is_dataset=False):
    """Return input_fn and feature_columns for vocabulary column tests.

    Args:
      numeric_check: A boolean flag. When set to be True, the labels in input_fn
          are set to be the expected input_layer value from the input features
          and embedding initialization. This is to allow test to conveniently
          do numerical check by comparing the labels against input_layer in
          model_fn.
      is_vocabulary_file: use categorical_column_with_vocabulary_file when set
          True, else categorical_column_with_vocabulary_list.
      is_dataset: A boolean value indicating whether the input_fn returns
          dataset or not.

    Returns:
      A tuple consists of an input_fn and a set of feature columns.
    """

    # Initialize embedding to
    # 1 0 0 0 0 ..
    # 0 2 0 0 0 ..
    # 0 0 3 0 0 ..
    # 0 0 0 4 0 ..
    embedding_init = np.zeros((_VOCAB_SIZE, _VOCAB_EMBEDDING_DIM))
    for i in range(_VOCAB_SIZE):
      embedding_init[i, i] = i + 1
    embedding_initializer = init_ops.constant_initializer(embedding_init)

    def input_fn(params):
      # Data index is [3, 2, 1, 0]
      feature_data = sparse_tensor.SparseTensor(
          indices=[[i, 0] for i in range(_VOCAB_SIZE)],
          values=[str(_VOCAB_SIZE - 1 - i) for i in range(_VOCAB_SIZE)],
          dense_shape=[_VOCAB_SIZE, 1])

      if numeric_check:
        # Expected input_layer is
        # 0 0 0 4 0 ..
        # 0 0 3 0 0 ..
        # 0 2 0 0 0 ..
        # 1 0 0 0 0 ..
        labels = np.zeros((_VOCAB_SIZE, _VOCAB_EMBEDDING_DIM), dtype=np.float32)
        for i in range(_VOCAB_SIZE):
          labels[i, _VOCAB_SIZE - i - 1] = _VOCAB_SIZE - i
      else:
        labels = np.zeros((_VOCAB_SIZE, 1), dtype=np.float32)

      data = dataset_lib.Dataset.from_tensor_slices(({
          'x': feature_data,
      }, labels))
      data = data.repeat()
      data = data.batch(params['batch_size'], drop_remainder=True)
      if is_dataset:
        return data
      iterator = data.make_one_shot_iterator()
      return iterator.get_next()

    return input_fn, self._get_vocab_feature_columns(
        embedding_initializer, is_vocabulary_file=is_vocabulary_file)

  def test_feature_in_two_embeddings(self):
    sparse_column = fc_lib.categorical_column_with_identity(
        key='x', num_buckets=10)
    feature_columns = [
        tpu_fc_v2.embedding_column_v2(categorical_column=sparse_column,
                                      dimension=2),
        tpu_fc_v2.embedding_column_v2(categorical_column=sparse_column,
                                      dimension=4)]
    with self.assertRaisesRegex(
        ValueError, 'is used with multiple embeddings and this '
        'is not supported.'):
      est = self._create_estimator_with_feature_columns(
          feature_columns)
      est.train(input_fn=(lambda params: 'Not used'), steps=1)

  def _test_two_features(self, shared_embedding, sequence_column,
                         input_method, use_cpu=False):
    sparse_column1 = fc_lib.categorical_column_with_identity(
        key='x', num_buckets=10)
    if sequence_column:
      sparse_column2 = fc_lib.sequence_categorical_column_with_identity(
          key='y', num_buckets=10)
    else:
      sparse_column2 = fc_lib.categorical_column_with_identity(
          key='y', num_buckets=10)

    if shared_embedding:
      if sequence_column:
        feature_columns = tpu_fc_v2.shared_embedding_columns_v2(
            [sparse_column1, sparse_column2], dimension=2,
            max_sequence_lengths=[0, 2])
      else:
        feature_columns = tpu_fc_v2.shared_embedding_columns_v2(
            [sparse_column1, sparse_column2], dimension=2)
    else:
      if sequence_column:
        feature_columns = [
            tpu_fc_v2.embedding_column_v2(categorical_column=sparse_column1,
                                          dimension=2),
            tpu_fc_v2.embedding_column_v2(categorical_column=sparse_column2,
                                          dimension=4, max_sequence_length=2)]
      else:
        feature_columns = [
            tpu_fc_v2.embedding_column_v2(categorical_column=sparse_column1,
                                          dimension=2),
            tpu_fc_v2.embedding_column_v2(categorical_column=sparse_column2,
                                          dimension=4)]

    def _input_fn(params):
      feature1_data = dataset_lib.Dataset.from_tensor_slices(
          sparse_tensor.SparseTensor(
              indices=[[i, j] for i in range(params['batch_size'])
                       for j in [0, 1]],
              values=[1] * (2 * params['batch_size']),
              dense_shape=[params['batch_size'], 2]))
      feature2_data = dataset_lib.Dataset.from_tensor_slices(
          sparse_tensor.SparseTensor(
              indices=[[i, j] for i in range(params['batch_size'])
                       for j in [0, 1]],
              values=[2] * (2 * params['batch_size']),
              dense_shape=[params['batch_size'], 2]))
      labels_data = dataset_lib.Dataset.from_tensor_slices(
          np.array([[0]] * params['batch_size'], dtype=np.int32))
      dataset = dataset_lib.Dataset.zip(
          (feature1_data, feature2_data, labels_data))
      dataset = dataset.repeat()
      dataset = dataset.batch(params['batch_size'], drop_remainder=True)

      def _map(x, y, z):
        return {'x': x, 'y': y}, z

      dataset = dataset.map(_map)
      return dataset

    est = self._create_estimator_with_feature_columns(feature_columns,
                                                      use_cpu=use_cpu,
                                                      input_method=input_method)
    est.train(input_fn=_input_fn, steps=1)
    checkpoint_reader = training.NewCheckpointReader(
        training.latest_checkpoint(est.config.model_dir))
    return checkpoint_reader.get_variable_to_shape_map().keys()

  @parameterized.named_parameters(
      ('non_shared_non_sequence_v1', False, False, _PER_HOST_V1),
      ('shared_non_sequence_v1', True, False, _PER_HOST_V1),
      ('non_shared_sequence_v1', False, True, _PER_HOST_V1),
      ('shared_sequence_v1', True, True, _PER_HOST_V1),
      ('non_shared_non_sequence_v2', False, False, _PER_HOST_V2),
      ('shared_non_sequence_v2', True, False, _PER_HOST_V2),
      ('non_shared_sequence_v2', False, True, _PER_HOST_V2),
      ('shared_sequence_v2', True, True, _PER_HOST_V2))
  def test_two_features_with_config_dicts(self,
                                          shared_embedding,
                                          sequence_column,
                                          input_method):
    y_max_seq_length = 2 if sequence_column else 0
    y_table = 't1' if shared_embedding else 't2'
    feature_to_config_dict = {
        'x': tpu_embedding.FeatureConfig(table_id='t1'),
        'y': tpu_embedding.FeatureConfig(table_id=y_table,
                                         max_sequence_length=y_max_seq_length)
    }
    table_to_config_dict = {
        't1': tpu_embedding.TableConfig(vocabulary_size=10, dimension=2)
    }
    if not shared_embedding:
      table_to_config_dict['t2'] = tpu_embedding.TableConfig(
          vocabulary_size=10, dimension=4)

    def _input_fn(params):
      feature1_data = dataset_lib.Dataset.from_tensor_slices(
          sparse_tensor.SparseTensor(
              indices=list(
                  itertools.product(range(params['batch_size']), [0, 1])),
              values=[1] * (2 * params['batch_size']),
              dense_shape=[params['batch_size'], 2]))
      feature2_data = dataset_lib.Dataset.from_tensor_slices(
          sparse_tensor.SparseTensor(
              indices=list(
                  itertools.product(range(params['batch_size']), [0, 1])),
              values=[2] * (2 * params['batch_size']),
              dense_shape=[params['batch_size'], 2]))
      labels_data = dataset_lib.Dataset.from_tensor_slices(
          np.array([[0]] * params['batch_size'], dtype=np.int32))
      dataset = dataset_lib.Dataset.zip(
          (feature1_data, feature2_data, labels_data))
      dataset = dataset.repeat()
      dataset = dataset.batch(params['batch_size'], drop_remainder=True)

      def _map(x, y, z):
        return {'x': x, 'y': y}, z

      dataset = dataset.map(_map)
      return dataset

    est = self._create_estimator_with_config_dicts(feature_to_config_dict,
                                                   table_to_config_dict,
                                                   input_method=input_method)
    est.train(input_fn=_input_fn, steps=1)

  def test_non_tpu_embedding_column(self):
    sparse_column = fc_lib.categorical_column_with_identity(
        key='x', num_buckets=10)
    sparse_column2 = fc_lib.categorical_column_with_identity(
        key='y', num_buckets=10)
    feature_columns = [
        tpu_fc_v2.embedding_column_v2(
            categorical_column=sparse_column, dimension=2),
        fc_lib.embedding_column(categorical_column=sparse_column2, dimension=4)
    ]

    with self.assertRaisesRegex(TypeError, 'Unsupported feature column'):
      est = self._create_estimator_with_feature_columns(
          feature_columns)
      est.train(input_fn=(lambda params: 'Not used'), steps=1)

  def test_feature_in_embedding_and_shared_embedding(self):
    sparse_column1 = fc_lib.categorical_column_with_identity(
        key='x', num_buckets=10)
    sparse_column2 = fc_lib.categorical_column_with_identity(
        key='y', num_buckets=10)

    feature_columns = [
        tpu_fc_v2.embedding_column_v2(categorical_column=sparse_column1,
                                      dimension=2)
    ] + tpu_fc_v2.shared_embedding_columns_v2([sparse_column1, sparse_column2],
                                              dimension=4)

    with self.assertRaisesRegex(
        ValueError, 'is used with multiple embeddings and this '
        'is not supported.'):
      est = self._create_estimator_with_feature_columns(
          feature_columns)
      est.train(input_fn=(lambda params: 'Not used'), steps=1)

  def test_sequence_column_with_no_max_length(self):
    sparse_column = fc_lib.sequence_categorical_column_with_identity(
        key='x', num_buckets=10)
    with self.assertRaisesRegex(
        ValueError, 'max_sequence_length must be greater than 0 '
        'for sequence columns. Got max_sequence_length'
        '=0 for sequence column x.'):
      tpu_fc_v2.embedding_column_v2(categorical_column=sparse_column,
                                    dimension=2)

  def test_non_sequence_column_with_max_length(self):
    sparse_column = fc_lib.categorical_column_with_identity(
        key='x', num_buckets=10)
    with self.assertRaisesRegex(
        ValueError, 'Non zero max_seq_length=2 specified for non '
        'sequence column x.'):
      tpu_fc_v2.embedding_column_v2(categorical_column=sparse_column,
                                    dimension=2,
                                    max_sequence_length=2)

  def test_sequence_column_shared_embedding_wrong_max_sequence_length(self):
    sparse_column_x = fc_lib.sequence_categorical_column_with_identity(
        key='x', num_buckets=10)
    sparse_column_y = fc_lib.sequence_categorical_column_with_identity(
        key='y', num_buckets=10)
    with self.assertRaisesRegex(
        ValueError, 'max_sequence_lengths and categorical_columns must be of'):
      tpu_fc_v2.shared_embedding_columns_v2(
          categorical_columns=[sparse_column_x, sparse_column_y], dimension=2,
          max_sequence_lengths=[2])

  def test_sequence_column_shared_embedding_non_sequence_with_max_length(self):
    sparse_column_x = fc_lib.sequence_categorical_column_with_identity(
        key='x', num_buckets=10)
    sparse_column_y = fc_lib.categorical_column_with_identity(
        key='y', num_buckets=10)
    with self.assertRaisesRegex(ValueError,
                                'Non zero max_seq_length=1 specified for non'):
      tpu_fc_v2.shared_embedding_columns_v2(
          categorical_columns=[sparse_column_x, sparse_column_y], dimension=2,
          max_sequence_lengths=[2, 1])

  def test_sequence_column_shared_embedding_sequence_without_max_length(self):
    sparse_column_x = fc_lib.sequence_categorical_column_with_identity(
        key='x', num_buckets=10)
    sparse_column_y = fc_lib.categorical_column_with_identity(
        key='y', num_buckets=10)
    with self.assertRaisesRegex(ValueError,
                                'max_sequence_length must be greater than 0'):
      tpu_fc_v2.shared_embedding_columns_v2(
          categorical_columns=[sparse_column_x, sparse_column_y], dimension=2)

  @parameterized.named_parameters(
      ('per_host_v1', _PER_HOST_V1),
      ('per_host_v2', _PER_HOST_V2))
  def test_sequence_column_length(self, input_method):
    sequence_column = fc_lib.sequence_categorical_column_with_identity(
        key='x', num_buckets=10)
    feature_columns = [
        tpu_fc_v2.embedding_column_v2(
            categorical_column=sequence_column,
            dimension=4,
            max_sequence_length=10)
    ]

    def _input_fn(params):
      sequence_lengths = np.random.randint(1, 10, params['batch_size'])
      total = sum(sequence_lengths)
      indices = []
      for i in range(params['batch_size']):
        for j in range(sequence_lengths[i]):
          indices.append([i, j])
      feature_data = dataset_lib.Dataset.from_tensor_slices(
          sparse_tensor.SparseTensor(
              indices=indices,
              values=[1] * total,
              dense_shape=[params['batch_size'], 10])
      )
      labels_data = dataset_lib.Dataset.from_tensor_slices(
          np.array(sequence_lengths, dtype=np.float32))
      dataset = dataset_lib.Dataset.zip(
          (feature_data, labels_data))
      dataset = dataset.repeat()
      dataset = dataset.batch(params['batch_size'], drop_remainder=True)
      def _map(x, y):
        return {'x': x}, y
      dataset = dataset.map(_map)
      return dataset

    est = self._create_estimator_with_feature_columns(
        feature_columns, numeric_check=True, input_method=input_method)
    res = est.evaluate(input_fn=_input_fn, steps=1)
    self.assertAllClose(res['loss'], 0)

  def test_unknown_partition_strategy(self):
    feature_to_config_dict = {'x': tpu_embedding.FeatureConfig(table_id='t1')}
    table_to_config_dict = {
        't1': tpu_embedding.TableConfig(vocabulary_size=10, dimension=2)
    }
    with self.assertRaisesRegex(
        ValueError, 'Invalid partition_strategy invalid. Must be '
        'one of "mod" or "div".'):
      self._create_estimator_with_config_dicts(
          feature_to_config_dict, table_to_config_dict, use_cpu=True,
          partition_strategy='invalid')

  def test_mod_partition_strategy_on_cpu(self):
    feature_to_config_dict = {'x': tpu_embedding.FeatureConfig(table_id='t1')}
    table_to_config_dict = {
        't1': tpu_embedding.TableConfig(vocabulary_size=10, dimension=2)
    }
    with self.assertRaisesRegex(
        ValueError, 'Mod sharding of embedding tables not '
        'supported on CPU.'):
      self._create_estimator_with_config_dicts(
          feature_to_config_dict, table_to_config_dict, use_cpu=True,
          partition_strategy='mod')

  @parameterized.named_parameters(
      ('non_shared_non_sequence_v1', False, False, _PER_HOST_V1),
      ('shared_non_sequence_v1', True, False, _PER_HOST_V1),
      ('non_shared_sequence_v1', False, True, _PER_HOST_V1),
      ('shared_sequence_v1', True, True, _PER_HOST_V1),
      ('non_shared_non_sequence_v2', False, False, _PER_HOST_V2),
      ('shared_non_sequence_v2', True, False, _PER_HOST_V2),
      ('non_shared_sequence_v2', False, True, _PER_HOST_V2),
      ('shared_sequence_v2', True, True, _PER_HOST_V2))
  def test_two_features(self, shared, sequence, input_method):
    cpu_names = self._test_two_features(shared_embedding=shared,
                                        sequence_column=sequence,
                                        input_method=input_method,
                                        use_cpu=True)
    tpu_names = self._test_two_features(shared_embedding=shared,
                                        sequence_column=sequence,
                                        input_method=input_method,
                                        use_cpu=False)
    # TPU will have some extra variables but all CPU variables should be in the
    # TPU checkpoint
    for name in cpu_names:
      self.assertIn(name, tpu_names)

  @parameterized.named_parameters(
      ('per_host_v1', _PER_HOST_V1),
      ('per_host_v2', _PER_HOST_V2))
  def test_dynamic_learning_rate(self, input_method):
    sparse_column_a = fc_lib.categorical_column_with_identity(
        key='a', num_buckets=10)
    sparse_column_b = fc_lib.categorical_column_with_identity(
        key='b', num_buckets=10)
    sparse_column_c = fc_lib.categorical_column_with_identity(
        key='c', num_buckets=10)
    sparse_column_d = fc_lib.categorical_column_with_identity(
        key='d', num_buckets=10)
    sparse_column_e = fc_lib.categorical_column_with_identity(
        key='e', num_buckets=10)
    sparse_column_f = fc_lib.categorical_column_with_identity(
        key='f', num_buckets=10)

    static_lr = 1
    def dynamic_learning_rate(global_step):
      return control_flow_ops.cond(
          math_ops.equal(global_step, 0), lambda: 2, lambda: 0)

    def shared_dynamic_learning_rate(global_step):
      return control_flow_ops.cond(
          math_ops.equal(global_step, 0), lambda: 3, lambda: 0)

    embedding_column_static = tpu_fc_v2.embedding_column_v2(
        categorical_column=sparse_column_a,
        dimension=2,
        initializer=init_ops.Ones())
    embedding_column_dynamic = tpu_fc_v2.embedding_column_v2(
        categorical_column=sparse_column_b,
        dimension=2,
        initializer=init_ops.Ones(),
        learning_rate_fn=dynamic_learning_rate)
    shared_embedding_columns_static = tpu_fc_v2.shared_embedding_columns_v2(
        [sparse_column_c, sparse_column_d],
        dimension=2,
        initializer=init_ops.Ones())
    shared_embedding_columns_dynamic = tpu_fc_v2.shared_embedding_columns_v2(
        [sparse_column_e, sparse_column_f],
        dimension=2,
        initializer=init_ops.Ones(),
        learning_rate_fn=shared_dynamic_learning_rate)
    feature_columns = ([embedding_column_static] + [embedding_column_dynamic] +
                       shared_embedding_columns_static +
                       shared_embedding_columns_dynamic)

    def _input_fn(params):
      feature_indices = [[0, 0], [1, 0], [1, 1], [1, 2]]
      feature_values = [3, 0, 1, 2]
      feature = sparse_tensor.SparseTensor(
          indices=feature_indices,
          values=feature_values,
          dense_shape=[2, 3])
      feature_datas = tuple(
          dataset_lib.Dataset.from_tensor_slices(feature) for _ in range(6))
      labels_data = dataset_lib.Dataset.from_tensor_slices(
          np.array([[0]] * 2, dtype=np.int32))
      dataset = dataset_lib.Dataset.zip(feature_datas + (labels_data,))
      dataset = dataset.repeat()

      def _map(a, b, c, d, e, f, z):
        return {'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f}, z

      dataset = dataset.map(_map)
      dataset = dataset.batch(params['batch_size'], drop_remainder=True)
      return dataset

    def _model_fn(features, labels, mode, params):
      """Creates simple TF model using feature_columns to create input layer."""
      del params
      assert mode == model_fn_lib.ModeKeys.TRAIN

      dense_layer = fc_lib.DenseFeatures(feature_columns)
      input_layer = dense_layer(features)
      predictions = layers.dense(
          input_layer, 1, kernel_initializer=init_ops.ones_initializer())

      loss = losses.mean_squared_error(labels, predictions)
      optimizer = training.AdagradOptimizer(learning_rate=0.5)
      optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)
      train_op = optimizer.minimize(
          loss, global_step=training.get_global_step())
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          train_op=train_op,
          loss=loss)

    run_config = create_run_config(
        iterations_per_loop=1,
        per_host_input_for_training=input_method)
    optimization_parameters = (
        tpu_estimator.StochasticGradientDescentParameters(
            learning_rate=static_lr))
    embedding_config_spec = tpu_estimator.EmbeddingConfigSpec(
        feature_columns=feature_columns,
        optimization_parameters=optimization_parameters)
    est = tpu_estimator.TPUEstimator(
        model_fn=_model_fn,
        config=run_config,
        train_batch_size=4,
        embedding_config_spec=embedding_config_spec)
    est.train(input_fn=_input_fn, steps=1)

    checkpoint_reader = training.NewCheckpointReader(
        training.latest_checkpoint(est.config.model_dir))
    embedding_static = checkpoint_reader.get_tensor(
        'dense_features/a_embedding/embedding_weights')
    embedding_dynamic = checkpoint_reader.get_tensor(
        'dense_features/b_embedding/embedding_weights')
    shared_embedding_static = checkpoint_reader.get_tensor(
        'c_d_shared_embedding')
    shared_embedding_dynamic = checkpoint_reader.get_tensor(
        'e_f_shared_embedding')

    unit_update = embedding_static - 1.
    unit_update_shared = shared_embedding_static - 1.
    # The asserts below are only valid if unit updates are not all zero.
    self.assertFalse(np.allclose(unit_update, 0.))
    self.assertFalse((np.allclose(unit_update_shared, 0.)))
    self.assertAllClose(embedding_dynamic - 1.,
                        unit_update * 2)
    self.assertAllClose(shared_embedding_dynamic - 1.,
                        unit_update_shared * 3)

    # train for another step
    est.train(input_fn=_input_fn, steps=1)

    checkpoint_reader2 = training.NewCheckpointReader(
        training.latest_checkpoint(est.config.model_dir))
    embedding_static2 = checkpoint_reader2.get_tensor(
        'dense_features/a_embedding/embedding_weights')
    embedding_dynamic2 = checkpoint_reader2.get_tensor(
        'dense_features/b_embedding/embedding_weights')
    shared_embedding_static2 = checkpoint_reader2.get_tensor(
        'c_d_shared_embedding')
    shared_embedding_dynamic2 = checkpoint_reader2.get_tensor(
        'e_f_shared_embedding')

    self.assertFalse(np.allclose(embedding_static,
                                 embedding_static2))
    self.assertFalse((np.allclose(shared_embedding_static,
                                  shared_embedding_static2)))
    self.assertAllClose(embedding_dynamic, embedding_dynamic2)
    self.assertAllClose(shared_embedding_dynamic, shared_embedding_dynamic2)


class TPUEstimatorWeightedFeatureColumnTest(TPUEstimatorFeatureColumnTestBase,
                                            parameterized.TestCase):

  @parameterized.named_parameters(
      ('per_host_v1', _PER_HOST_V1),
      ('per_host_v2', _PER_HOST_V2))
  def test_embedding_with_weighted_categorical_column(self, input_method):
    num_buckets = 3
    embedding_dim = 5
    sparse_id_column = fc_lib.categorical_column_with_identity(
        key='ids', num_buckets=num_buckets)
    weighted_sparse_id_column = fc_lib.weighted_categorical_column(
        categorical_column=sparse_id_column, weight_feature_key='values')

    embedding_init = np.zeros((num_buckets, embedding_dim))
    # Embedding initialized to
    # 1 1 1 1 1
    # 2 2 2 2 2
    # 3 3 3 3 3
    for i in range(num_buckets):
      embedding_init[i, :] = [i + 1] * embedding_dim

    feature_columns = [
        tpu_fc_v2.embedding_column_v2(
            categorical_column=weighted_sparse_id_column,
            dimension=embedding_dim,
            combiner='mean',
            initializer=init_ops.constant_initializer(embedding_init))
    ]

    def _input_fn(params):
      sample_size = 2
      dense_shape = (sample_size, num_buckets)
      indices = ((0, 0), (0, 2), (1, 0), (1, 1))
      id_values = (2, 1, 1, 0)
      weight_values = (0.5, 1.0, 0.2, 0.0)

      inputs = sparse_tensor.SparseTensor(
          indices=indices, values=id_values, dense_shape=dense_shape)
      weights = sparse_tensor.SparseTensor(
          indices=indices, values=weight_values, dense_shape=dense_shape)

      # Setup labels so that the loss is zero
      labels = np.zeros((sample_size, embedding_dim), dtype=np.float32)
      for j in range(embedding_dim):
        # "mean" is the weighted sum divided by the total weight.
        labels[0, j] = (3 * 0.5 + 2 * 1.0) / (0.5 + 1.0)
        labels[1, j] = (2 * 0.2 + 1 * 0.0) / (0.2 + 0.0)

      data = dataset_lib.Dataset.from_tensor_slices(({
          'ids': inputs,
          'values': weights
      }, labels))
      data = data.repeat()
      data = data.batch(params['batch_size'], drop_remainder=True)
      return data

    est = self._create_estimator_with_feature_columns(
        feature_columns, numeric_check=True, input_method=input_method)
    est.train(input_fn=_input_fn, steps=1)
    res = est.evaluate(input_fn=_input_fn, steps=1)
    self.assertAllClose(res['loss'], 0)

  @parameterized.named_parameters(
      ('per_host_v1', _PER_HOST_V1),
      ('per_host_v2', _PER_HOST_V2))
  def test_shared_embedding_with_weighted_categorical_column_and_dataset(
      self, input_method):
    num_buckets = 3
    embedding_dim = 5
    sparse_id_column1 = fc_lib.categorical_column_with_identity(
        key='ids1', num_buckets=num_buckets)
    weighted_sparse_id_column = fc_lib.weighted_categorical_column(
        categorical_column=sparse_id_column1, weight_feature_key='values')
    sparse_id_column2 = fc_lib.categorical_column_with_identity(
        key='ids2', num_buckets=num_buckets)

    # Embedding initialized to
    # 1 1 1 1 1
    # 2 2 2 2 2
    # 3 3 3 3 3
    embedding_init = np.zeros((num_buckets, embedding_dim))
    for i in range(num_buckets):
      embedding_init[i, :] = [i + 1] * embedding_dim

    feature_columns = tpu_fc_v2.shared_embedding_columns_v2(
        categorical_columns=[weighted_sparse_id_column, sparse_id_column2],
        dimension=embedding_dim,
        combiner='sum',
        initializer=init_ops.constant_initializer(embedding_init))

    def _input_fn(params):
      sample_size = 2
      dense_shape = (sample_size, num_buckets)
      id1_indices = ((0, 0), (0, 2), (1, 0), (1, 1))
      id1_values = (2, 1, 1, 0)
      id1_weight_values = (1, 2, 3, 0)  # test integer weights
      id2_indices = ((0, 1), (1, 0), (1, 2))
      id2_values = (1, 2, 0)

      inputs1 = sparse_tensor.SparseTensor(
          indices=id1_indices, values=id1_values, dense_shape=dense_shape)
      inputs1_weights = sparse_tensor.SparseTensor(
          indices=id1_indices,
          values=id1_weight_values,
          dense_shape=dense_shape)
      inputs2 = sparse_tensor.SparseTensor(
          indices=id2_indices, values=id2_values, dense_shape=dense_shape)

      # Setup labels so that the loss is zero
      labels = np.zeros((2, embedding_dim * 2), dtype=np.float32)
      for j in range(embedding_dim):
        labels[0, j] = 3 * 1 + 2 * 2
        labels[1, j] = 2 * 3
      for j in range(embedding_dim, embedding_dim * 2):
        labels[0, j] = 2
        labels[1, j] = 3 + 1

      data = dataset_lib.Dataset.from_tensor_slices(({
          'ids1': inputs1,
          'ids2': inputs2,
          'values': inputs1_weights
      }, labels))
      data = data.repeat()
      data = data.batch(params['batch_size'], drop_remainder=True)
      return data

    est = self._create_estimator_with_feature_columns(
        feature_columns, numeric_check=True, input_method=input_method)
    est.train(input_fn=_input_fn, steps=1)
    res = est.evaluate(input_fn=_input_fn, steps=1)
    self.assertAllClose(res['loss'], 0)

  def test_embedding_with_with_weighted_categorical_column_with_vocab_error(
      self):
    vocab_list = [str(i) for i in range(_VOCAB_SIZE)]
    vocab_column = fc_lib.categorical_column_with_vocabulary_list(
        key='x',
        vocabulary_list=vocab_list,
        num_oov_buckets=_VOCAB_NUM_BUCKETS - _VOCAB_SIZE)
    weighted_vocab_column = fc_lib.weighted_categorical_column(
        categorical_column=vocab_column, weight_feature_key='values')

    # Embedding initialized to
    # 0 0 0 0 0 ...
    # 1 1 1 1 1 ...
    # 2 2 2 2 2 ...
    # 3 3 3 3 3 ...
    embedding_init = np.zeros((_VOCAB_SIZE, _VOCAB_EMBEDDING_DIM))
    for i in range(_VOCAB_SIZE):
      embedding_init[i, :] = [i] * _VOCAB_EMBEDDING_DIM
    embedding_initializer = init_ops.constant_initializer(embedding_init)

    feature_columns = [
        tpu_fc_v2.embedding_column_v2(
            categorical_column=weighted_vocab_column,
            dimension=_VOCAB_EMBEDDING_DIM,
            initializer=embedding_initializer),
    ]

    def _input_fn(params):
      # Dense data after vocab -> integer conversion
      # 2 _ 1 _ _
      # 1 3 _ _ _
      sample_size = 2
      dense_shape = (sample_size, _VOCAB_NUM_BUCKETS)
      indices = ((0, 0), (0, 2), (1, 0), (1, 1))
      id_values = [str(id_value) for id_value in (2, 1, 1, 0)]
      weight_values = (0.5, 1.0, 0.2, 0.0)
      inputs = sparse_tensor.SparseTensor(
          indices=indices, values=(id_values), dense_shape=dense_shape)

      inputs = sparse_tensor.SparseTensor(
          indices=indices, values=id_values, dense_shape=dense_shape)
      weights = sparse_tensor.SparseTensor(
          indices=indices, values=weight_values, dense_shape=dense_shape)

      # setup labels to be the same as what input_layer so the loss is zero
      # Expected input_layer is
      labels = np.zeros((sample_size, _VOCAB_EMBEDDING_DIM), dtype=np.float32)
      for j in range(_VOCAB_EMBEDDING_DIM):
        # "mean" is the weighted sum divided by the total weight.
        labels[0, j] = (2 * 0.5 + 1 * 1.0) / (0.5 + 1.0)
        labels[1, j] = (1 * 0.2 + 3 * 0.0) / (0.2 + 0.0)

      data = dataset_lib.Dataset.from_tensor_slices(({
          'x': inputs,
          'values': weights,
      }, labels))
      data = data.repeat()
      data = data.batch(params['batch_size'], drop_remainder=True)
      return data

    est = self._create_estimator_with_feature_columns(
        feature_columns, numeric_check=True)
    with self.assertRaisesRegex(
        ValueError, 'SparseTensor with string as values are not supported.'):
      est.train(input_fn=_input_fn, steps=1)

  def test_embedding_with_weighted_categorical_column_dense_weights_error(self):
    num_buckets = 5
    embedding_dim = 10
    sparse_id_column = fc_lib.categorical_column_with_identity(
        key='ids', num_buckets=num_buckets)
    weighted_sparse_id_column = fc_lib.weighted_categorical_column(
        categorical_column=sparse_id_column, weight_feature_key='values')

    feature_columns = [
        tpu_fc_v2.embedding_column_v2(
            categorical_column=weighted_sparse_id_column,
            dimension=embedding_dim)
    ]

    def _input_fn(params):
      sample_size = 2
      dense_shape = (sample_size, num_buckets)
      indices = ((0, 0), (0, 2), (1, 0), (1, 1))
      id_values = (2, 1, 1, 0)
      weight_values = (0.5, 1.0, 0.2, 0.0)

      inputs = sparse_tensor.SparseTensor(
          indices=indices, values=id_values, dense_shape=dense_shape)
      weights = sparse_ops.sparse_tensor_to_dense(
          sparse_tensor.SparseTensor(
              indices=indices, values=weight_values, dense_shape=dense_shape))

      labels = np.zeros((sample_size, embedding_dim), dtype=np.float32)

      data = dataset_lib.Dataset.from_tensor_slices(({
          'ids': inputs,
          'values': weights
      }, labels))
      data = data.repeat()
      data = data.batch(params['batch_size'], drop_remainder=True)
      return data

    est = self._create_estimator_with_feature_columns(
        feature_columns, numeric_check=True)
    with self.assertRaisesRegex(ValueError,
                                'Dense weights are not supported on TPU'):
      est.train(input_fn=_input_fn, steps=1)

  def test_embedding_with_weighted_categorical_column_share_weights_error(self):
    num_buckets = 5
    embedding_dim = 10
    sparse_id_column1 = fc_lib.categorical_column_with_identity(
        key='ids1', num_buckets=num_buckets)
    weighted_sparse_id_column1 = fc_lib.weighted_categorical_column(
        categorical_column=sparse_id_column1, weight_feature_key='values')
    sparse_id_column2 = fc_lib.categorical_column_with_identity(
        key='ids2', num_buckets=num_buckets)
    weighted_sparse_id_column2 = fc_lib.weighted_categorical_column(
        categorical_column=sparse_id_column2, weight_feature_key='values')

    feature_columns = tpu_fc_v2.shared_embedding_columns_v2(
        categorical_columns=[
            weighted_sparse_id_column1, weighted_sparse_id_column2
        ],
        dimension=embedding_dim)

    def _input_fn(params):
      sample_size = 2
      dense_shape = (sample_size, num_buckets)
      indices = ((0, 0), (0, 2), (1, 0), (1, 1))
      id_values = (2, 1, 1, 0)
      weight_values = (0.5, 1.0, 0.2, 0.0)

      inputs1 = sparse_tensor.SparseTensor(
          indices=indices, values=id_values, dense_shape=dense_shape)
      inputs2 = sparse_tensor.SparseTensor(
          indices=indices, values=id_values, dense_shape=dense_shape)
      weights = sparse_tensor.SparseTensor(
          indices=indices, values=weight_values, dense_shape=dense_shape)

      labels = np.zeros((sample_size, embedding_dim), dtype=np.float32)

      data = dataset_lib.Dataset.from_tensor_slices(({
          'ids1': inputs1,
          'ids2': inputs2,
          'values': weights
      }, labels))
      data = data.repeat()
      data = data.batch(params['batch_size'], drop_remainder=True)
      return data

    est = self._create_estimator_with_feature_columns(
        feature_columns, numeric_check=True)
    with self.assertRaisesRegex(
        ValueError,
        'Please check if the weights are present in feature dict. Also note'
        ' weight-sharing among weighted_categorical_column is not supported on '
        'TPU.'):
      est.train(input_fn=_input_fn, steps=1)

  def _test_tensor_core_embedding(self,
                                  shared_embedding,
                                  both_embeddings,
                                  input_method,
                                  use_cpu=False):
    sparse_column1 = fc_lib.categorical_column_with_identity(
        key='x', num_buckets=10)
    sparse_column2 = fc_lib.categorical_column_with_identity(
        key='y', num_buckets=10)

    if shared_embedding:
      feature_columns = tpu_fc_v2.shared_embedding_columns_v2(
          [sparse_column1, sparse_column2],
          dimension=2,
          embedding_lookup_device='tpu_tensor_core',
          tensor_core_shape=[None, 2])
    else:
      feature_columns = [
          tpu_fc_v2.embedding_column_v2(
              categorical_column=sparse_column1,
              dimension=2,
              embedding_lookup_device='tpu_tensor_core',
              tensor_core_shape=[None, 2]),
      ]
      if both_embeddings:
        feature_columns.append(
            tpu_fc_v2.embedding_column_v2(
                categorical_column=sparse_column2,
                dimension=4,
                embedding_lookup_device='tpu_tensor_core',
                tensor_core_shape=[None, 2]))
      else:
        feature_columns.append(
            tpu_fc_v2.embedding_column_v2(
                categorical_column=sparse_column2, dimension=4))

    def _input_fn(params):
      indices = []
      for i in range(params['batch_size']):
        for j in [0, 1]:
          indices.append([i, j])
      feature1_data = dataset_lib.Dataset.from_tensor_slices(
          sparse_tensor.SparseTensor(
              indices=indices,
              values=[1] * (2 * params['batch_size']),
              dense_shape=[params['batch_size'], 2]))
      feature2_data = dataset_lib.Dataset.from_tensor_slices(
          sparse_tensor.SparseTensor(
              indices=indices,
              values=[2] * (2 * params['batch_size']),
              dense_shape=[params['batch_size'], 2]))
      labels_data = dataset_lib.Dataset.from_tensor_slices(
          np.array([[0]] * params['batch_size'], dtype=np.int32))
      dataset = dataset_lib.Dataset.zip(
          (feature1_data, feature2_data, labels_data))
      dataset = dataset.repeat()
      dataset = dataset.batch(params['batch_size'], drop_remainder=True)

      def _map(x, y, z):
        return {'x': x, 'y': y}, z

      dataset = dataset.map(_map)
      return dataset

    est = self._create_estimator_with_feature_columns(
        feature_columns, use_cpu=use_cpu, input_method=input_method)
    est.train(input_fn=_input_fn, steps=1)
    checkpoint_reader = training.NewCheckpointReader(
        training.latest_checkpoint(est.config.model_dir))
    return checkpoint_reader.get_variable_to_shape_map().keys()

  @parameterized.named_parameters(
      ('non_shared_single_v1', False, False, _PER_HOST_V1),
      ('non_shared_both_v1', False, True, _PER_HOST_V1),
      ('shared_v1', True, True, _PER_HOST_V1),
      ('non_shared_single_v2', False, False, _PER_HOST_V2),
      ('non_shared_both_v2', False, True, _PER_HOST_V2),
      ('shared_v2', True, True, _PER_HOST_V2))
  def test_tensor_core_embedding(self, shared, both_embeddings, input_method):
    cpu_names = self._test_tensor_core_embedding(
        shared_embedding=shared,
        both_embeddings=both_embeddings,
        input_method=input_method,
        use_cpu=True)
    tpu_names = self._test_tensor_core_embedding(
        shared_embedding=shared,
        both_embeddings=both_embeddings,
        input_method=input_method,
        use_cpu=False)
    # TPU will have some extra variables but all CPU variables should be in the
    # TPU checkpoint
    for name in cpu_names:
      self.assertIn(name, tpu_names)


if __name__ == '__main__':
  test.main()
