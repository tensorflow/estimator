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
"""Tests for feature_column."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.framework import test_util
from tensorflow.python.platform import flags
from tensorflow_estimator.python.estimator.canned import linear


def _initialized_session(config=None):
  sess = tf.compat.v1.Session(config=config)
  sess.run(tf.compat.v1.global_variables_initializer())
  sess.run(tf.compat.v1.tables_initializer())
  return sess


def get_linear_model_bias(name='linear_model'):
  with tf.compat.v1.variable_scope(name, reuse=True):
    return tf.compat.v1.get_variable('bias_weights')


def get_linear_model_column_var(column, name='linear_model'):
  return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                     name + '/' + column.name)[0]


class BaseFeatureColumnForTests(tf.__internal__.feature_column.FeatureColumn):
  """A base FeatureColumn useful to avoid boiler-plate in tests.

  Provides dummy implementations for abstract methods that raise ValueError in
  order to avoid re-defining all abstract methods for each test sub-class.
  """

  @property
  def parents(self):
    raise ValueError('Should not use this method.')

  @classmethod
  def from_config(cls, config, custom_objects=None, columns_by_name=None):
    raise ValueError('Should not use this method.')

  def get_config(self):
    raise ValueError('Should not use this method.')


class SortableFeatureColumnTest(tf.test.TestCase):

  @test_util.run_deprecated_v1
  def test_linear_model(self):
    price = tf.feature_column.numeric_column('price')
    with tf.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      model = linear.LinearModel([price])
      predictions = model(features)
      price_var, bias = model.variables
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        self.assertAllClose([[0.]], self.evaluate(price_var))
        self.assertAllClose([[0.], [0.]], self.evaluate(predictions))
        sess.run(price_var.assign([[10.]]))
        self.assertAllClose([[10.], [50.]], self.evaluate(predictions))

  @test_util.run_deprecated_v1
  def test_linear_model_sanitizes_scope_names(self):
    price = tf.feature_column.numeric_column('price > 100')
    with tf.Graph().as_default():
      features = {'price > 100': [[1.], [5.]]}
      model = linear.LinearModel([price])
      predictions = model(features)
      price_var, bias = model.variables
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        self.assertAllClose([[0.]], self.evaluate(price_var))
        self.assertAllClose([[0.], [0.]], self.evaluate(predictions))
        sess.run(price_var.assign([[10.]]))
        self.assertAllClose([[10.], [50.]], self.evaluate(predictions))


class BucketizedColumnTest(tf.test.TestCase):

  def test_linear_model_one_input_value(self):
    """Tests linear_model() for input with shape=[1]."""
    price = tf.feature_column.numeric_column('price', shape=[1])
    bucketized_price = tf.feature_column.bucketized_column(
        price, boundaries=[0, 2, 4, 6])
    with tf.Graph().as_default():
      features = {'price': [[-1.], [1.], [5.], [6.]]}
      model = linear.LinearModel([bucketized_price])
      predictions = model(features)
      bucketized_price_var, bias = model.variables
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        # One weight variable per bucket, all initialized to zero.
        self.assertAllClose([[0.], [0.], [0.], [0.], [0.]],
                            self.evaluate(bucketized_price_var))
        self.assertAllClose([[0.], [0.], [0.], [0.]],
                            self.evaluate(predictions))
        sess.run(
            bucketized_price_var.assign([[10.], [20.], [30.], [40.], [50.]]))
        # price -1. is in the 0th bucket, whose weight is 10.
        # price 1. is in the 1st bucket, whose weight is 20.
        # price 5. is in the 3rd bucket, whose weight is 40.
        # price 6. is in the 4th bucket, whose weight is 50.
        self.assertAllClose([[10.], [20.], [40.], [50.]],
                            self.evaluate(predictions))
        sess.run(bias.assign([1.]))
        self.assertAllClose([[11.], [21.], [41.], [51.]],
                            self.evaluate(predictions))

  def test_linear_model_two_input_values(self):
    """Tests linear_model() for input with shape=[2]."""
    price = tf.feature_column.numeric_column('price', shape=[2])
    bucketized_price = tf.feature_column.bucketized_column(
        price, boundaries=[0, 2, 4, 6])
    with tf.Graph().as_default():
      features = {'price': [[-1., 1.], [5., 6.]]}
      model = linear.LinearModel([bucketized_price])
      predictions = model(features)
      bucketized_price_var, bias = model.variables
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        # One weight per bucket per input column, all initialized to zero.
        self.assertAllClose(
            [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
            self.evaluate(bucketized_price_var))
        self.assertAllClose([[0.], [0.]], self.evaluate(predictions))
        sess.run(
            bucketized_price_var.assign([[10.], [20.], [30.], [40.], [50.],
                                         [60.], [70.], [80.], [90.], [100.]]))
        # 1st example:
        #   price -1. is in the 0th bucket, whose weight is 10.
        #   price 1. is in the 6th bucket, whose weight is 70.
        # 2nd example:
        #   price 5. is in the 3rd bucket, whose weight is 40.
        #   price 6. is in the 9th bucket, whose weight is 100.
        self.assertAllClose([[80.], [140.]], self.evaluate(predictions))
        sess.run(bias.assign([1.]))
        self.assertAllClose([[81.], [141.]], self.evaluate(predictions))


class HashedCategoricalColumnTest(tf.test.TestCase):

  @test_util.run_deprecated_v1
  def test_linear_model(self):
    wire_column = tf.feature_column.categorical_column_with_hash_bucket(
        'wire', 4)
    self.assertEqual(4, wire_column.num_buckets)
    with tf.Graph().as_default():
      model = linear.LinearModel((wire_column,))
      predictions = model({
          wire_column.name:
              tf.compat.v1.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=('marlo', 'skywalker', 'omar'),
                  dense_shape=(2, 2))
      })
      wire_var, bias = model.variables

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,), (0.,)), self.evaluate(wire_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(wire_var.assign(((1.,), (2.,), (3.,), (4.,))))
      # 'marlo' -> 3: wire_var[3] = 4
      # 'skywalker' -> 2, 'omar' -> 2: wire_var[2] + wire_var[2] = 3+3 = 6
      self.assertAllClose(((4.,), (6.,)), self.evaluate(predictions))


class CrossedColumnTest(tf.test.TestCase):

  @test_util.run_deprecated_v1
  def test_linear_model(self):
    """Tests linear_model.

    Uses data from test_get_sparse_tensors_simple.
    """
    a = tf.feature_column.numeric_column('a', dtype=tf.int32, shape=(2,))
    b = tf.feature_column.bucketized_column(a, boundaries=(0, 1))
    crossed = tf.feature_column.crossed_column([b, 'c'],
                                               hash_bucket_size=5,
                                               hash_key=5)
    with tf.Graph().as_default():
      model = linear.LinearModel((crossed,))
      predictions = model({
          'a':
              tf.constant(((-1., .5), (.5, 1.))),
          'c':
              tf.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=['cA', 'cB', 'cC'],
                  dense_shape=(2, 2)),
      })
      crossed_var, bias = model.variables
      with _initialized_session() as sess:
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,), (0.,), (0.,)),
                            self.evaluate(crossed_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        sess.run(crossed_var.assign(((1.,), (2.,), (3.,), (4.,), (5.,))))
        # Expected ids after cross = (1, 0, 1, 3, 4, 2)
        self.assertAllClose(((3.,), (14.,)), self.evaluate(predictions))
        sess.run(bias.assign((.1,)))
        self.assertAllClose(((3.1,), (14.1,)), self.evaluate(predictions))

  def test_linear_model_with_weights(self):

    class _TestColumnWithWeights(BaseFeatureColumnForTests,
                                 fc.CategoricalColumn):
      """Produces sparse IDs and sparse weights."""

      @property
      def _is_v2_column(self):
        return True

      @property
      def name(self):
        return 'test_column'

      @property
      def parse_example_spec(self):
        return {
            self.name: tf.io.VarLenFeature(tf.int32),
            '{}_weights'.format(self.name): tf.io.VarLenFeature(tf.float32),
        }

      @property
      def num_buckets(self):
        return 5

      def transform_feature(self, transformation_cache, state_manager):
        return (transformation_cache.get(self.name, state_manager),
                transformation_cache.get('{}_weights'.format(self.name),
                                         state_manager))

      def get_sparse_tensors(self, transformation_cache, state_manager):
        """Populates both id_tensor and weight_tensor."""
        ids_and_weights = transformation_cache.get(self, state_manager)
        return fc.CategoricalColumn.IdWeightPair(
            id_tensor=ids_and_weights[0], weight_tensor=ids_and_weights[1])

    t = _TestColumnWithWeights()
    crossed = tf.feature_column.crossed_column([t, 'c'],
                                               hash_bucket_size=5,
                                               hash_key=5)
    with tf.Graph().as_default():
      with self.assertRaisesRegexp(
          ValueError,
          'crossed_column does not support weight_tensor.*{}'.format(t.name)):
        model = linear.LinearModel((crossed,))
        model({
            t.name:
                tf.SparseTensor(
                    indices=((0, 0), (1, 0), (1, 1)),
                    values=[0, 1, 2],
                    dense_shape=(2, 2)),
            '{}_weights'.format(t.name):
                tf.SparseTensor(
                    indices=((0, 0), (1, 0), (1, 1)),
                    values=[1., 10., 2.],
                    dense_shape=(2, 2)),
            'c':
                tf.SparseTensor(
                    indices=((0, 0), (1, 0), (1, 1)),
                    values=['cA', 'cB', 'cC'],
                    dense_shape=(2, 2)),
        })


class LinearModelTest(tf.test.TestCase):

  def test_raises_if_empty_feature_columns(self):
    with self.assertRaisesRegexp(ValueError,
                                 'feature_columns must not be empty'):
      linear.LinearModel(feature_columns=[])

  def test_should_be_feature_column(self):
    with self.assertRaisesRegexp(ValueError, 'must be a FeatureColumn'):
      linear.LinearModel(feature_columns='NotSupported')

  def test_should_be_dense_or_categorical_column(self):

    class NotSupportedColumn(BaseFeatureColumnForTests):

      @property
      def _is_v2_column(self):
        return True

      @property
      def name(self):
        return 'NotSupportedColumn'

      def transform_feature(self, transformation_cache, state_manager):
        pass

      @property
      def parse_example_spec(self):
        pass

    with self.assertRaisesRegexp(
        ValueError, 'must be either a DenseColumn or CategoricalColumn'):
      linear.LinearModel(feature_columns=[NotSupportedColumn()])

  def test_does_not_support_dict_columns(self):
    with self.assertRaisesRegexp(
        ValueError, 'Expected feature_columns to be iterable, found dict.'):
      linear.LinearModel(
          feature_columns={'a': tf.feature_column.numeric_column('a')})

  def test_raises_if_duplicate_name(self):
    with self.assertRaisesRegexp(
        ValueError, 'Duplicate feature column name found for columns'):
      linear.LinearModel(feature_columns=[
          tf.feature_column.numeric_column('a'),
          tf.feature_column.numeric_column('a')
      ])

  def test_not_dict_input_features(self):
    price = tf.feature_column.numeric_column('price')
    with tf.Graph().as_default():
      features = [[1.], [5.]]
      model = linear.LinearModel([price])
      with self.assertRaisesRegexp(ValueError, 'We expected a dictionary here'):
        model(features)

  def test_dense_bias(self):
    price = tf.feature_column.numeric_column('price')
    with tf.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      model = linear.LinearModel([price])
      predictions = model(features)
      price_var, bias = model.variables
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        sess.run(price_var.assign([[10.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[15.], [55.]], self.evaluate(predictions))

  def test_sparse_bias(self):
    wire_cast = tf.feature_column.categorical_column_with_hash_bucket(
        'wire_cast', 4)
    with tf.Graph().as_default():
      wire_tensor = tf.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor}
      model = linear.LinearModel([wire_cast])
      predictions = model(features)
      wire_cast_var, bias = model.variables
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        self.assertAllClose([[0.], [0.], [0.], [0.]],
                            self.evaluate(wire_cast_var))
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [10015.]], self.evaluate(predictions))

  def test_dense_and_sparse_bias(self):
    wire_cast = tf.feature_column.categorical_column_with_hash_bucket(
        'wire_cast', 4)
    price = tf.feature_column.numeric_column('price')
    with tf.Graph().as_default():
      wire_tensor = tf.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor, 'price': [[1.], [5.]]}
      model = linear.LinearModel([wire_cast, price])
      predictions = model(features)
      price_var, wire_cast_var, bias = model.variables
      with _initialized_session() as sess:
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        sess.run(price_var.assign([[10.]]))
        self.assertAllClose([[1015.], [10065.]], self.evaluate(predictions))

  def test_dense_and_sparse_column(self):
    """When the column is both dense and sparse, uses sparse tensors."""

    class _DenseAndSparseColumn(BaseFeatureColumnForTests,
                                tf.__internal__.feature_column.DenseColumn,
                                fc.CategoricalColumn):

      @property
      def _is_v2_column(self):
        return True

      @property
      def name(self):
        return 'dense_and_sparse_column'

      @property
      def parse_example_spec(self):
        return {self.name: tf.io.VarLenFeature(self.dtype)}

      def transform_feature(self, transformation_cache, state_manager):
        return transformation_cache.get(self.name, state_manager)

      @property
      def variable_shape(self):
        raise ValueError('Should not use this method.')

      def get_dense_tensor(self, transformation_cache, state_manager):
        raise ValueError('Should not use this method.')

      @property
      def num_buckets(self):
        return 4

      def get_sparse_tensors(self, transformation_cache, state_manager):
        sp_tensor = tf.SparseTensor(
            indices=[[0, 0], [1, 0], [1, 1]],
            values=[2, 0, 3],
            dense_shape=[2, 2])
        return fc.CategoricalColumn.IdWeightPair(sp_tensor, None)

    dense_and_sparse_column = _DenseAndSparseColumn()
    with tf.Graph().as_default():
      sp_tensor = tf.SparseTensor(
          values=['omar', 'stringer', 'marlo'],
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {dense_and_sparse_column.name: sp_tensor}
      model = linear.LinearModel([dense_and_sparse_column])
      predictions = model(features)
      dense_and_sparse_column_var, bias = model.variables
      with _initialized_session() as sess:
        sess.run(
            dense_and_sparse_column_var.assign([[10.], [100.], [1000.],
                                                [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [10015.]], self.evaluate(predictions))

  def test_dense_multi_output(self):
    price = tf.feature_column.numeric_column('price')
    with tf.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      model = linear.LinearModel([price], units=3)
      predictions = model(features)
      price_var, bias = model.variables
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((3,)), self.evaluate(bias))
        self.assertAllClose(np.zeros((1, 3)), self.evaluate(price_var))
        sess.run(price_var.assign([[10., 100., 1000.]]))
        sess.run(bias.assign([5., 6., 7.]))
        self.assertAllClose([[15., 106., 1007.], [55., 506., 5007.]],
                            self.evaluate(predictions))

  def test_sparse_multi_output(self):
    wire_cast = tf.feature_column.categorical_column_with_hash_bucket(
        'wire_cast', 4)
    with tf.Graph().as_default():
      wire_tensor = tf.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor}
      model = linear.LinearModel([wire_cast], units=3)
      predictions = model(features)
      wire_cast_var, bias = model.variables
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((3,)), self.evaluate(bias))
        self.assertAllClose(np.zeros((4, 3)), self.evaluate(wire_cast_var))
        sess.run(
            wire_cast_var.assign([[10., 11., 12.], [100., 110., 120.],
                                  [1000., 1100., 1200.],
                                  [10000., 11000., 12000.]]))
        sess.run(bias.assign([5., 6., 7.]))
        self.assertAllClose([[1005., 1106., 1207.], [10015., 11017., 12019.]],
                            self.evaluate(predictions))

  def test_dense_multi_dimension(self):
    price = tf.feature_column.numeric_column('price', shape=2)
    with tf.Graph().as_default():
      features = {'price': [[1., 2.], [5., 6.]]}
      model = linear.LinearModel([price])
      predictions = model(features)
      price_var, _ = model.variables
      with _initialized_session() as sess:
        self.assertAllClose([[0.], [0.]], self.evaluate(price_var))
        sess.run(price_var.assign([[10.], [100.]]))
        self.assertAllClose([[210.], [650.]], self.evaluate(predictions))

  def test_sparse_multi_rank(self):
    wire_cast = tf.feature_column.categorical_column_with_hash_bucket(
        'wire_cast', 4)
    with tf.Graph().as_default():
      wire_tensor = tf.compat.v1.sparse_placeholder(tf.string)
      wire_value = tf.compat.v1.SparseTensorValue(
          values=['omar', 'stringer', 'marlo', 'omar'],  # hashed = [2, 0, 3, 2]
          indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1]],
          dense_shape=[2, 2, 2])
      features = {'wire_cast': wire_tensor}
      model = linear.LinearModel([wire_cast])
      predictions = model(features)
      wire_cast_var, _ = model.variables
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((4, 1)), self.evaluate(wire_cast_var))
        self.assertAllClose(
            np.zeros((2, 1)),
            predictions.eval(feed_dict={wire_tensor: wire_value}))
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        self.assertAllClose(
            [[1010.], [11000.]],
            predictions.eval(feed_dict={wire_tensor: wire_value}))

  def test_sparse_combiner(self):
    wire_cast = tf.feature_column.categorical_column_with_hash_bucket(
        'wire_cast', 4)
    with tf.Graph().as_default():
      wire_tensor = tf.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor}
      model = linear.LinearModel([wire_cast], sparse_combiner='mean')
      predictions = model(features)
      wire_cast_var, bias = model.variables
      with _initialized_session() as sess:
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [5010.]], self.evaluate(predictions))

  def test_sparse_combiner_sqrtn(self):
    wire_cast = tf.feature_column.categorical_column_with_hash_bucket(
        'wire_cast', 4)
    with tf.Graph().as_default():
      wire_tensor = tf.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor}
      model = linear.LinearModel([wire_cast], sparse_combiner='sqrtn')
      predictions = model(features)
      wire_cast_var, bias = model.variables
      with _initialized_session() as sess:
        self.evaluate(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        self.evaluate(bias.assign([5.]))
        self.assertAllClose([[1005.], [7083.139]], self.evaluate(predictions))

  def test_sparse_combiner_with_negative_weights(self):
    wire_cast = tf.feature_column.categorical_column_with_hash_bucket(
        'wire_cast', 4)
    wire_cast_weights = tf.feature_column.weighted_categorical_column(
        wire_cast, 'weights')

    with tf.Graph().as_default():
      wire_tensor = tf.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {
          'wire_cast': wire_tensor,
          'weights': tf.constant([[1., 1., -1.0]])
      }
      model = linear.LinearModel([wire_cast_weights], sparse_combiner='sum')
      predictions = model(features)
      wire_cast_var, bias = model.variables
      with _initialized_session() as sess:
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [-9985.]], self.evaluate(predictions))

  def test_dense_multi_dimension_multi_output(self):
    price = tf.feature_column.numeric_column('price', shape=2)
    with tf.Graph().as_default():
      features = {'price': [[1., 2.], [5., 6.]]}
      model = linear.LinearModel([price], units=3)
      predictions = model(features)
      price_var, bias = model.variables
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((3,)), self.evaluate(bias))
        self.assertAllClose(np.zeros((2, 3)), self.evaluate(price_var))
        sess.run(price_var.assign([[1., 2., 3.], [10., 100., 1000.]]))
        sess.run(bias.assign([2., 3., 4.]))
        self.assertAllClose([[23., 205., 2007.], [67., 613., 6019.]],
                            self.evaluate(predictions))

  def test_raises_if_shape_mismatch(self):
    price = tf.feature_column.numeric_column('price', shape=2)
    with tf.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      with self.assertRaisesRegexp(
          Exception,
          r'Cannot reshape a tensor with 2 elements to shape \[2,2\]'):
        model = linear.LinearModel([price])
        model(features)

  def test_dense_reshaping(self):
    price = tf.feature_column.numeric_column('price', shape=[1, 2])
    with tf.Graph().as_default():
      features = {'price': [[[1., 2.]], [[5., 6.]]]}
      model = linear.LinearModel([price])
      predictions = model(features)
      price_var, bias = model.variables
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        self.assertAllClose([[0.], [0.]], self.evaluate(price_var))
        self.assertAllClose([[0.], [0.]], self.evaluate(predictions))
        sess.run(price_var.assign([[10.], [100.]]))
        self.assertAllClose([[210.], [650.]], self.evaluate(predictions))

  def test_dense_multi_column(self):
    price1 = tf.feature_column.numeric_column('price1', shape=2)
    price2 = tf.feature_column.numeric_column('price2')
    with tf.Graph().as_default():
      features = {'price1': [[1., 2.], [5., 6.]], 'price2': [[3.], [4.]]}
      model = linear.LinearModel([price1, price2])
      predictions = model(features)
      price1_var, price2_var, bias = model.variables
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        self.assertAllClose([[0.], [0.]], self.evaluate(price1_var))
        self.assertAllClose([[0.]], self.evaluate(price2_var))
        self.assertAllClose([[0.], [0.]], self.evaluate(predictions))
        sess.run(price1_var.assign([[10.], [100.]]))
        sess.run(price2_var.assign([[1000.]]))
        sess.run(bias.assign([7.]))
        self.assertAllClose([[3217.], [4657.]], self.evaluate(predictions))

  def test_dense_trainable_default(self):
    price = tf.feature_column.numeric_column('price')
    with tf.Graph().as_default() as g:
      features = {'price': [[1.], [5.]]}
      model = linear.LinearModel([price])
      model(features)
      price_var, bias = model.variables
      trainable_vars = g.get_collection(
          tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
      self.assertIn(bias, trainable_vars)
      self.assertIn(price_var, trainable_vars)

  def test_sparse_trainable_default(self):
    wire_cast = tf.feature_column.categorical_column_with_hash_bucket(
        'wire_cast', 4)
    with tf.Graph().as_default() as g:
      wire_tensor = tf.SparseTensor(
          values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      features = {'wire_cast': wire_tensor}
      model = linear.LinearModel([wire_cast])
      model(features)
      trainable_vars = g.get_collection(
          tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
      wire_cast_var, bias = model.variables
      self.assertIn(bias, trainable_vars)
      self.assertIn(wire_cast_var, trainable_vars)

  def test_dense_trainable_false(self):
    price = tf.feature_column.numeric_column('price')
    with tf.Graph().as_default() as g:
      features = {'price': [[1.], [5.]]}
      model = linear.LinearModel([price], trainable=False)
      model(features)
      trainable_vars = g.get_collection(
          tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual([], trainable_vars)

  def test_sparse_trainable_false(self):
    wire_cast = tf.feature_column.categorical_column_with_hash_bucket(
        'wire_cast', 4)
    with tf.Graph().as_default() as g:
      wire_tensor = tf.SparseTensor(
          values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      features = {'wire_cast': wire_tensor}
      model = linear.LinearModel([wire_cast], trainable=False)
      model(features)
      trainable_vars = g.get_collection(
          tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual([], trainable_vars)

  def test_column_order(self):
    price_a = tf.feature_column.numeric_column('price_a')
    price_b = tf.feature_column.numeric_column('price_b')
    wire_cast = tf.feature_column.categorical_column_with_hash_bucket(
        'wire_cast', 4)
    with tf.Graph().as_default():
      features = {
          'price_a': [[1.]],
          'price_b': [[3.]],
          'wire_cast':
              tf.SparseTensor(
                  values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      }
      model = linear.LinearModel([price_a, wire_cast, price_b])
      model(features)

      my_vars = model.variables
      self.assertIn('price_a', my_vars[0].name)
      self.assertIn('price_b', my_vars[1].name)
      self.assertIn('wire_cast', my_vars[2].name)

    with tf.Graph().as_default():
      features = {
          'price_a': [[1.]],
          'price_b': [[3.]],
          'wire_cast':
              tf.SparseTensor(
                  values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      }
      model = linear.LinearModel([wire_cast, price_b, price_a])
      model(features)

      my_vars = model.variables
      self.assertIn('price_a', my_vars[0].name)
      self.assertIn('price_b', my_vars[1].name)
      self.assertIn('wire_cast', my_vars[2].name)

  def test_variable_names(self):
    price1 = tf.feature_column.numeric_column('price1')
    dense_feature = tf.feature_column.numeric_column('dense_feature')
    dense_feature_bucketized = tf.feature_column.bucketized_column(
        dense_feature, boundaries=[0.])
    some_sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)
    some_embedding_column = tf.feature_column.embedding_column(
        some_sparse_column, dimension=10)
    all_cols = [price1, dense_feature_bucketized, some_embedding_column]

    with tf.Graph().as_default():
      model = linear.LinearModel(all_cols)
      features = {
          'price1': [[3.], [4.]],
          'dense_feature': [[-1.], [4.]],
          'sparse_feature': [['a'], ['x']],
      }
      model(features)
      for var in model.variables:
        self.assertIsInstance(var, tf.compat.v1.Variable)
      variable_names = [var.name for var in model.variables]
      self.assertCountEqual([
          'linear_model/dense_feature_bucketized/weights:0',
          'linear_model/price1/weights:0',
          'linear_model/sparse_feature_embedding/embedding_weights:0',
          'linear_model/sparse_feature_embedding/weights:0',
          'linear_model/bias_weights:0',
      ], variable_names)

  def test_fit_and_predict(self):
    columns = [tf.feature_column.numeric_column('a')]

    model = linear.LinearModel(columns)
    model.compile(
        optimizer=tf.compat.v1.train.RMSPropOptimizer(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    x = {'a': np.random.random((10, 1))}
    y = np.random.randint(0, 2, size=(10, 1))
    model.fit(x, y, epochs=1, batch_size=5)
    model.fit(x, y, epochs=1, batch_size=5)
    model.evaluate(x, y, batch_size=5)
    model.predict(x, batch_size=5)

  def test_static_batch_size_mismatch(self):
    price1 = tf.feature_column.numeric_column('price1')
    price2 = tf.feature_column.numeric_column('price2')
    with tf.Graph().as_default():
      features = {
          'price1': [[1.], [5.], [7.]],  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
    with self.assertRaisesRegexp(
        ValueError,
        r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
      model = linear.LinearModel([price1, price2])
      model(features)

  def test_subset_of_static_batch_size_mismatch(self):
    price1 = tf.feature_column.numeric_column('price1')
    price2 = tf.feature_column.numeric_column('price2')
    price3 = tf.feature_column.numeric_column('price3')
    with tf.Graph().as_default():
      features = {
          'price1': tf.compat.v1.placeholder(dtype=tf.int64),  # batchsize = 3
          'price2': [[3.], [4.]],  # batchsize = 2
          'price3': [[3.], [4.], [5.]]  # batchsize = 3
      }
      with self.assertRaisesRegexp(
          ValueError,
          r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
        model = linear.LinearModel([price1, price2, price3])
        model(features)

  def test_runtime_batch_size_mismatch(self):
    price1 = tf.feature_column.numeric_column('price1')
    price2 = tf.feature_column.numeric_column('price2')
    with tf.Graph().as_default():
      features = {
          'price1': tf.compat.v1.placeholder(dtype=tf.int64),  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
      model = linear.LinearModel([price1, price2])
      predictions = model(features)
      with _initialized_session() as sess:
        with self.assertRaisesRegexp(tf.errors.OpError,
                                     'must have the same size and shape'):
          sess.run(
              predictions, feed_dict={features['price1']: [[1.], [5.], [7.]]})

  def test_runtime_batch_size_matches(self):
    price1 = tf.feature_column.numeric_column('price1')
    price2 = tf.feature_column.numeric_column('price2')
    with tf.Graph().as_default():
      features = {
          'price1': tf.compat.v1.placeholder(dtype=tf.int64),  # batchsize = 2
          'price2': tf.compat.v1.placeholder(dtype=tf.int64),  # batchsize = 2
      }
      model = linear.LinearModel([price1, price2])
      predictions = model(features)
      with _initialized_session() as sess:
        sess.run(
            predictions,
            feed_dict={
                features['price1']: [[1.], [5.]],
                features['price2']: [[1.], [5.]],
            })

  @test_util.run_deprecated_v1
  def test_with_1d_sparse_tensor(self):
    price = tf.feature_column.numeric_column('price')
    price_buckets = tf.feature_column.bucketized_column(
        price, boundaries=[
            0.,
            10.,
            100.,
        ])
    body_style = tf.feature_column.categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])

    # Provides 1-dim tensor and dense tensor.
    features = {
        'price':
            tf.constant([
                -1.,
                12.,
            ]),
        'body-style':
            tf.SparseTensor(
                indices=((0,), (1,)),
                values=('sedan', 'hardtop'),
                dense_shape=(2,)),
    }
    self.assertEqual(1, features['price'].shape.ndims)
    self.assertEqual(1, features['body-style'].dense_shape.get_shape()[0])

    model = linear.LinearModel([price_buckets, body_style])
    net = model(features)
    with _initialized_session() as sess:
      body_style_var, price_buckets_var, bias = model.variables

      sess.run(price_buckets_var.assign([[10.], [100.], [1000.], [10000.]]))
      sess.run(body_style_var.assign([[-10.], [-100.], [-1000.]]))
      sess.run(bias.assign([5.]))

      self.assertAllClose([[10 - 1000 + 5.], [1000 - 10 + 5.]],
                          self.evaluate(net))

  @test_util.run_deprecated_v1
  def test_with_1d_unknown_shape_sparse_tensor(self):
    price = tf.feature_column.numeric_column('price')
    price_buckets = tf.feature_column.bucketized_column(
        price, boundaries=[
            0.,
            10.,
            100.,
        ])
    body_style = tf.feature_column.categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])
    country = tf.feature_column.categorical_column_with_vocabulary_list(
        'country', vocabulary_list=['US', 'JP', 'CA'])

    # Provides 1-dim tensor and dense tensor.
    features = {
        'price': tf.compat.v1.placeholder(tf.float32),
        'body-style': tf.compat.v1.sparse_placeholder(tf.string),
        'country': tf.compat.v1.placeholder(tf.string),
    }
    self.assertIsNone(features['price'].shape.ndims)
    self.assertIsNone(features['body-style'].get_shape().ndims)

    price_data = np.array([-1., 12.])
    body_style_data = tf.compat.v1.SparseTensorValue(
        indices=((0,), (1,)), values=('sedan', 'hardtop'), dense_shape=(2,))
    country_data = np.array(['US', 'CA'])

    model = linear.LinearModel([price_buckets, body_style, country])
    net = model(features)
    body_style_var, _, price_buckets_var, bias = model.variables
    with _initialized_session() as sess:
      sess.run(price_buckets_var.assign([[10.], [100.], [1000.], [10000.]]))
      sess.run(body_style_var.assign([[-10.], [-100.], [-1000.]]))
      sess.run(bias.assign([5.]))

      self.assertAllClose([[10 - 1000 + 5.], [1000 - 10 + 5.]],
                          sess.run(
                              net,
                              feed_dict={
                                  features['price']: price_data,
                                  features['body-style']: body_style_data,
                                  features['country']: country_data
                              }))

  @test_util.run_deprecated_v1
  def test_with_rank_0_feature(self):
    price = tf.feature_column.numeric_column('price')
    features = {
        'price': tf.constant(0),
    }
    self.assertEqual(0, features['price'].shape.ndims)

    # Static rank 0 should fail
    with self.assertRaisesRegexp(ValueError, 'Feature .* cannot have rank 0'):
      model = linear.LinearModel([price])
      model(features)

    # Dynamic rank 0 should fail
    features = {
        'price': tf.compat.v1.placeholder(tf.float32),
    }
    model = linear.LinearModel([price])
    net = model(features)
    self.assertEqual(1, net.shape[1])
    with _initialized_session() as sess:
      with self.assertRaisesOpError('Feature .* cannot have rank 0'):
        sess.run(net, feed_dict={features['price']: np.array(1)})

  def test_multiple_linear_models(self):
    price = tf.feature_column.numeric_column('price')
    with tf.Graph().as_default():
      features1 = {'price': [[1.], [5.]]}
      features2 = {'price': [[2.], [10.]]}
      model1 = linear.LinearModel([price])
      model2 = linear.LinearModel([price])
      predictions1 = model1(features1)
      predictions2 = model2(features2)
      price_var1, bias1 = model1.variables
      price_var2, bias2 = model2.variables
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias1))
        sess.run(price_var1.assign([[10.]]))
        sess.run(bias1.assign([5.]))
        self.assertAllClose([[15.], [55.]], self.evaluate(predictions1))
        self.assertAllClose([0.], self.evaluate(bias2))
        sess.run(price_var2.assign([[10.]]))
        sess.run(bias2.assign([5.]))
        self.assertAllClose([[25.], [105.]], self.evaluate(predictions2))


class VocabularyFileCategoricalColumnTest(tf.test.TestCase):

  def setUp(self):
    super(VocabularyFileCategoricalColumnTest, self).setUp()

    # Contains strings, character names from 'The Wire': omar, stringer, marlo
    self._wire_vocabulary_file_name = os.path.join(
        flags.FLAGS['test_srcdir'].value,
        'org_tensorflow_estimator/tensorflow_estimator',
        'python/estimator/canned/testdata/wire_vocabulary.txt')

    # self._wire_vocabulary_file_name = test.test_src_dir_path(
    #     'python/estimator/canned/testdata/wire_vocabulary.txt')
    self._wire_vocabulary_size = 3

  # TODO(scottzhu): Reenable test once the issue for reading test file is fixed.
  @test_util.run_deprecated_v1
  def DISABLED_test_linear_model(self):
    wire_column = tf.compat.v1.feature_column.categorical_column_with_vocabulary_file(
        key='wire',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size,
        num_oov_buckets=1)
    self.assertEqual(4, wire_column.num_buckets)
    with tf.Graph().as_default():
      model = linear.LinearModel((wire_column,))
      predictions = model({
          wire_column.name:
              tf.compat.v1.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=('marlo', 'skywalker', 'omar'),
                  dense_shape=(2, 2))
      })
      wire_var, bias = model.variables

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,), (0.,)), self.evaluate(wire_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(wire_var.assign(((1.,), (2.,), (3.,), (4.,))))
      # 'marlo' -> 2: wire_var[2] = 3
      # 'skywalker' -> 3, 'omar' -> 0: wire_var[3] + wire_var[0] = 4+1 = 5
      self.assertAllClose(((3.,), (5.,)), self.evaluate(predictions))


class VocabularyListCategoricalColumnTest(tf.test.TestCase):

  @test_util.run_deprecated_v1
  def test_linear_model(self):
    wire_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'),
        num_oov_buckets=1)
    self.assertEqual(4, wire_column.num_buckets)
    with tf.Graph().as_default():
      model = linear.LinearModel((wire_column,))
      predictions = model({
          wire_column.name:
              tf.compat.v1.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=('marlo', 'skywalker', 'omar'),
                  dense_shape=(2, 2))
      })
      wire_var, bias = model.variables

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,), (0.,)), self.evaluate(wire_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(wire_var.assign(((1.,), (2.,), (3.,), (4.,))))
      # 'marlo' -> 2: wire_var[2] = 3
      # 'skywalker' -> 3, 'omar' -> 0: wire_var[3] + wire_var[0] = 4+1 = 5
      self.assertAllClose(((3.,), (5.,)), self.evaluate(predictions))


class IdentityCategoricalColumnTest(tf.test.TestCase):

  @test_util.run_deprecated_v1
  def test_linear_model(self):
    column = tf.feature_column.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    self.assertEqual(3, column.num_buckets)
    with tf.Graph().as_default():
      model = linear.LinearModel((column,))
      predictions = model({
          column.name:
              tf.compat.v1.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 2, 1),
                  dense_shape=(2, 2))
      })
      weight_var, bias = model.variables

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,)), self.evaluate(weight_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(weight_var.assign(((1.,), (2.,), (3.,))))
      # weight_var[0] = 1
      # weight_var[2] + weight_var[1] = 3+2 = 5
      self.assertAllClose(((1.,), (5.,)), self.evaluate(predictions))


class IndicatorColumnTest(tf.test.TestCase):

  @test_util.run_deprecated_v1
  def test_linear_model(self):
    animal = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_identity(
            'animal', num_buckets=4))
    with tf.Graph().as_default():
      features = {
          'animal':
              tf.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }

      model = linear.LinearModel([animal])
      predictions = model(features)
      weight_var, _ = model.variables

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      # All should be zero-initialized.
      self.assertAllClose([[0.], [0.], [0.], [0.]], self.evaluate(weight_var))
      self.assertAllClose([[0.]], self.evaluate(predictions))
      self.evaluate(weight_var.assign([[1.], [2.], [3.], [4.]]))
      self.assertAllClose([[2. + 3.]], self.evaluate(predictions))


class EmbeddingColumnTest(tf.test.TestCase, parameterized.TestCase):

  @test_util.run_deprecated_v1
  def test_linear_model(self):
    # Inputs.
    batch_size = 4
    vocabulary_size = 3
    sparse_input = tf.compat.v1.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        # example 2, ids []
        # example 3, ids [1]
        indices=((0, 0), (1, 0), (1, 4), (3, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(batch_size, 5))

    # Embedding variable.
    embedding_dimension = 2
    embedding_shape = (vocabulary_size, embedding_dimension)
    zeros_embedding_values = np.zeros(embedding_shape)

    def _initializer(shape, dtype, partition_info=None):
      self.assertAllEqual(embedding_shape, shape)
      self.assertEqual(tf.float32, dtype)
      self.assertIsNone(partition_info)
      return zeros_embedding_values

    # Build columns.
    categorical_column = tf.feature_column.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = tf.feature_column.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)

    with tf.Graph().as_default():
      model = linear.LinearModel((embedding_column,))
      predictions = model({categorical_column.name: sparse_input})
      expected_var_names = (
          'linear_model/bias_weights:0',
          'linear_model/aaa_embedding/weights:0',
          'linear_model/aaa_embedding/embedding_weights:0',
      )
      self.assertCountEqual(expected_var_names, [
          v.name for v in tf.compat.v1.get_collection(
              tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
      ])
      trainable_vars = {
          v.name: v for v in tf.compat.v1.get_collection(
              tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
      }
      self.assertCountEqual(expected_var_names, trainable_vars.keys())
      bias = trainable_vars['linear_model/bias_weights:0']
      embedding_weights = trainable_vars[
          'linear_model/aaa_embedding/embedding_weights:0']
      linear_weights = trainable_vars['linear_model/aaa_embedding/weights:0']

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      # Predictions with all zero weights.
      self.assertAllClose(np.zeros((1,)), self.evaluate(bias))
      self.assertAllClose(zeros_embedding_values,
                          self.evaluate(embedding_weights))
      self.assertAllClose(
          np.zeros((embedding_dimension, 1)), self.evaluate(linear_weights))
      self.assertAllClose(np.zeros((batch_size, 1)), self.evaluate(predictions))

      # Predictions with all non-zero weights.
      self.evaluate(
          embedding_weights.assign((
              (1., 2.),  # id 0
              (3., 5.),  # id 1
              (7., 11.)  # id 2
          )))
      self.evaluate(linear_weights.assign(((4.,), (6.,))))
      # example 0, ids [2], embedding[0] = [7, 11]
      # example 1, ids [0, 1], embedding[1] = mean([1, 2] + [3, 5]) = [2, 3.5]
      # example 2, ids [], embedding[2] = [0, 0]
      # example 3, ids [1], embedding[3] = [3, 5]
      # sum(embeddings * linear_weights)
      # = [4*7 + 6*11, 4*2 + 6*3.5, 4*0 + 6*0, 4*3 + 6*5] = [94, 29, 0, 42]
      self.assertAllClose(((94.,), (29.,), (0.,), (42.,)),
                          self.evaluate(predictions))


class SharedEmbeddingColumnTest(tf.test.TestCase, parameterized.TestCase):

  @test_util.run_deprecated_v1
  def test_linear_model(self):
    # Inputs.
    batch_size = 2
    vocabulary_size = 3
    # -1 values are ignored.
    input_a = np.array([
        [2, -1, -1],  # example 0, ids [2]
        [0, 1, -1]
    ])  # example 1, ids [0, 1]
    input_b = np.array([
        [0, -1, -1],  # example 0, ids [0]
        [-1, -1, -1]
    ])  # example 1, ids []

    # Embedding variable.
    embedding_dimension = 2
    embedding_shape = (vocabulary_size, embedding_dimension)
    zeros_embedding_values = np.zeros(embedding_shape)

    def _initializer(shape, dtype, partition_info=None):
      self.assertAllEqual(embedding_shape, shape)
      self.assertEqual(tf.float32, dtype)
      self.assertIsNone(partition_info)
      return zeros_embedding_values

    # Build columns.
    categorical_column_a = tf.feature_column.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    categorical_column_b = tf.feature_column.categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size)
    embedding_column_a, embedding_column_b = tf.feature_column.shared_embeddings(
        [categorical_column_a, categorical_column_b],
        dimension=embedding_dimension,
        initializer=_initializer)

    with tf.Graph().as_default():
      model = linear.LinearModel((embedding_column_a, embedding_column_b))
      predictions = model({
          categorical_column_a.name: input_a,
          categorical_column_b.name: input_b
      })

      # Linear weights do not follow the column name. But this is a rare use
      # case, and fixing it would add too much complexity to the code.
      expected_var_names = (
          'linear_model/bias_weights:0',
          'linear_model/aaa_shared_embedding/weights:0',
          'aaa_bbb_shared_embedding:0',
          'linear_model/bbb_shared_embedding/weights:0',
      )
      self.assertCountEqual(expected_var_names, [
          v.name for v in tf.compat.v1.get_collection(
              tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
      ])
      trainable_vars = {
          v.name: v for v in tf.compat.v1.get_collection(
              tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
      }
      self.assertCountEqual(expected_var_names, trainable_vars.keys())
      bias = trainable_vars['linear_model/bias_weights:0']
      embedding_weights = trainable_vars['aaa_bbb_shared_embedding:0']
      linear_weights_a = trainable_vars[
          'linear_model/aaa_shared_embedding/weights:0']
      linear_weights_b = trainable_vars[
          'linear_model/bbb_shared_embedding/weights:0']

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      # Predictions with all zero weights.
      self.assertAllClose(np.zeros((1,)), self.evaluate(bias))
      self.assertAllClose(zeros_embedding_values,
                          self.evaluate(embedding_weights))
      self.assertAllClose(
          np.zeros((embedding_dimension, 1)), self.evaluate(linear_weights_a))
      self.assertAllClose(
          np.zeros((embedding_dimension, 1)), self.evaluate(linear_weights_b))
      self.assertAllClose(np.zeros((batch_size, 1)), self.evaluate(predictions))

      # Predictions with all non-zero weights.
      self.evaluate(
          embedding_weights.assign((
              (1., 2.),  # id 0
              (3., 5.),  # id 1
              (7., 11.)  # id 2
          )))
      self.evaluate(linear_weights_a.assign(((4.,), (6.,))))
      # example 0, ids [2], embedding[0] = [7, 11]
      # example 1, ids [0, 1], embedding[1] = mean([1, 2] + [3, 5]) = [2, 3.5]
      # sum(embeddings * linear_weights)
      # = [4*7 + 6*11, 4*2 + 6*3.5] = [94, 29]
      self.evaluate(linear_weights_b.assign(((3.,), (5.,))))
      # example 0, ids [0], embedding[0] = [1, 2]
      # example 1, ids [], embedding[1] = 0, 0]
      # sum(embeddings * linear_weights)
      # = [3*1 + 5*2, 3*0 +5*0] = [13, 0]
      self.assertAllClose([[94. + 13.], [29.]], self.evaluate(predictions))


class WeightedCategoricalColumnTest(tf.test.TestCase):

  @test_util.run_deprecated_v1
  def test_linear_model(self):
    column = tf.feature_column.weighted_categorical_column(
        categorical_column=tf.feature_column.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with tf.Graph().as_default():
      model = linear.LinearModel((column,))
      predictions = model({
          'ids':
              tf.compat.v1.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 2, 1),
                  dense_shape=(2, 2)),
          'values':
              tf.compat.v1.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(.5, 1., .1),
                  dense_shape=(2, 2))
      })
      weight_var, bias = model.variables

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,)), self.evaluate(weight_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(weight_var.assign(((1.,), (2.,), (3.,))))
      # weight_var[0] * weights[0, 0] = 1 * .5 = .5
      # weight_var[2] * weights[1, 0] + weight_var[1] * weights[1, 1]
      # = 3*1 + 2*.1 = 3+.2 = 3.2
      self.assertAllClose(((.5,), (3.2,)), self.evaluate(predictions))

  def test_linear_model_mismatched_shape(self):
    column = tf.feature_column.weighted_categorical_column(
        categorical_column=tf.feature_column.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with tf.Graph().as_default():
      with self.assertRaisesRegexp(ValueError,
                                   r'Dimensions.*are not compatible'):
        model = linear.LinearModel((column,))
        model({
            'ids':
                tf.compat.v1.SparseTensorValue(
                    indices=((0, 0), (1, 0), (1, 1)),
                    values=(0, 2, 1),
                    dense_shape=(2, 2)),
            'values':
                tf.compat.v1.SparseTensorValue(
                    indices=((0, 0), (0, 1), (1, 0), (1, 1)),
                    values=(.5, 11., 1., .1),
                    dense_shape=(2, 2))
        })

  def test_linear_model_mismatched_dense_values(self):
    column = tf.feature_column.weighted_categorical_column(
        categorical_column=tf.feature_column.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with tf.Graph().as_default():
      model = linear.LinearModel((column,), sparse_combiner='mean')
      predictions = model({
          'ids':
              tf.compat.v1.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 2, 1),
                  dense_shape=(2, 2)),
          'values': ((.5,), (1.,))
      })
      # Disabling the constant folding optimizer here since it changes the
      # error message differently on CPU and GPU.
      config = tf.compat.v1.ConfigProto()
      config.graph_options.rewrite_options.constant_folding = (
          rewriter_config_pb2.RewriterConfig.OFF)
      with _initialized_session(config):
        with self.assertRaisesRegexp(tf.errors.OpError, 'Incompatible shapes'):
          self.evaluate(predictions)

  def test_linear_model_mismatched_dense_shape(self):
    column = tf.feature_column.weighted_categorical_column(
        categorical_column=tf.feature_column.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with tf.Graph().as_default():
      model = linear.LinearModel((column,))
      predictions = model({
          'ids':
              tf.compat.v1.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 2, 1),
                  dense_shape=(2, 2)),
          'values': ((.5,), (1.,), (.1,))
      })
      weight_var, bias = model.variables

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,)), self.evaluate(weight_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(weight_var.assign(((1.,), (2.,), (3.,))))
      # weight_var[0] * weights[0, 0] = 1 * .5 = .5
      # weight_var[2] * weights[1, 0] + weight_var[1] * weights[1, 1]
      # = 3*1 + 2*.1 = 3+.2 = 3.2
      self.assertAllClose(((.5,), (3.2,)), self.evaluate(predictions))


@test_util.run_all_in_graph_and_eager_modes
class LinearModelLayerSerializationTest(tf.test.TestCase,
                                        parameterized.TestCase):

  @parameterized.named_parameters(
      ('default', 1, 'sum', None, None),
      ('trainable', 6, 'mean', True, 'trainable'),
      ('not_trainable', 10, 'sum', False, 'frozen'))
  def test_get_config(self, units, sparse_combiner, trainable, name):
    cols = [
        tf.feature_column.numeric_column('a'),
        tf.feature_column.categorical_column_with_identity(
            key='b', num_buckets=3)
    ]
    layer = linear._LinearModelLayer(
        cols, units=units, sparse_combiner=sparse_combiner,
        trainable=trainable, name=name)
    config = layer.get_config()

    self.assertEqual(config['name'], layer.name)
    self.assertEqual(config['trainable'], trainable)
    self.assertEqual(config['units'], units)
    self.assertEqual(config['sparse_combiner'], sparse_combiner)
    self.assertLen(config['feature_columns'], 2)
    self.assertEqual(
        config['feature_columns'][0]['class_name'], 'NumericColumn')
    self.assertEqual(
        config['feature_columns'][1]['class_name'], 'IdentityCategoricalColumn')

  @parameterized.named_parameters(
      ('default', 1, 'sum', None, None),
      ('trainable', 6, 'mean', True, 'trainable'),
      ('not_trainable', 10, 'sum', False, 'frozen'))
  def test_from_config(self, units, sparse_combiner, trainable, name):
    cols = [
        tf.feature_column.numeric_column('a'),
        tf.feature_column.categorical_column_with_vocabulary_list(
            'b', vocabulary_list=('1', '2', '3')),
        tf.feature_column.categorical_column_with_hash_bucket(
            key='c', hash_bucket_size=3)
    ]
    orig_layer = linear._LinearModelLayer(
        cols, units=units, sparse_combiner=sparse_combiner,
        trainable=trainable, name=name)
    config = orig_layer.get_config()

    new_layer = linear._LinearModelLayer.from_config(config)

    self.assertEqual(new_layer.name, orig_layer.name)
    self.assertEqual(new_layer._units, units)
    self.assertEqual(new_layer._sparse_combiner, sparse_combiner)
    self.assertEqual(new_layer.trainable, trainable)
    self.assertLen(new_layer._feature_columns, 3)
    self.assertEqual(new_layer._feature_columns[0].name, 'a')
    self.assertEqual(
        new_layer._feature_columns[1].vocabulary_list, ('1', '2', '3'))
    self.assertEqual(new_layer._feature_columns[2].num_buckets, 3)


if __name__ == '__main__':
  tf.test.main()
