# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests boosted_trees estimators and model_fn."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import os
import tempfile

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.kernels.boosted_trees import boosted_trees_pb2
from tensorflow.python.feature_column import feature_column as feature_column_old
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import boosted_trees_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_boosted_trees_ops
from tensorflow.python.ops import resources
from tensorflow.python.platform import googletest
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator.canned import boosted_trees
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys

NUM_FEATURES = 3

BUCKET_BOUNDARIES = [-2., .5, 12.]  # Boundaries for all the features.
INPUT_FEATURES = np.array(
    [
        [12.5, 1.0, -2.001, -2.0001, -1.999],  # feature_0 quantized:[3,2,0,0,1]
        [2.0, -3.0, 0.5, 0.0, 0.4995],  # feature_1 quantized:[2,0,2,1,1]
        [3.0, 20.0, 50.0, -100.0, 102.75],  # feature_2 quantized:[2,3,3,0,3]
    ],
    dtype=np.float32)

CLASSIFICATION_LABELS = [[0.], [1.], [1.], [0.], [0.]]
MULTI_CLASS_LABELS = [[0], [1], [1], [0], [0], [2], [2]]
REGRESSION_LABELS = [[1.5], [0.3], [0.2], [2.], [5.]]
MULTI_DIM_REGRESSION_LABELS = [[1.5, -2.5], [0.3, -1.3], [0.2, -1.2],
                               [2., -3.0], [5., -6.0], [6.1, -7.10],
                               [7.01, -8.01]]
FEATURES_DICT = {'f_%d' % i: INPUT_FEATURES[i] for i in range(NUM_FEATURES)}

# EXAMPLE_ID is not exposed to Estimator yet, but supported at model_fn level.
EXAMPLE_IDS = [0, 1, 2, 3, 4]
EXAMPLE_ID_COLUMN = '__example_id__'


def _make_train_input_fn(is_classification, single_logit=True):
  """Makes train input_fn for classification/regression."""

  def _input_fn():
    example_ids = EXAMPLE_IDS
    features_dict = dict(FEATURES_DICT)  # copies the dict to add an entry.
    if single_logit:
      labels = CLASSIFICATION_LABELS if is_classification else REGRESSION_LABELS
    else:
      labels = MULTI_CLASS_LABELS if is_classification else MULTI_DIM_REGRESSION_LABELS

      def _add_additional_examples(feature, values):
        features_dict[feature] = np.concatenate(
            [features_dict[feature], values])

      _add_additional_examples('f_0', [1., 5.0])  # f0 quantized:[...,2,2]
      _add_additional_examples('f_1', [0.6, 11.1])  # f1 quantized:[...,2,2]
      _add_additional_examples('f_2', [11.99, 0.6])  # f2 quantized:[...,2,2]
      example_ids += [5, 6]
    features_dict[EXAMPLE_ID_COLUMN] = tf.constant(
        example_ids, dtype=tf.dtypes.int64)
    return features_dict, labels

  return _input_fn


def _make_train_input_fn_dataset(is_classification, batch=None, repeat=None):
  """Makes input_fn using Dataset."""

  def _input_fn():
    features_dict = dict(FEATURES_DICT)  # copies the dict to add an entry.
    features_dict[EXAMPLE_ID_COLUMN] = tf.constant(
        EXAMPLE_IDS, dtype=tf.dtypes.int64)
    labels = CLASSIFICATION_LABELS if is_classification else REGRESSION_LABELS
    if batch:
      ds = tf.compat.v1.data.Dataset.zip(
          (tf.compat.v1.data.Dataset.from_tensor_slices(features_dict),
           tf.compat.v1.data.Dataset.from_tensor_slices(labels))).batch(batch)
    else:
      ds = tf.compat.v1.data.Dataset.zip(
          (tf.compat.v1.data.Dataset.from_tensors(features_dict),
           tf.compat.v1.data.Dataset.from_tensors(labels)))
    # repeat indefinitely by default, or stop at the given step.
    ds = ds.repeat(repeat)
    return ds

  return _input_fn


class BoostedTreesEstimatorTest(tf.test.TestCase):

  def setUp(self):
    self._head = boosted_trees._create_regression_head(label_dimension=1)
    self._numeric_feature_columns = {
        tf.feature_column.numeric_column('f_%d' % i, dtype=tf.dtypes.float32)
        for i in range(NUM_FEATURES)
    }

    self._feature_columns = {
        tf.feature_column.bucketized_column(f, BUCKET_BOUNDARIES)
        for f in self._numeric_feature_columns
    }

  def _assert_checkpoint(self,
                         model_dir,
                         global_step,
                         finalized_trees,
                         attempted_layers,
                         bucket_boundaries=None):
    self._assert_checkpoint_and_return_model(model_dir, global_step,
                                             finalized_trees, attempted_layers,
                                             bucket_boundaries)

  def _assert_checkpoint_and_return_model(self,
                                          model_dir,
                                          global_step,
                                          finalized_trees,
                                          attempted_layers,
                                          bucket_boundaries=None):
    reader = tf.train.load_checkpoint(model_dir)
    self.assertEqual(global_step,
                     reader.get_tensor(tf.compat.v1.GraphKeys.GLOBAL_STEP))
    serialized = reader.get_tensor('boosted_trees:0_serialized')
    ensemble_proto = boosted_trees_pb2.TreeEnsemble()
    ensemble_proto.ParseFromString(serialized)

    self.assertEqual(
        finalized_trees,
        sum([1 for t in ensemble_proto.tree_metadata if t.is_finalized]))
    self.assertEqual(attempted_layers,
                     ensemble_proto.growing_metadata.num_layers_attempted)

    if bucket_boundaries:
      for i, bucket_boundary in enumerate(bucket_boundaries):
        self.assertAllClose(
            bucket_boundary,
            reader.get_tensor(
                'boosted_trees/QuantileAccumulator:0_bucket_boundaries_' +
                str(i)))

    return ensemble_proto

  @test_util.run_in_graph_and_eager_modes()
  def testV2(self):
    # Test into the future.
    with tf.compat.forward_compatibility_horizon(2019, 8, 9):
      control_flow_util.enable_control_flow_v2()
      tf.compat.v1.enable_resource_variables()

      categorical = tf.feature_column.categorical_column_with_vocabulary_list(
          key='f_0', vocabulary_list=('bad', 'good', 'ok'))
      indicator_col = tf.feature_column.indicator_column(categorical)
      bucketized_col = tf.feature_column.bucketized_column(
          tf.feature_column.numeric_column('f_1', dtype=tf.dtypes.float32),
          BUCKET_BOUNDARIES)
      numeric_col = tf.feature_column.numeric_column(
          'f_2', dtype=tf.dtypes.float32)
      int_numeric_col = tf.feature_column.numeric_column(
          'f_3', dtype=tf.dtypes.int64)

      labels = np.array([[0], [1], [1], [1], [1]], dtype=np.float32)
      input_fn = numpy_io.numpy_input_fn(
          x={
              'f_0': np.array(['bad', 'good', 'good', 'ok', 'bad']),
              'f_1': np.array([1, 1, 1, 1, 1]),
              'f_2': np.array([12.5, 1.0, -2.001, -2.0001, -1.999]),
              'f_3': np.array([100, 120, 110, 105, 156]),
          },
          y=labels,
          num_epochs=None,
          batch_size=5,
          shuffle=False)
      feature_columns = [
          numeric_col, bucketized_col, indicator_col, int_numeric_col
      ]

      est = boosted_trees.BoostedTreesClassifier(
          feature_columns=feature_columns,
          n_batches_per_layer=1,
          n_trees=1,
          max_depth=5,
          quantile_sketch_epsilon=0.33)

      # It will stop after 5 steps because of the max depth and num trees.
      num_steps = 100
      # Train for a few steps, and validate final checkpoint.
      est.train(input_fn, steps=num_steps)
      self._assert_checkpoint_and_return_model(
          est.model_dir,
          global_step=5,
          finalized_trees=1,
          attempted_layers=5,
          bucket_boundaries=[[-2.001, -1.999, 12.5], [100, 110, 156]])
      eval_res = est.evaluate(input_fn=input_fn, steps=1)
      self.assertAllClose(eval_res['accuracy'], 1.0)

  @test_util.run_in_graph_and_eager_modes()
  def testSwitchingConditionalAccumulatorForV1(self):
    # Test into the future.
    with tf.compat.forward_compatibility_horizon(2019, 8, 9):
      categorical = tf.feature_column.categorical_column_with_vocabulary_list(
          key='f_0', vocabulary_list=('bad', 'good', 'ok'))
      indicator_col = tf.feature_column.indicator_column(categorical)
      bucketized_col = tf.feature_column.bucketized_column(
          tf.feature_column.numeric_column('f_1', dtype=tf.dtypes.float32),
          BUCKET_BOUNDARIES)
      numeric_col = tf.feature_column.numeric_column(
          'f_2', dtype=tf.dtypes.float32)

      labels = np.array([[0], [1], [1], [1], [1]], dtype=np.float32)
      input_fn = numpy_io.numpy_input_fn(
          x={
              'f_0': np.array(['bad', 'good', 'good', 'ok', 'bad']),
              'f_1': np.array([1, 1, 1, 1, 1]),
              'f_2': np.array([12.5, 1.0, -2.001, -2.0001, -1.999]),
          },
          y=labels,
          num_epochs=None,
          batch_size=5,
          shuffle=False)
      feature_columns = [numeric_col, bucketized_col, indicator_col]

      est = boosted_trees.BoostedTreesClassifier(
          feature_columns=feature_columns,
          n_batches_per_layer=1,
          n_trees=1,
          max_depth=5,
          quantile_sketch_epsilon=0.33)

      # It will stop after 5 steps because of the max depth and num trees.
      num_steps = 100
      # Train for a few steps, and validate final checkpoint.
      est.train(input_fn, steps=num_steps)
      self._assert_checkpoint_and_return_model(
          est.model_dir,
          global_step=5,
          finalized_trees=1,
          attempted_layers=5,
          bucket_boundaries=[[-2.001, -1.999, 12.5]])
      eval_res = est.evaluate(input_fn=input_fn, steps=1)
      self.assertAllClose(eval_res['accuracy'], 1.0)

  @test_util.run_in_graph_and_eager_modes()
  def testSavedModel(self):
    self._feature_columns = {
        tf.feature_column.numeric_column('f_%d' % i, dtype=tf.dtypes.float32)
        for i in range(NUM_FEATURES)
    }
    input_fn = _make_train_input_fn_dataset(is_classification=True)
    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)
    est.train(input_fn, steps=10)
    tmpdir = tempfile.mkdtemp()
    export_dir = os.path.join(tmpdir, 'saved_model')
    input_receiver_fn = (
        export_lib.build_supervised_input_receiver_fn_from_input_fn(input_fn))
    export_dir = est.export_saved_model(
        export_dir, input_receiver_fn, as_text=True)

    # Restore, to validate that the export was well-formed.
    tag_set = export_lib.EXPORT_TAG_MAP[ModeKeys.PREDICT]
    with tf.Graph().as_default() as graph:
      with tf.compat.v1.Session(graph=graph) as sess:
        tf.compat.v1.saved_model.load(sess, tag_set, export_dir)
        graph_ops = [x.name for x in graph.get_operations()]
        # Assert Tree Ensemble resource op is in graph def.
        self.assertTrue('boosted_trees' in graph_ops)
        # Assert Quantile Accumulator resource op is in graph def.
        self.assertTrue('boosted_trees/QuantileAccumulator' in graph_ops)
        saveable_objects = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.SAVEABLE_OBJECTS)
        saveable_objects = sorted(saveable_objects, key=lambda obj: obj.name)
        # Assert QuantileAccumulator is in saveable object collection.
        self.assertEqual('boosted_trees/QuantileAccumulator:0',
                         saveable_objects[0].name)
        # Assert TreeEnsemble is in saveable object collection.
        self.assertEqual('boosted_trees:0', saveable_objects[1].name)

    # Clean up.
    tf.compat.v1.gfile.DeleteRecursively(tmpdir)

  @test_util.run_in_graph_and_eager_modes()
  def testFirstCheckpointWorksFine(self):
    """Tests that eval/pred doesn't crash with the very first checkpoint.

    The step-0 checkpoint will have only an empty ensemble, and a separate eval
    job might read from it and crash.
    This test ensures that prediction/evaluation works fine with it.
    """
    input_fn = _make_train_input_fn(is_classification=True)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)

    class BailOutWithoutTraining(tf.compat.v1.train.SessionRunHook):

      def before_run(self, run_context):
        raise StopIteration('to bail out.')

    est.train(
        input_fn,
        steps=100,  # must stop at 0 anyway.
        hooks=[BailOutWithoutTraining()])
    self._assert_checkpoint(
        est.model_dir, global_step=0, finalized_trees=0, attempted_layers=0)
    # Empty ensemble returns 0 logits, so that all output labels are 0.
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['accuracy'], 0.6)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose([[0], [0], [0], [0], [0]],
                        [pred['class_ids'] for pred in predictions])

  @test_util.run_in_graph_and_eager_modes()
  def testInvalidInputParameters(self):
    make_est = functools.partial(
        boosted_trees.BoostedTreesRegressor,
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=3,
        learning_rate=0.01,
        quantile_sketch_epsilon=0.01,
        l1_regularization=0.1,
        l2_regularization=0.21,
        tree_complexity=0.1,
        min_node_weight=0.0)
    # First test using all valid parameters.
    make_est()
    # Parameters must be positive.
    with self.assertRaisesRegexp(ValueError, '> 0'):
      make_est(n_trees=0)
    with self.assertRaisesRegexp(ValueError, '> 0'):
      make_est(max_depth=0)
    with self.assertRaisesRegexp(ValueError, '> 0'):
      make_est(learning_rate=0)
    with self.assertRaisesRegexp(ValueError, '> 0'):
      make_est(quantile_sketch_epsilon=0)
    # Parameters must be non-negative.
    with self.assertRaisesRegexp(ValueError, '>= 0'):
      make_est(l1_regularization=-0.1)
    with self.assertRaisesRegexp(ValueError, '>= 0'):
      make_est(l2_regularization=-0.1)
    with self.assertRaisesRegexp(ValueError, '>= 0'):
      make_est(tree_complexity=-0.1)
    with self.assertRaisesRegexp(ValueError, '>= 0'):
      make_est(min_node_weight=-0.1)

  @test_util.run_in_graph_and_eager_modes()
  def testTrainAndEvaluateBinaryClassifier(self):
    input_fn = _make_train_input_fn(is_classification=True)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['accuracy'], 1.0)

  @test_util.run_in_graph_and_eager_modes()
  def testTrainAndEvaluateMultiClassClassifier(self):
    input_fn = _make_train_input_fn(is_classification=True, single_logit=False)
    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_classes=3,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['accuracy'], 1.0)

  @test_util.run_in_graph_and_eager_modes()
  def testTrainAndEvaluateBinaryClassifierWithOnlyFloatColumn(self):
    self._feature_columns = {
        tf.feature_column.numeric_column('f_%d' % i, dtype=tf.dtypes.float32)
        for i in range(NUM_FEATURES)
    }

    input_fn = _make_train_input_fn(is_classification=True)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5,
        quantile_sketch_epsilon=0.33)

    # Prediction will provide all zeros because buckets are not ready.
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose([[0], [0], [0], [0], [0]],
                        [pred['class_ids'] for pred in predictions])

    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    # It will stop after 5 steps because of the max depth and num trees.
    est.train(input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir,
        global_step=5,
        finalized_trees=1,
        attempted_layers=5,
        bucket_boundaries=[[-2.001, -1.999, 12.5], [-3., 0.4995, 2.],
                           [-100., 20., 102.75]])
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['accuracy'], 1.0)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose([[0], [1], [1], [0], [0]],
                        [pred['class_ids'] for pred in predictions])

  @test_util.run_in_graph_and_eager_modes()
  def testTrainAndEvaluateBinaryClassifierWithEmptyShape(self):
    self._feature_columns = {
        tf.feature_column.numeric_column(
            'f_%d' % i, shape=(), dtype=tf.dtypes.float32)
        for i in range(NUM_FEATURES)
    }

    input_fn = _make_train_input_fn(is_classification=True)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5,
        quantile_sketch_epsilon=0.33)

    # Prediction will provide all zeros because buckets are not ready.
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose([[0], [0], [0], [0], [0]],
                        [pred['class_ids'] for pred in predictions])

    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    # It will stop after 5 steps because of the max depth and num trees.
    est.train(input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir,
        global_step=5,
        finalized_trees=1,
        attempted_layers=5,
        bucket_boundaries=[[-2.001, -1.999, 12.5], [-3., 0.4995, 2.],
                           [-100., 20., 102.75]])
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['accuracy'], 1.0)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose([[0], [1], [1], [0], [0]],
                        [pred['class_ids'] for pred in predictions])

  @test_util.run_in_graph_and_eager_modes()
  def testTrainAndEvaluateBinaryClassifierWithWeightColumn(self):
    feature_and_weight_dict_weight_1 = {
        'weight': np.array([1., 10., 1., 1., 1.], dtype=np.float32)
    }
    feature_and_weight_dict_weight_2 = {
        'weight': np.array([10., 1., 2., 1., 1.], dtype=np.float32)
    }
    self._feature_columns = {
        tf.feature_column.numeric_column('f_%d' % i, dtype=tf.dtypes.float32)
        for i in range(NUM_FEATURES)
    }
    weights = tf.feature_column.numeric_column(
        'weight', dtype=tf.dtypes.float32)
    feature_and_weight_dict_weight_1.update(FEATURES_DICT.copy())
    feature_and_weight_dict_weight_2.update(FEATURES_DICT.copy())
    input_fn_1 = numpy_io.numpy_input_fn(
        x=feature_and_weight_dict_weight_1,
        y=np.array(CLASSIFICATION_LABELS),
        num_epochs=None,
        batch_size=5,
        shuffle=False)
    input_fn_2 = numpy_io.numpy_input_fn(
        x=feature_and_weight_dict_weight_2,
        y=np.array(CLASSIFICATION_LABELS),
        num_epochs=None,
        batch_size=5,
        shuffle=False)
    est_1 = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        weight_column=weights,
        n_trees=1,
        max_depth=5)
    est_2 = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        weight_column=weights,
        n_trees=1,
        max_depth=5)
    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est_1.train(input_fn_1, steps=num_steps)
    self._assert_checkpoint(
        est_1.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res_1 = est_1.evaluate(input_fn=input_fn_1, steps=1)
    self.assertAllClose(eval_res_1['accuracy'], 1.0)
    est_2.train(input_fn_2, steps=num_steps)
    self._assert_checkpoint(
        est_2.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res_2 = est_2.evaluate(input_fn=input_fn_2, steps=1)
    self.assertAllClose(eval_res_2['accuracy'], 1.0)
    self.assertTrue(est_1.model_dir != est_2.model_dir)
    self.assertTrue(eval_res_1 != eval_res_2)

  @test_util.run_in_graph_and_eager_modes()
  def testTrainAndEvaluateBinaryClassifierWithWeightColumnAsString(self):
    feature_and_weight_dict_weight_1 = {
        'weight': np.array([1., 10., 1., 1., 1.], dtype=np.float32)
    }
    feature_and_weight_dict_weight_2 = {
        'weight': np.array([10., 1., 2., 1., 1.], dtype=np.float32)
    }
    self._feature_columns = {
        tf.feature_column.numeric_column('f_%d' % i, dtype=tf.dtypes.float32)
        for i in range(NUM_FEATURES)
    }
    feature_and_weight_dict_weight_1.update(FEATURES_DICT.copy())
    feature_and_weight_dict_weight_2.update(FEATURES_DICT.copy())
    input_fn_1 = numpy_io.numpy_input_fn(
        x=feature_and_weight_dict_weight_1,
        y=np.array(CLASSIFICATION_LABELS),
        num_epochs=None,
        batch_size=5,
        shuffle=False)
    input_fn_2 = numpy_io.numpy_input_fn(
        x=feature_and_weight_dict_weight_2,
        y=np.array(CLASSIFICATION_LABELS),
        num_epochs=None,
        batch_size=5,
        shuffle=False)
    est_1 = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        weight_column='weight',
        n_trees=1,
        max_depth=5)
    est_2 = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        weight_column='weight',
        n_trees=1,
        max_depth=5)
    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est_1.train(input_fn_1, steps=num_steps)
    self._assert_checkpoint(
        est_1.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res_1 = est_1.evaluate(input_fn=input_fn_1, steps=1)
    self.assertAllClose(eval_res_1['accuracy'], 1.0)
    est_2.train(input_fn_2, steps=num_steps)
    self._assert_checkpoint(
        est_2.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res_2 = est_2.evaluate(input_fn=input_fn_2, steps=1)
    self.assertAllClose(eval_res_2['accuracy'], 1.0)
    self.assertTrue(est_1.model_dir != est_2.model_dir)
    self.assertTrue(eval_res_1 != eval_res_2)

  @test_util.run_in_graph_and_eager_modes()
  def testTrainAndEvaluateBinaryClassifierWithMultiDimFloatColumn(self):
    self._feature_columns = {
        tf.feature_column.numeric_column(
            'f', shape=(3,), dtype=tf.dtypes.float32)
    }
    input_fn = numpy_io.numpy_input_fn(
        x={'f': np.transpose(np.copy(INPUT_FEATURES))},
        y=np.array(CLASSIFICATION_LABELS),
        num_epochs=None,
        batch_size=5,
        shuffle=False)
    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)
    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['accuracy'], 1.0)

  @test_util.run_in_graph_and_eager_modes()
  def testTrainAndEvaluateBinaryClassifierWithMixedColumns(self):
    bucketized_col = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column('f_0', dtype=tf.dtypes.float32),
        BUCKET_BOUNDARIES)
    categorical = tf.feature_column.categorical_column_with_vocabulary_list(
        key='f_1', vocabulary_list=('bad', 'good', 'ok'))
    indicator_col = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            key='f_2', vocabulary_list=('bad', 'good', 'ok')))
    numeric_col = tf.feature_column.numeric_column(
        'f_3', dtype=tf.dtypes.float32)

    labels = np.array([[0], [1], [1], [1], [1]], dtype=np.float32)
    input_fn = numpy_io.numpy_input_fn(
        x={
            'f_0': np.array([1, 1, 1, 1, 1]),
            'f_1': np.array(['bad', 'good', 'good', 'ok', 'bad']),
            'f_2': np.array(['bad', 'good', 'good', 'ok', 'bad']),
            'f_3': np.array([12.5, 1.0, -2.001, -2.0001, -1.999]),
        },
        y=labels,
        num_epochs=None,
        batch_size=5,
        shuffle=False)
    feature_columns = [numeric_col, bucketized_col, categorical, indicator_col]

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5,
        quantile_sketch_epsilon=0.33)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(input_fn, steps=num_steps)
    self._assert_checkpoint_and_return_model(
        est.model_dir,
        global_step=5,
        finalized_trees=1,
        attempted_layers=5,
        bucket_boundaries=[[-2.001, -1.999, 12.5]])
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['accuracy'], 1.0)

  @test_util.run_in_graph_and_eager_modes()
  def testTrainTwiceAndEvaluateBinaryClassifier(self):
    input_fn = _make_train_input_fn(is_classification=True)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=5,
        max_depth=10)

    num_steps = 2
    # Train for a few steps, and validate final checkpoint.
    est.train(input_fn, steps=num_steps)
    est.train(input_fn, steps=num_steps)

    self._assert_checkpoint(
        est.model_dir,
        global_step=num_steps * 2,
        finalized_trees=0,
        attempted_layers=4)
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['accuracy'], 1.0)

  @test_util.run_in_graph_and_eager_modes()
  def testInferBinaryClassifier(self):
    train_input_fn = _make_train_input_fn(is_classification=True)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(train_input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose([[0], [1], [1], [0], [0]],
                        [pred['class_ids'] for pred in predictions])

  @test_util.run_in_graph_and_eager_modes()
  def testTrainClassifierWithRankOneLabel(self):
    """Tests that label with rank-1 tensor is also accepted by classifier."""

    def _input_fn_with_rank_one_label():
      return FEATURES_DICT, [0., 1., 1., 0., 0.]

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(_input_fn_with_rank_one_label, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=_input_fn_with_rank_one_label, steps=1)
    self.assertAllClose(eval_res['accuracy'], 1.0)

  @test_util.run_in_graph_and_eager_modes()
  def testTrainClassifierWithLabelVocabulary(self):
    apple, banana = 'apple', 'banana'

    def _input_fn_with_label_vocab():
      return FEATURES_DICT, [[apple], [banana], [banana], [apple], [apple]]

    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5,
        label_vocabulary=[apple, banana])
    est.train(input_fn=_input_fn_with_label_vocab, steps=5)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=_input_fn_with_label_vocab, steps=1)
    self.assertAllClose(eval_res['accuracy'], 1.0)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose([[0], [1], [1], [0], [0]],
                        [pred['class_ids'] for pred in predictions])

  @test_util.run_in_graph_and_eager_modes()
  def testTrainClassifierWithIntegerLabel(self):

    def _input_fn_with_integer_label():
      return (FEATURES_DICT,
              tf.constant([[0], [1], [1], [0], [0]], tf.dtypes.int32))

    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)
    est.train(input_fn=_input_fn_with_integer_label, steps=5)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=_input_fn_with_integer_label, steps=1)
    self.assertAllClose(eval_res['accuracy'], 1.0)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose([[0], [1], [1], [0], [0]],
                        [pred['class_ids'] for pred in predictions])

  @test_util.run_in_graph_and_eager_modes()
  def testTrainClassifierWithDataset(self):
    train_input_fn = _make_train_input_fn_dataset(is_classification=True)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)
    est.train(train_input_fn, steps=100)  # will stop after 5 steps anyway.
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=train_input_fn, steps=1)
    self.assertAllClose(eval_res['accuracy'], 1.0)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose([[0], [1], [1], [0], [0]],
                        [pred['class_ids'] for pred in predictions])

  @test_util.run_in_graph_and_eager_modes()
  def testTrainAndEvaluateRegressor(self):
    input_fn = _make_train_input_fn(is_classification=False)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=2,
        max_depth=5)

    # It will stop after 10 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=10, finalized_trees=2, attempted_layers=10)
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['average_loss'], 1.008551)

  @test_util.run_in_graph_and_eager_modes()
  def testTrainAndEvaluateMultiDimensionalRegressor(self):
    input_fn = _make_train_input_fn(is_classification=False, single_logit=False)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=2,
        max_depth=5,
        label_dimension=2)

    # It will stop after 10 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=10, finalized_trees=2, attempted_layers=10)
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['average_loss'], 3.14401078224)

  @test_util.run_in_graph_and_eager_modes()
  def testInferRegressor(self):
    train_input_fn = _make_train_input_fn(is_classification=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(train_input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose(
        [[0.571619], [0.262821], [0.124549], [0.956801], [1.769801]],
        [pred['predictions'] for pred in predictions])

  @test_util.run_in_graph_and_eager_modes()
  def testTrainRegressorWithRankOneLabel(self):
    """Tests that label with rank-1 tensor is also accepted by regressor."""

    def _input_fn_with_rank_one_label():
      return FEATURES_DICT, [1.5, 0.3, 0.2, 2., 5.]

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(_input_fn_with_rank_one_label, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=_input_fn_with_rank_one_label, steps=1)
    self.assertAllClose(eval_res['average_loss'], 2.478283)

  @test_util.run_in_graph_and_eager_modes()
  def testTrainRegressorWithDataset(self):
    train_input_fn = _make_train_input_fn_dataset(is_classification=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)
    est.train(train_input_fn, steps=100)  # will stop after 5 steps anyway.
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=train_input_fn, steps=1)
    self.assertAllClose(eval_res['average_loss'], 2.478283)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose(
        [[0.571619], [0.262821], [0.124549], [0.956801], [1.769801]],
        [pred['predictions'] for pred in predictions])

  @test_util.run_in_graph_and_eager_modes()
  def testTrainRegressorWithDatasetBatch(self):
    # The batch_size as the entire data size should yield the same result as
    # dataset without batching.
    train_input_fn = _make_train_input_fn_dataset(
        is_classification=False, batch=5)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)
    est.train(train_input_fn, steps=100)  # will stop after 5 steps anyway.
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=train_input_fn, steps=1)
    self.assertAllClose(eval_res['average_loss'], 2.478283)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose(
        [[0.571619], [0.262821], [0.124549], [0.956801], [1.769801]],
        [pred['predictions'] for pred in predictions])

  @test_util.run_in_graph_and_eager_modes()
  def testTrainRegressorWithDatasetLargerBatch(self):
    # The batch_size as the multiple of the entire data size should still yield
    # the same result.
    train_input_fn = _make_train_input_fn_dataset(
        is_classification=False, batch=15)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)
    est.train(train_input_fn, steps=100)  # will stop after 5 steps anyway.
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    eval_res = est.evaluate(input_fn=train_input_fn, steps=1)
    self.assertAllClose(eval_res['average_loss'], 2.478283)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose(
        [[0.571619], [0.262821], [0.124549], [0.956801], [1.769801]],
        [pred['predictions'] for pred in predictions])

  @test_util.run_in_graph_and_eager_modes()
  def testTrainRegressorWithDatasetSmallerBatch(self):
    # Even when using small batches, if (n_batches_per_layer * batch_size) makes
    # the same entire data size, the result should be the same.
    train_input_fn = _make_train_input_fn_dataset(
        is_classification=False, batch=1)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=5,
        n_trees=1,
        max_depth=5)
    # Train stops after (n_batches_per_layer * n_trees * max_depth) steps.
    est.train(train_input_fn, steps=100)
    self._assert_checkpoint(
        est.model_dir, global_step=25, finalized_trees=1, attempted_layers=5)
    # 5 batches = one epoch.
    eval_res = est.evaluate(input_fn=train_input_fn, steps=5)
    self.assertAllClose(eval_res['average_loss'], 2.478283)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose(
        [[0.571619], [0.262821], [0.124549], [0.956801], [1.769801]],
        [pred['predictions'] for pred in predictions])

  @test_util.run_in_graph_and_eager_modes()
  def testTrainRegressorWithDatasetWhenInputIsOverEarlier(self):
    train_input_fn = _make_train_input_fn_dataset(
        is_classification=False, repeat=3)  # to stop input after 3 steps.
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)
    # Note that training will stop when input exhausts.
    # This might not be a typical pattern, but dataset.repeat(3) causes
    # the input stream to cease after 3 steps.
    est.train(train_input_fn, steps=100)
    self._assert_checkpoint(
        est.model_dir, global_step=3, finalized_trees=0, attempted_layers=3)
    eval_res = est.evaluate(input_fn=train_input_fn, steps=1)
    self.assertAllClose(eval_res['average_loss'], 3.777295)
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose(
        [[0.353850], [0.254100], [0.106850], [0.712100], [1.012100]],
        [pred['predictions'] for pred in predictions])

  @test_util.run_in_graph_and_eager_modes()
  def testTrainEvaluateAndPredictWithIndicatorColumn(self):
    categorical = tf.feature_column.categorical_column_with_vocabulary_list(
        key='categorical', vocabulary_list=('bad', 'good', 'ok'))
    feature_indicator = tf.feature_column.indicator_column(categorical)
    bucketized_col = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column(
            'an_uninformative_feature', dtype=tf.dtypes.float32),
        BUCKET_BOUNDARIES)

    labels = np.array([[0.], [5.7], [5.7], [0.], [0.]], dtype=np.float32)
    # Our categorical feature defines the labels perfectly
    input_fn = numpy_io.numpy_input_fn(
        x={
            'an_uninformative_feature': np.array([1, 1, 1, 1, 1]),
            'categorical': np.array(['bad', 'good', 'good', 'ok', 'bad']),
        },
        y=labels,
        batch_size=5,
        shuffle=False)

    # Train depth 1 tree.
    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=[bucketized_col, feature_indicator],
        n_batches_per_layer=1,
        n_trees=1,
        learning_rate=1.0,
        max_depth=1)

    num_steps = 1
    est.train(input_fn, steps=num_steps)
    ensemble = self._assert_checkpoint_and_return_model(
        est.model_dir, global_step=1, finalized_trees=1, attempted_layers=1)

    # We learnt perfectly.
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['loss'], 0)

    predictions = list(est.predict(input_fn))
    self.assertAllClose(labels, [pred['predictions'] for pred in predictions])

    self.assertEqual(3, len(ensemble.trees[0].nodes))

    # Check that the split happened on 'good' value, which will be encoded as
    # feature with index 2 (0-numeric, 1 - 'bad')
    self.assertEqual(2, ensemble.trees[0].nodes[0].bucketized_split.feature_id)
    self.assertEqual(0, ensemble.trees[0].nodes[0].bucketized_split.threshold)

  @test_util.run_in_graph_and_eager_modes()
  def testTrainEvaluateAndPredictWithOnlyIndicatorColumn(self):
    categorical = tf.feature_column.categorical_column_with_vocabulary_list(
        key='categorical', vocabulary_list=('bad', 'good', 'ok'))
    feature_indicator = tf.feature_column.indicator_column(categorical)

    labels = np.array([[0.], [5.7], [5.7], [0.], [0.]], dtype=np.float32)
    # Our categorical feature defines the labels perfectly
    input_fn = numpy_io.numpy_input_fn(
        x={
            'categorical': np.array(['bad', 'good', 'good', 'ok', 'bad']),
        },
        y=labels,
        batch_size=5,
        shuffle=False)

    # Train depth 1 tree.
    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=[feature_indicator],
        n_batches_per_layer=1,
        n_trees=1,
        learning_rate=1.0,
        max_depth=1)

    num_steps = 1
    est.train(input_fn, steps=num_steps)
    ensemble = self._assert_checkpoint_and_return_model(
        est.model_dir, global_step=1, finalized_trees=1, attempted_layers=1)

    # We learnt perfectly.
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['loss'], 0)

    predictions = list(est.predict(input_fn))
    self.assertAllClose(labels, [pred['predictions'] for pred in predictions])

    self.assertEqual(3, len(ensemble.trees[0].nodes))

    # Check that the split happened on 'good' value, which will be encoded as
    # feature with index 1 (0 - 'bad', 2 - 'ok')
    self.assertEqual(1, ensemble.trees[0].nodes[0].bucketized_split.feature_id)
    self.assertEqual(0, ensemble.trees[0].nodes[0].bucketized_split.threshold)

  @test_util.run_in_graph_and_eager_modes()
  def testTrainEvaluateAndPredictWithCategoricalColumn(self):
    categorical_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key='categorical', vocabulary_list=('bad', 'good', 'ok'))
    bucketized_col = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column(
            'an_uninformative_feature', dtype=tf.dtypes.float32),
        BUCKET_BOUNDARIES)

    labels = np.array([[0.], [5.7], [5.7], [0.], [0.]], dtype=np.float32)
    # Our categorical feature defines the labels perfectly
    input_fn = numpy_io.numpy_input_fn(
        x={
            'an_uninformative_feature': np.array([1, 1, 1, 1, 1]),
            'categorical': np.array(['bad', 'good', 'good', 'ok', 'bad']),
        },
        y=labels,
        batch_size=5,
        shuffle=False)

    # Train depth 1 tree.
    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=[bucketized_col, categorical_col],
        n_batches_per_layer=1,
        n_trees=1,
        learning_rate=1.0,
        max_depth=1)

    num_steps = 1
    est.train(input_fn, steps=num_steps)
    ensemble = self._assert_checkpoint_and_return_model(
        est.model_dir, global_step=1, finalized_trees=1, attempted_layers=1)

    # We learnt perfectly.
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['loss'], 0)

    predictions = list(est.predict(input_fn))
    self.assertAllClose(labels, [pred['predictions'] for pred in predictions])

    self.assertEqual(3, len(ensemble.trees[0].nodes))

    # Check that the split happened on 'good' value, which is the index 1 feature
    # (0-numeric, 1-categorical), and the index 1 feature value of categorical.
    self.assertEqual(1, ensemble.trees[0].nodes[0].categorical_split.feature_id)
    self.assertEqual(1, ensemble.trees[0].nodes[0].categorical_split.value)

  @test_util.run_in_graph_and_eager_modes()
  def testTrainEvaluateAndPredictWithOnlyCategoricalColumn(self):
    categorical = tf.feature_column.categorical_column_with_vocabulary_list(
        key='categorical', vocabulary_list=('bad', 'good', 'ok'))

    labels = np.array([[0.], [5.7], [5.7], [0.], [0.]], dtype=np.float32)
    # Our categorical feature defines the labels perfectly
    input_fn = numpy_io.numpy_input_fn(
        x={
            'categorical': np.array(['bad', 'good', 'good', 'ok', 'bad']),
        },
        y=labels,
        batch_size=5,
        shuffle=False)

    # Train depth 1 tree.
    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=[categorical],
        n_batches_per_layer=1,
        n_trees=1,
        learning_rate=1.0,
        max_depth=1)

    num_steps = 1
    est.train(input_fn, steps=num_steps)
    ensemble = self._assert_checkpoint_and_return_model(
        est.model_dir, global_step=1, finalized_trees=1, attempted_layers=1)

    # We learnt perfectly.
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['loss'], 0)

    predictions = list(est.predict(input_fn))
    self.assertAllClose(labels, [pred['predictions'] for pred in predictions])

    self.assertEqual(3, len(ensemble.trees[0].nodes))

    # Check that the split happened on 'good' value, which will be encoded as
    # feature with index 1 (0 - 'bad', 2 - 'ok')
    self.assertEqual(1, ensemble.trees[0].nodes[0].categorical_split.value)

  @test_util.run_in_graph_and_eager_modes()
  def testFeatureImportancesWithTrainedEnsemble(self):
    input_fn = _make_train_input_fn(is_classification=True)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=2,
        max_depth=5)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(input_fn, steps=num_steps)
    importances = est.experimental_feature_importances(normalize=False)
    expected_importances = collections.OrderedDict(
        (('f_0_bucketized', 0.833933), ('f_2_bucketized', 0.606342),
         ('f_1_bucketized', 0.0)))
    self.assertAllEqual(expected_importances.keys(), importances.keys())
    self.assertAllClose(
        list(expected_importances.values()), list(importances.values()))
    importances = est.experimental_feature_importances(normalize=True)
    expected_importances = collections.OrderedDict(
        (('f_0_bucketized', 0.579010), ('f_2_bucketized', 0.420990),
         ('f_1_bucketized', 0.0)))
    self.assertAllEqual(expected_importances.keys(), importances.keys())
    self.assertAllClose(
        list(expected_importances.values()), list(importances.values()))

  @test_util.run_in_graph_and_eager_modes()
  def testFeatureImportancesOnEmptyEnsemble(self):
    input_fn = _make_train_input_fn(is_classification=True)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)

    class BailOutWithoutTraining(tf.compat.v1.train.SessionRunHook):

      def before_run(self, run_context):
        raise StopIteration('to bail out.')

    # The step-0 checkpoint will have only an empty ensemble.
    est.train(
        input_fn,
        steps=100,  # must stop at 0 anyway.
        hooks=[BailOutWithoutTraining()])

    with self.assertRaisesRegexp(ValueError, 'empty serialized string'):
      est.experimental_feature_importances(normalize=False)

    with self.assertRaisesRegexp(ValueError, 'empty serialized string'):
      est.experimental_feature_importances(normalize=True)

  def _create_fake_checkpoint_with_tree_ensemble_proto(self, est,
                                                       tree_ensemble_text):
    with tf.Graph().as_default():
      with ops.name_scope('boosted_trees') as name:
        tree_ensemble = boosted_trees_ops.TreeEnsemble(name=name)
        tree_ensemble_proto = boosted_trees_pb2.TreeEnsemble()
        text_format.Merge(tree_ensemble_text, tree_ensemble_proto)
        stamp_token, _ = tree_ensemble.serialize()
        restore_op = tree_ensemble.deserialize(
            stamp_token, tree_ensemble_proto.SerializeToString())

        with tf.compat.v1.Session() as sess:
          resources.initialize_resources(resources.shared_resources()).run()
          restore_op.run()
          saver = tf.compat.v1.train.Saver()
          save_path = os.path.join(est.model_dir, 'model.ckpt')
          saver.save(sess, save_path)

  @test_util.run_in_graph_and_eager_modes()
  def testFeatureImportancesOnNonEmptyEnsemble(self):
    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=2,
        max_depth=5)

    tree_ensemble_text = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 2.0
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 3.0
            }
          }
          nodes {
            bucketized_split {
              feature_id: 1
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 2.0
            }
          }
          nodes {
            leaf {
              scalar: -0.34
            }
          }
          nodes {
            leaf {
              scalar: 1.34
            }
          }
          nodes {
            leaf {
              scalar: 0.0
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              left_id: 7
              right_id: 8
            }
            metadata {
              gain: 1.0
            }
          }
          nodes {
            leaf {
              scalar: 3.34
            }
          }
          nodes {
            leaf {
              scalar: 1.34
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 0
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 1.0
            }
          }
          nodes {
            leaf {
              scalar: 3.34
            }
          }
          nodes {
            bucketized_split {
              feature_id: 2
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 1.0
            }
          }
          nodes {
            leaf {
              scalar: 3.34
            }
          }
          nodes {
            leaf {
              scalar: 1.34
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        """
    self._create_fake_checkpoint_with_tree_ensemble_proto(
        est, tree_ensemble_text)

    importances = est.experimental_feature_importances(normalize=False)
    # Gain sum for each features:
    # = 1.0 * [3 + 1, 2, 2] + 1.0 * [1, 1, 0]
    expected_importances = collections.OrderedDict(
        (('f_0_bucketized', 5.0), ('f_2_bucketized', 3.0), ('f_1_bucketized',
                                                            2.0)))
    self.assertAllEqual(expected_importances.keys(), importances.keys())
    self.assertAllClose(
        list(expected_importances.values()), list(importances.values()))
    # Normalize importances.
    importances = est.experimental_feature_importances(normalize=True)
    expected_importances = collections.OrderedDict(
        (('f_0_bucketized', 0.5), ('f_2_bucketized', 0.3), ('f_1_bucketized',
                                                            0.2)))
    self.assertAllEqual(expected_importances.keys(), importances.keys())
    self.assertAllClose(
        list(expected_importances.values()), list(importances.values()))

  @test_util.run_in_graph_and_eager_modes()
  def testFeatureImportancesWithTreeWeights(self):
    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=3,
        max_depth=5)

    tree_ensemble_text = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 0
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 12.5
            }
          }
          nodes {
            bucketized_split {
              feature_id: 1
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 5.0
            }
          }
          nodes {
            leaf {
              scalar: -0.34
            }
          }
          nodes {
            leaf {
              scalar: 1.34
            }
          }
          nodes {
            leaf {
              scalar: 0.0
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 5.0
            }
          }
          nodes {
            leaf {
              scalar: -0.34
            }
          }
          nodes {
            leaf {
              scalar: 1.34
            }
          }
        }
        trees {
          nodes {
            leaf {
              scalar: 0.0
            }
          }
        }
        tree_weights: 0.4
        tree_weights: 0.6
        tree_weights: 1.0
        """
    self._create_fake_checkpoint_with_tree_ensemble_proto(
        est, tree_ensemble_text)

    importances = est.experimental_feature_importances(normalize=False)
    # Gain sum for each features:
    # = 0.4 * [12.5, 0, 5] + 0.6 * [0, 5, 0] + 1.0 * [0, 0, 0]
    expected_importances = collections.OrderedDict(
        (('f_0_bucketized', 5.0), ('f_2_bucketized', 3.0), ('f_1_bucketized',
                                                            2.0)))
    self.assertAllEqual(expected_importances.keys(), importances.keys())
    self.assertAllClose(
        list(expected_importances.values()), list(importances.values()))
    # Normalize importances.
    importances = est.experimental_feature_importances(normalize=True)
    expected_importances = collections.OrderedDict(
        (('f_0_bucketized', 0.5), ('f_2_bucketized', 0.3), ('f_1_bucketized',
                                                            0.2)))
    self.assertAllEqual(expected_importances.keys(), importances.keys())
    self.assertAllClose(
        list(expected_importances.values()), list(importances.values()))

  @test_util.run_in_graph_and_eager_modes()
  def testFeatureImportancesWithAllEmptyTree(self):
    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=2,
        max_depth=5)

    tree_ensemble_text = """
        trees {
          nodes {
            leaf {
              scalar: 0.0
            }
          }
        }
        trees {
          nodes {
            leaf {
              scalar: 0.0
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        """
    self._create_fake_checkpoint_with_tree_ensemble_proto(
        est, tree_ensemble_text)
    importances = est.experimental_feature_importances(normalize=False)
    expected_importances = {
        'f_2_bucketized': 0.0,
        'f_1_bucketized': 0.0,
        'f_0_bucketized': 0.0
    }
    self.assertAllClose(expected_importances, importances)
    with self.assertRaisesRegexp(AssertionError,
                                 'all empty or contain only a root node'):
      est.experimental_feature_importances(normalize=True)

  @test_util.run_in_graph_and_eager_modes()
  def testNegativeFeatureImportances(self):
    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5)

    # In order to generate a negative feature importances,
    # We assign an invalid value -1 to tree_weights here.
    tree_ensemble_text = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 5.0
            }
          }
          nodes {
            leaf {
              scalar: -0.34
            }
          }
          nodes {
            leaf {
              scalar: 1.34
            }
          }
        }
        tree_weights: -1.0
        """
    self._create_fake_checkpoint_with_tree_ensemble_proto(
        est, tree_ensemble_text)
    importances = est.experimental_feature_importances(normalize=False)
    # The gains stored in the splits can be negative
    # if people are using complexity regularization.
    expected_importances = {
        'f_2_bucketized': 0.0,
        'f_0_bucketized': 0.0,
        'f_1_bucketized': -5.0
    }
    self.assertAllClose(expected_importances, importances)
    with self.assertRaisesRegexp(AssertionError, 'non-negative'):
      est.experimental_feature_importances(normalize=True)

  @test_util.run_in_graph_and_eager_modes()
  def testFeatureImportancesNamesForCategoricalColumn(self):
    categorical = tf.feature_column.categorical_column_with_vocabulary_list(
        key='categorical', vocabulary_list=('bad', 'good', 'ok'))
    feature_indicator = tf.feature_column.indicator_column(categorical)
    bucketized_col = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column('continuous', dtype=tf.dtypes.float32),
        BUCKET_BOUNDARIES)
    bucketized_indicator = tf.feature_column.indicator_column(bucketized_col)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=[
            feature_indicator, bucketized_col, bucketized_indicator
        ],
        n_batches_per_layer=1,
        n_trees=2,
        learning_rate=1.0,
        max_depth=1)

    tree_ensemble_text = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 5.0
            }
          }
          nodes {
            bucketized_split {
              feature_id: 4
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 2.0
            }
          }
          nodes {
            leaf {
              scalar: -0.34
            }
          }
          nodes {
            leaf {
              scalar: 1.34
            }
          }
          nodes {
            leaf {
              scalar: 0.0
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 0
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 1.0
            }
          }
          nodes {
            bucketized_split {
              feature_id: 5
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 2.0
            }
          }
          nodes {
            leaf {
              scalar: -2.34
            }
          }
          nodes {
            leaf {
              scalar: 3.34
            }
          }
          nodes {
            leaf {
              scalar: 4.34
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        """
    self._create_fake_checkpoint_with_tree_ensemble_proto(
        est, tree_ensemble_text)

    feature_col_names_expected = [
        'categorical', 'categorical', 'categorical', 'continuous_bucketized',
        'continuous_bucketized', 'continuous_bucketized',
        'continuous_bucketized', 'continuous_bucketized'
    ]
    self.assertAllEqual(feature_col_names_expected, est._feature_col_names)
    importances = est.experimental_feature_importances(normalize=False)
    expected_importances = collections.OrderedDict(
        (('categorical', 6.0), ('continuous_bucketized', 4.0)))
    self.assertAllEqual(expected_importances.keys(), importances.keys())
    self.assertAllClose(
        list(expected_importances.values()), list(importances.values()))
    # Normalize importances.
    importances = est.experimental_feature_importances(normalize=True)
    expected_importances = collections.OrderedDict(
        (('categorical', 0.6), ('continuous_bucketized', 0.4)))
    self.assertAllEqual(expected_importances.keys(), importances.keys())
    self.assertAllClose(
        list(expected_importances.values()), list(importances.values()))

  @test_util.run_in_graph_and_eager_modes()
  def testForCustomDenseColumn(self):

    # Create an arbitrary custom DenseColumn. As long as the column conforms to
    # the FeatureColumn API specifications, it should be supported.
    class MyCustomDense(tf.__internal__.feature_column.DenseColumn):

      def __init__(self, key):
        self.key = key

      @property
      def _is_v2_column(self):
        return True

      @property
      def name(self):
        return self.key

      def parents(self):
        return []

      def get_dense_tensor(self, transformation_cache, state_manager):
        return transformation_cache.get(self, state_manager)

      def transform_feature(self, transformation_cache, state_manager):
        return transformation_cache.get(self.key, state_manager)

      @property
      def shape(self):
        return (1,)

      @property
      def dtype(self):
        return tf.dtypes.float32

      @property
      def variable_shape(self):
        return tf.TensorShape(self.shape)

      def _get_config(cls):
        return {'key': self.key}

      @property
      def parse_example_spec(self):
        return {self.key: tf.io.FixedLenFeature(self.shape, self.dtype, -1)}

    custom_continuous = MyCustomDense(key='f_0')
    numeric = tf.feature_column.numeric_column(key='f_1')
    v1_numeric = feature_column_old._numeric_column('f_2')
    input_fn = _make_train_input_fn(is_classification=False)
    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=[custom_continuous, numeric, v1_numeric],
        n_batches_per_layer=1,
        n_trees=2,
        learning_rate=1.0,
        max_depth=1)
    est.train(input_fn, steps=5)

  @test_util.run_in_graph_and_eager_modes()
  def testTreeComplexityIsSetCorrectly(self):
    input_fn = _make_train_input_fn(is_classification=True)

    num_steps = 10
    # Tree complexity is set but no pruning.
    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5,
        tree_complexity=1e-3)
    with self.assertRaisesRegexp(ValueError, 'Tree complexity have no effect'):
      est.train(input_fn, steps=num_steps)

    # Pruning but no tree complexity.
    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5,
        pruning_mode='pre')
    with self.assertRaisesRegexp(ValueError,
                                 'tree_complexity must be positive'):
      est.train(input_fn, steps=num_steps)

    # All is good.
    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5,
        pruning_mode='pre',
        tree_complexity=1e-3)
    est.train(input_fn, steps=num_steps)

  @test_util.run_in_graph_and_eager_modes()
  def testTrainAndEvaluateEstimator(self):
    input_fn = _make_train_input_fn(is_classification=False)

    est = boosted_trees.BoostedTreesEstimator(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=2,
        head=self._head,
        max_depth=5)

    # It will stop after 10 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=10, finalized_trees=2, attempted_layers=10)
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['average_loss'], 1.008551)

  @test_util.run_in_graph_and_eager_modes()
  def testTrainAndEvaluateEstimatorWithCenterBias(self):
    input_fn = _make_train_input_fn(is_classification=False)

    est = boosted_trees.BoostedTreesEstimator(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=2,
        head=self._head,
        max_depth=5,
        center_bias=True)

    # It will stop after 11 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(input_fn, steps=num_steps)
    # 10 steps for training and 2 step for bias centering.
    self._assert_checkpoint(
        est.model_dir, global_step=12, finalized_trees=2, attempted_layers=10)
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['average_loss'], 0.614642)

  @test_util.run_in_graph_and_eager_modes()
  def testTrainAndEvaluateEstimatorWithPrePruning(self):
    input_fn = _make_train_input_fn(is_classification=False)

    est = boosted_trees.BoostedTreesEstimator(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=2,
        head=self._head,
        max_depth=5,
        tree_complexity=0.001,
        pruning_mode='pre')

    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(input_fn, steps=num_steps)
    # We stop actually after 2*depth*n_trees steps (via a hook) because we still
    # could not grow 2 trees of depth 5 (due to pre-pruning).
    self._assert_checkpoint(
        est.model_dir, global_step=21, finalized_trees=0, attempted_layers=21)
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['average_loss'], 3.83943)

  @test_util.run_in_graph_and_eager_modes()
  def testTrainAndEvaluateEstimatorWithPostPruning(self):
    input_fn = _make_train_input_fn(is_classification=False)

    est = boosted_trees.BoostedTreesEstimator(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=2,
        head=self._head,
        max_depth=5,
        tree_complexity=0.001,
        pruning_mode='post')

    # It will stop after 10 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=10, finalized_trees=2, attempted_layers=10)
    eval_res = est.evaluate(input_fn=input_fn, steps=1)
    self.assertAllClose(eval_res['average_loss'], 2.37652)

  @test_util.run_in_graph_and_eager_modes()
  def testInferEstimator(self):
    train_input_fn = _make_train_input_fn(is_classification=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesEstimator(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5,
        head=self._head)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(train_input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=5, finalized_trees=1, attempted_layers=5)
    # Validate predictions.
    predictions = list(est.predict(input_fn=predict_input_fn))
    self.assertAllClose(
        [[0.571619], [0.262821], [0.124549], [0.956801], [1.769801]],
        [pred['predictions'] for pred in predictions])

  @test_util.run_in_graph_and_eager_modes()
  def testInferEstimatorWithCenterBias(self):
    train_input_fn = _make_train_input_fn(is_classification=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesEstimator(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5,
        center_bias=True,
        head=self._head)

    # It will stop after 6 steps because of the max depth and num trees (5 for
    # training and 2 for bias centering).
    num_steps = 100
    # Train for a few steps, and validate final checkpoint.
    est.train(train_input_fn, steps=num_steps)
    self._assert_checkpoint(
        est.model_dir, global_step=7, finalized_trees=1, attempted_layers=5)
    # Validate predictions.
    predictions = list(est.predict(input_fn=predict_input_fn))

    self.assertAllClose(
        [[1.634501], [1.325703], [1.187431], [2.019683], [2.832683]],
        [pred['predictions'] for pred in predictions])


class BoostedTreesDebugOutputsTest(tf.test.TestCase):
  """Test debug/model explainability outputs for individual predictions.

  Includes directional feature contributions (DFC).
  """

  def setUp(self):
    self._head = boosted_trees._create_regression_head(label_dimension=1)
    self._feature_columns = {
        tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column(
                'f_%d' % i, dtype=tf.dtypes.float32), BUCKET_BOUNDARIES)
        for i in range(NUM_FEATURES)
    }

  @test_util.run_in_graph_and_eager_modes()
  def testBinaryClassifierThatDFCIsInPredictions(self):
    train_input_fn = _make_train_input_fn(is_classification=True)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=3, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5,
        center_bias=True)

    num_steps = 100
    # Train for a few steps. Validate debug outputs in prediction dicts.
    est.train(train_input_fn, steps=num_steps)
    debug_predictions = est.experimental_predict_with_explanations(
        predict_input_fn)
    biases, dfcs = zip(*[(pred['bias'], pred['dfc'])
                         for pred in debug_predictions])
    self.assertAllClose([0.4] * 5, biases)
    expected_dfcs = (collections.OrderedDict(
        (('f_0_bucketized', -0.1210861345357448),
         ('f_2_bucketized', -0.03925492981448114), ('f_1_bucketized', 0.0))),
                     collections.OrderedDict(
                         (('f_0_bucketized', 0.19650601422250574),
                          ('f_2_bucketized',
                           0.02693827052766018), ('f_1_bucketized', 0.0))),
                     collections.OrderedDict(
                         (('f_0_bucketized', 0.16057487356133376),
                          ('f_2_bucketized',
                           0.02693827052766018), ('f_1_bucketized', 0.0))),
                     collections.OrderedDict(
                         (('f_0_bucketized', -0.1210861345357448),
                          ('f_2_bucketized',
                           -0.03925492981448114), ('f_1_bucketized', 0.0))),
                     collections.OrderedDict(
                         (('f_0_bucketized', -0.10832468554550384),
                          ('f_2_bucketized',
                           0.02693827052766018), ('f_1_bucketized', 0.0))))
    self.assertAllClose(expected_dfcs, dfcs)
    # Assert sum(dfcs) + bias == probabilities.
    expected_probabilities = [
        0.23965894, 0.62344426, 0.58751315, 0.23965894, 0.31861359
    ]
    probabilities = [
        sum(dfc.values()) + bias for (dfc, bias) in zip(dfcs, biases)
    ]
    self.assertAllClose(expected_probabilities, probabilities)

    # When user doesn't include bias or dfc in predict_keys, make sure to still
    # include dfc and bias.
    debug_predictions = est.experimental_predict_with_explanations(
        predict_input_fn, predict_keys=['probabilities'])
    for prediction_dict in debug_predictions:
      self.assertTrue('bias' in prediction_dict)
      self.assertTrue('dfc' in prediction_dict)
      self.assertTrue('probabilities' in prediction_dict)
      self.assertEqual(len(prediction_dict), 3)

  @test_util.run_in_graph_and_eager_modes()
  def testRegressorThatDFCIsInPredictions(self):
    train_input_fn = _make_train_input_fn(is_classification=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5,
        center_bias=True)

    num_steps = 100
    # Train for a few steps. Validate debug outputs in prediction dicts.
    est.train(train_input_fn, steps=num_steps)
    debug_predictions = est.experimental_predict_with_explanations(
        predict_input_fn)
    biases, dfcs = zip(*[(pred['bias'], pred['dfc'])
                         for pred in debug_predictions])
    self.assertAllClose([1.8] * 5, biases)
    expected_dfcs = (collections.OrderedDict(
        (('f_1_bucketized', -0.09500002861022949),
         ('f_0_bucketized', -0.07049942016601562), ('f_2_bucketized', 0.0))),
                     collections.OrderedDict(
                         (('f_0_bucketized', -0.5376303195953369),
                          ('f_1_bucketized',
                           0.06333339214324951), ('f_2_bucketized', 0.0))),
                     collections.OrderedDict(
                         (('f_0_bucketized', -0.5175694227218628),
                          ('f_1_bucketized',
                           -0.09500002861022949), ('f_2_bucketized', 0.0))),
                     collections.OrderedDict(
                         (('f_0_bucketized', 0.1563495397567749),
                          ('f_1_bucketized',
                           0.06333339214324951), ('f_2_bucketized', 0.0))),
                     collections.OrderedDict(
                         (('f_0_bucketized', 0.96934974193573),
                          ('f_1_bucketized',
                           0.06333339214324951), ('f_2_bucketized', 0.0))))
    self.assertAllClose(expected_dfcs, dfcs)

    # Assert sum(dfcs) + bias == predictions.
    expected_predictions = [[1.6345005], [1.32570302], [1.1874305],
                            [2.01968288], [2.83268309]]
    predictions = [
        [sum(dfc.values()) + bias] for (dfc, bias) in zip(dfcs, biases)
    ]
    self.assertAllClose(expected_predictions, predictions)

    # Test when user doesn't include bias or dfc in predict_keys.
    debug_predictions = est.experimental_predict_with_explanations(
        predict_input_fn, predict_keys=['predictions'])
    for prediction_dict in debug_predictions:
      self.assertTrue('bias' in prediction_dict)
      self.assertTrue('dfc' in prediction_dict)
      self.assertTrue('predictions' in prediction_dict)
      self.assertEqual(len(prediction_dict), 3)

  @test_util.run_in_graph_and_eager_modes()
  def testDFCClassifierWithOnlyFloatColumn(self):
    feature_columns = {
        tf.feature_column.numeric_column('f_%d' % i, dtype=tf.dtypes.float32)
        for i in range(NUM_FEATURES)
    }
    input_fn = _make_train_input_fn(is_classification=True)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)
    est = boosted_trees.BoostedTreesClassifier(
        feature_columns=feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5,
        center_bias=True,
        quantile_sketch_epsilon=0.33)

    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    est.train(input_fn, steps=num_steps)
    predictions = list(est.predict(predict_input_fn))
    debug_predictions = list(
        est.experimental_predict_with_explanations(predict_input_fn))
    # Predictions using vanilla est.predict are equal to the sum of DFCs.
    for pred, debug_pred in zip(predictions, debug_predictions):
      self.assertTrue('bias' in debug_pred)
      self.assertTrue('dfc' in debug_pred)
      self.assertTrue('probabilities' in debug_pred)
      self.assertAlmostEqual(
          sum(debug_pred['dfc'].values()) + debug_pred['bias'],
          pred['probabilities'][1])

  @test_util.run_in_graph_and_eager_modes()
  def testDFCRegressorWithOnlyFloatColumn(self):
    feature_columns = {
        tf.feature_column.numeric_column('f_%d' % i, dtype=tf.dtypes.float32)
        for i in range(NUM_FEATURES)
    }
    input_fn = _make_train_input_fn(is_classification=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)
    est = boosted_trees.BoostedTreesRegressor(
        feature_columns=feature_columns,
        n_batches_per_layer=1,
        n_trees=1,
        max_depth=5,
        center_bias=True,
        quantile_sketch_epsilon=0.33)
    # It will stop after 5 steps because of the max depth and num trees.
    num_steps = 100
    est.train(input_fn, steps=num_steps)
    predictions = list(est.predict(predict_input_fn))
    debug_predictions = list(
        est.experimental_predict_with_explanations(predict_input_fn))
    # Predictions using vanilla est.predict are equal to the sum of DFCs.
    for pred, debug_pred in zip(predictions, debug_predictions):
      self.assertAlmostEqual(
          sum(debug_pred['dfc'].values()) + debug_pred['bias'],
          pred['predictions'])

  @test_util.run_in_graph_and_eager_modes()
  def testContribEstimatorThatDFCIsInPredictions(self):
    # pylint:disable=protected-access
    head = boosted_trees._create_regression_head(label_dimension=1)
    train_input_fn = _make_train_input_fn(is_classification=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x=FEATURES_DICT, y=None, batch_size=1, num_epochs=1, shuffle=False)

    est = boosted_trees.BoostedTreesEstimator(
        feature_columns=self._feature_columns,
        n_batches_per_layer=1,
        head=head,
        n_trees=1,
        max_depth=5,
        center_bias=True)
    # pylint:enable=protected-access

    num_steps = 100
    # Train for a few steps. Validate debug outputs in prediction dicts.
    est.train(train_input_fn, steps=num_steps)
    debug_predictions = est.experimental_predict_with_explanations(
        predict_input_fn)
    biases, dfcs = zip(*[(pred['bias'], pred['dfc'])
                         for pred in debug_predictions])
    self.assertAllClose([1.8] * 5, biases)
    expected_dfcs = (collections.OrderedDict(
        (('f_1_bucketized', -0.09500002861022949),
         ('f_0_bucketized', -0.07049942016601562), ('f_2_bucketized', 0.0))),
                     collections.OrderedDict(
                         (('f_0_bucketized', -0.5376303195953369),
                          ('f_1_bucketized',
                           0.06333339214324951), ('f_2_bucketized', 0.0))),
                     collections.OrderedDict(
                         (('f_0_bucketized', -0.5175694227218628),
                          ('f_1_bucketized',
                           -0.09500002861022949), ('f_2_bucketized', 0.0))),
                     collections.OrderedDict(
                         (('f_0_bucketized', 0.1563495397567749),
                          ('f_1_bucketized',
                           0.06333339214324951), ('f_2_bucketized', 0.0))),
                     collections.OrderedDict(
                         (('f_0_bucketized', 0.96934974193573),
                          ('f_1_bucketized',
                           0.06333339214324951), ('f_2_bucketized', 0.0))))
    for expected, dfc in zip(expected_dfcs, dfcs):
      self.assertAllEqual(expected.keys(), dfc.keys())
      self.assertAllClose(list(expected.values()), list(dfc.values()))
    # Assert sum(dfcs) + bias == predictions.
    expected_predictions = [[1.6345005], [1.32570302], [1.1874305],
                            [2.01968288], [2.83268309]]
    predictions = [
        [sum(dfc.values()) + bias] for (dfc, bias) in zip(dfcs, biases)
    ]
    self.assertAllClose(expected_predictions, predictions)

    # Test when user doesn't include bias or dfc in predict_keys.
    debug_predictions = est.experimental_predict_with_explanations(
        predict_input_fn, predict_keys=['predictions'])
    for prediction_dict in debug_predictions:
      self.assertIn('bias', prediction_dict)
      self.assertIn('dfc', prediction_dict)
      self.assertIn('predictions', prediction_dict)
      self.assertEqual(len(prediction_dict), 3)


class ModelFnTests(tf.test.TestCase):
  """Tests bt_model_fn including unexposed internal functionalities."""

  def setUp(self):
    self._numeric_feature_columns = {
        tf.feature_column.numeric_column('f_%d' % i, dtype=tf.dtypes.float32)
        for i in range(NUM_FEATURES)
    }
    self._feature_columns = {
        tf.feature_column.bucketized_column(numeric_column, BUCKET_BOUNDARIES)
        for numeric_column in self._numeric_feature_columns
    }

  def _get_expected_ensembles_for_classification(self):
    first_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.387675
            }
          }
          nodes {
            leaf {
              scalar: -0.181818
            }
          }
          nodes {
            leaf {
              scalar: 0.0625
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    second_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.387675
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 3
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 0.0
              original_leaf {
                scalar: -0.181818
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 0
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.105518
              original_leaf {
                scalar: 0.0625
              }
            }
          }
          nodes {
            leaf {
              scalar: -0.348397
            }
          }
          nodes {
            leaf {
              scalar: -0.181818
            }
          }
          nodes {
            leaf {
              scalar: 0.224091
            }
          }
          nodes {
            leaf {
              scalar: 0.056815
            }
          }
        }
        trees {
          nodes {
            leaf {
              scalar: 0.0
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 0
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
          last_layer_node_start: 0
          last_layer_node_end: 1
        }
        """
    third_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.387675
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 3
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 0.0
              original_leaf {
                scalar: -0.181818
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 0
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.105518
              original_leaf {
                scalar: 0.0625
              }
            }
          }
          nodes {
            leaf {
              scalar: -0.348397
            }
          }
          nodes {
            leaf {
              scalar: -0.181818
            }
          }
          nodes {
            leaf {
              scalar: 0.224091
            }
          }
          nodes {
            leaf {
              scalar: 0.056815
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 0
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.287131
            }
          }
          nodes {
            leaf {
              scalar: 0.162042
            }
          }
          nodes {
            leaf {
              scalar: -0.086986
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 2
          num_layers_attempted: 3
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    return (first_round, second_round, third_round)

  def _get_expected_ensembles_for_classification_with_floats(self):
    first_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.387675
            }
          }
          nodes {
            leaf {
              scalar: -0.181818
            }
          }
          nodes {
            leaf {
              scalar: 0.0625
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    second_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.387675
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 0
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 0.0
              original_leaf {
                scalar: -0.181818
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 3
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.46758
              original_leaf {
                scalar: 0.0625
              }
            }
          }
          nodes {
            leaf {
              scalar: -0.181817993522
            }
          }
          nodes {
            leaf {
              scalar: -0.348397
            }
          }
          nodes {
            leaf {
              scalar: 0.238795
            }
          }
          nodes {
            leaf {
              scalar: -0.109513
            }
          }
        }
        trees {
          nodes {
            leaf {
              scalar: 0.0
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 0
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
          last_layer_node_start: 0
          last_layer_node_end: 1
        }
        """
    return (first_round, second_round)

  def _get_expected_ensembles_for_classification_with_bias(self):
    first_round = """
        trees {
          nodes {
            leaf {
              scalar: -0.405086
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
        }
        """
    second_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.407711
              original_leaf {
                scalar: -0.405086
              }
            }
          }
          nodes {
            leaf {
              scalar: -0.556054
            }
          }
          nodes {
            leaf {
              scalar: -0.301233
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    third_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.407711
              original_leaf {
                scalar: -0.405086
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 3
              left_id: 3
              right_id: 4
            }
            metadata {
              original_leaf {
                scalar: -0.556054
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 0
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.09876
              original_leaf {
                scalar: -0.301233
              }
            }
          }
          nodes {
            leaf {
              scalar: -0.698072
            }
          }
          nodes {
            leaf {
              scalar: -0.556054
            }
          }
          nodes {
            leaf {
              scalar: -0.106016
            }
          }
          nodes {
            leaf {
              scalar: -0.27349
            }
          }
        }
        trees {
          nodes {
            leaf {
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
          last_layer_node_end: 1
        }
        """
    forth_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.4077113
              original_leaf {
                scalar: -0.405086
              }
            }
          }
          nodes {
            bucketized_split {
              threshold: 3
              left_id: 3
              right_id: 4
            }
            metadata {
              original_leaf {
                scalar: -0.556054
              }
            }
          }
          nodes {
            bucketized_split {
              threshold: 0
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.09876
              original_leaf {
                scalar: -0.301233
              }
            }
          }
          nodes {
            leaf {
              scalar: -0.698072
            }
          }
          nodes {
            leaf {
              scalar: -0.556054
            }
          }
          nodes {
            leaf {
              scalar: -0.106016
            }
          }
          nodes {
            leaf {
              scalar: -0.27349
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 2
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.289927
            }
          }
          nodes {
            leaf {
              scalar: -0.134588
            }
          }
          nodes {
            leaf {
              scalar: 0.083838
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 1
        }
        growing_metadata {
          num_trees_attempted: 2
          num_layers_attempted: 3
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    return (first_round, second_round, third_round, forth_round)

  def _get_expected_ensembles_for_regression(self):
    first_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 1.169715
            }
          }
          nodes {
            leaf {
              scalar: 0.241322
            }
          }
          nodes {
            leaf {
              scalar: 0.083951
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    second_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 1.169715
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 1
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 2.673407
              original_leaf {
                scalar: 0.241322
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 0
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.324102
              original_leaf {
                scalar: 0.083951
              }
            }
          }
          nodes {
            leaf {
              scalar: 0.563167
            }
          }
          nodes {
            leaf {
              scalar: 0.247047
            }
          }
          nodes {
            leaf {
              scalar: 0.095273
            }
          }
          nodes {
            leaf {
              scalar: 0.222102
            }
          }
        }
        trees {
          nodes {
            leaf {
              scalar: 0.0
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 0
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
          last_layer_node_start: 0
          last_layer_node_end: 1
        }
        """
    third_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 1.169715
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 1
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 2.673407
              original_leaf {
                scalar: 0.241322
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 0
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.324102
              original_leaf {
                scalar: 0.083951
              }
            }
          }
          nodes {
            leaf {
              scalar: 0.563167
            }
          }
          nodes {
            leaf {
              scalar: 0.247047
            }
          }
          nodes {
            leaf {
              scalar: 0.095273
            }
          }
          nodes {
            leaf {
              scalar: 0.222102
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 0
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.981025
            }
          }
          nodes {
            leaf {
              scalar: 0.005166
            }
          }
          nodes {
            leaf {
              scalar: 0.180281
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 2
          num_layers_attempted: 3
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    return (first_round, second_round, third_round)

  def _get_expected_ensembles_for_regression_with_bias(self):
    first_round = """
        trees {
          nodes {
            leaf {
              scalar: 1.799974
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
        }
        """
    second_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 1.190442
              original_leaf {
                scalar: 1.799974
              }
            }
          }
          nodes {
            leaf {
              scalar: 1.862786
            }
          }
          nodes {
            leaf {
              scalar: 1.706149
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    third_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 1.190442
              original_leaf {
                scalar: 1.799974
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 1
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 2.683594
              original_leaf {
                scalar: 1.862786
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 0
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.322693
              original_leaf {
                scalar: 1.706149
              }
            }
          }
          nodes {
            leaf {
              scalar: 2.024487
            }
          }
          nodes {
            leaf {
              scalar: 1.710319
            }
          }
          nodes {
            leaf {
              scalar: 1.559208
            }
          }
          nodes {
            leaf {
              scalar: 1.686037
            }
          }
        }
        trees {
          nodes {
            leaf {
              scalar: 0.0
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 0
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
          last_layer_node_start: 0
          last_layer_node_end: 1
        }
        """
    forth_round = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 1.190442
              original_leaf {
                scalar:  1.799974
              }
            }
          }
          nodes {
            bucketized_split {
              threshold: 1
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 2.683594
              original_leaf {
                scalar: 1.8627863
              }
            }
          }
          nodes {
            bucketized_split {
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.322693
              original_leaf {
                scalar: 1.706149
              }
            }
          }
          nodes {
            leaf {
              scalar: 2.024487
            }
          }
          nodes {
            leaf {
              scalar: 1.710319
            }
          }
          nodes {
            leaf {
              scalar: 1.5592078
            }
          }
          nodes {
            leaf {
              scalar: 1.686037
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 0.972589
            }
          }
          nodes {
            leaf {
              scalar: -0.137592
            }
          }
          nodes {
            leaf {
              scalar: 0.034926
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 1
        }
        growing_metadata {
          num_trees_attempted: 2
          num_layers_attempted: 3
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
        """
    return (first_round, second_round, third_round, forth_round)

  def _get_train_op_and_ensemble(self,
                                 head,
                                 config,
                                 is_classification,
                                 train_in_memory,
                                 center_bias=False,
                                 use_numeric_columns=False):
    """Calls bt_model_fn() and returns the train_op and ensemble_serialzed."""
    train_op, ensemble_serialized, _ = self._get_train_op_and_ensemble_and_boundaries(
        head, config, is_classification, train_in_memory, center_bias,
        use_numeric_columns)
    return train_op, ensemble_serialized

  def _get_train_op_and_ensemble_and_boundaries(self,
                                                head,
                                                config,
                                                is_classification,
                                                train_in_memory,
                                                center_bias=False,
                                                use_numeric_columns=False):
    """Calls bt_model_fn() and returns the train_op and ensemble_serialzed."""
    features, labels = _make_train_input_fn(is_classification)()

    tree_hparams = boosted_trees._TreeHParams(  # pylint:disable=protected-access
        n_trees=2,
        max_depth=2,
        learning_rate=0.1,
        l1=0.,
        l2=0.01,
        tree_complexity=0.,
        min_node_weight=0.,
        center_bias=center_bias,
        pruning_mode='none',
        quantile_sketch_epsilon=0.01)

    if use_numeric_columns:
      columns = self._numeric_feature_columns
      num_resources = 2
    else:
      columns = self._feature_columns
      num_resources = 1
    estimator_spec = boosted_trees._bt_model_fn(  # pylint:disable=protected-access
        features=features,
        labels=labels,
        mode=ModeKeys.TRAIN,
        head=head,
        feature_columns=columns,
        tree_hparams=tree_hparams,
        example_id_column_name=EXAMPLE_ID_COLUMN,
        n_batches_per_layer=1,
        config=config,
        train_in_memory=train_in_memory)
    resources.initialize_resources(resources.shared_resources()).run()
    tf.compat.v1.initializers.global_variables().run()
    tf.compat.v1.initializers.local_variables().run()

    # Gets the train_op and serialized proto of the ensemble.
    shared_resources = resources.shared_resources()
    self.assertEqual(num_resources, len(shared_resources))
    train_op = estimator_spec.train_op
    with tf.control_dependencies([train_op]):
      _, ensemble_serialized = (
          gen_boosted_trees_ops.boosted_trees_serialize_ensemble(
              shared_resources[0].handle))

      if use_numeric_columns:
        bucket_boundaries = boosted_trees_ops.get_bucket_boundaries(
            shared_resources[1].handle, num_features=len(columns))
      else:
        bucket_boundaries = []

    return train_op, ensemble_serialized, bucket_boundaries

  def testTrainClassifierInMemory(self):
    tf.compat.v1.reset_default_graph()
    expected_first, expected_second, expected_third = (
        self._get_expected_ensembles_for_classification())
    with tf.Graph().as_default(), self.cached_session() as sess:
      train_op, ensemble_serialized = self._get_train_op_and_ensemble(
          boosted_trees._create_classification_head(n_classes=2),
          run_config.RunConfig(),
          is_classification=True,
          train_in_memory=True)
      _, serialized = sess.run([train_op, ensemble_serialized])
      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)

      # Run one more time and validate the trained ensemble.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

  def testTrainClassifierWithFloatColumns(self):
    tf.compat.v1.reset_default_graph()
    expected_first, expected_second = (
        self._get_expected_ensembles_for_classification_with_floats())
    expected_buckets = [[-2.001, -2.0001, -1.999, 1., 12.5],
                        [-3., 0., 0.4995, 0.5, 2.],
                        [-100., 3., 20., 50., 102.75]]
    with tf.Graph().as_default(), self.cached_session() as sess:
      train_op, ensemble_serialized, buckets = (
          self._get_train_op_and_ensemble_and_boundaries(
              boosted_trees._create_classification_head(n_classes=2),
              run_config.RunConfig(),
              is_classification=True,
              train_in_memory=False,
              # We are dealing with numeric values that will be quantized.
              use_numeric_columns=True))
      _, serialized, evaluated_buckets = sess.run(
          [train_op, ensemble_serialized, buckets])
      # First an ensemble didn't change
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals('', ensemble_proto)
      self.assertAllClose(expected_buckets, evaluated_buckets)

      # Run one more time and validate the trained ensemble.
      _, serialized, evaluated_buckets = sess.run(
          [train_op, ensemble_serialized, buckets])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)
      self.assertAllClose(expected_buckets, evaluated_buckets)

      # Third round training and validation.
      _, serialized, evaluated_buckets = sess.run(
          [train_op, ensemble_serialized, buckets])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)
      self.assertAllClose(expected_buckets, evaluated_buckets)

  def testTrainClassifierWithFloatColumnsInMemory(self):
    tf.compat.v1.reset_default_graph()
    expected_first, expected_second = (
        self._get_expected_ensembles_for_classification_with_floats())
    expected_buckets = [[-2.001, -2.0001, -1.999, 1., 12.5],
                        [-3., 0., 0.4995, 0.5, 2.],
                        [-100., 3., 20., 50., 102.75]]
    with tf.Graph().as_default(), self.cached_session() as sess:
      train_op, ensemble_serialized, buckets = (
          self._get_train_op_and_ensemble_and_boundaries(
              boosted_trees._create_classification_head(n_classes=2),
              run_config.RunConfig(),
              is_classification=True,
              train_in_memory=True,
              # We are dealing with numeric values that will be quantized.
              use_numeric_columns=True))
      _, serialized, evaluated_buckets = sess.run(
          [train_op, ensemble_serialized, buckets])
      # First an ensemble didn't change
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals('', ensemble_proto)
      self.assertAllClose(expected_buckets, evaluated_buckets)

      # Run one more time and validate the trained ensemble.
      _, serialized, evaluated_buckets = sess.run(
          [train_op, ensemble_serialized, buckets])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)
      self.assertAllClose(expected_buckets, evaluated_buckets)

      # Third round training and validation.
      _, serialized, evaluated_buckets = sess.run(
          [train_op, ensemble_serialized, buckets])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)
      self.assertAllClose(expected_buckets, evaluated_buckets)

  def testTrainClassifierWithCenterBiasInMemory(self):
    tf.compat.v1.reset_default_graph()

    # When bias centering is on, we expect the very first node to have the
    expected_first, expected_second, expected_third, expected_forth = (
        self._get_expected_ensembles_for_classification_with_bias())

    with tf.Graph().as_default(), self.cached_session() as sess:
      train_op, ensemble_serialized = self._get_train_op_and_ensemble(
          boosted_trees._create_classification_head(n_classes=2),
          run_config.RunConfig(),
          is_classification=True,
          train_in_memory=True,
          center_bias=True)

      # 4 iterations to center bias.
      for _ in range(4):
        _, serialized = sess.run([train_op, ensemble_serialized])

      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)

      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

      # Forth round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)

      self.assertProtoEquals(expected_forth, ensemble_proto)

  def testTrainClassifierNonInMemory(self):
    tf.compat.v1.reset_default_graph()
    expected_first, expected_second, expected_third = (
        self._get_expected_ensembles_for_classification())
    with tf.Graph().as_default(), self.cached_session() as sess:
      train_op, ensemble_serialized = self._get_train_op_and_ensemble(
          boosted_trees._create_classification_head(n_classes=2),
          run_config.RunConfig(),
          is_classification=True,
          train_in_memory=False)
      _, serialized = sess.run([train_op, ensemble_serialized])
      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)

      # Run one more time and validate the trained ensemble.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

  def testTrainClassifierWithCenterBiasNonInMemory(self):
    tf.compat.v1.reset_default_graph()

    # When bias centering is on, we expect the very first node to have the
    expected_first, expected_second, expected_third, expected_forth = (
        self._get_expected_ensembles_for_classification_with_bias())

    with tf.Graph().as_default(), self.cached_session() as sess:
      train_op, ensemble_serialized = self._get_train_op_and_ensemble(
          boosted_trees._create_classification_head(n_classes=2),
          run_config.RunConfig(),
          is_classification=True,
          train_in_memory=False,
          center_bias=True)
      # 4 iterations to center bias.
      for _ in range(4):
        _, serialized = sess.run([train_op, ensemble_serialized])
      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)

      # Run one more time and validate the trained ensemble.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

      # Forth round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_forth, ensemble_proto)

  def testTrainRegressorInMemory(self):
    tf.compat.v1.reset_default_graph()
    expected_first, expected_second, expected_third = (
        self._get_expected_ensembles_for_regression())
    with tf.Graph().as_default(), self.cached_session() as sess:
      train_op, ensemble_serialized = self._get_train_op_and_ensemble(
          boosted_trees._create_regression_head(label_dimension=1),
          run_config.RunConfig(),
          is_classification=False,
          train_in_memory=True)
      _, serialized = sess.run([train_op, ensemble_serialized])
      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)

      # Run one more time and validate the trained ensemble.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

  def testTrainRegressorInMemoryWithCenterBias(self):
    tf.compat.v1.reset_default_graph()
    expected_first, expected_second, expected_third, expected_forth = (
        self._get_expected_ensembles_for_regression_with_bias())
    with tf.Graph().as_default(), self.cached_session() as sess:
      train_op, ensemble_serialized = self._get_train_op_and_ensemble(
          boosted_trees._create_regression_head(label_dimension=1),
          run_config.RunConfig(),
          is_classification=False,
          train_in_memory=True,
          center_bias=True)
      # 3 iterations to center bias.
      for _ in range(3):
        _, serialized = sess.run([train_op, ensemble_serialized])
      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)

      self.assertProtoEquals(expected_first, ensemble_proto)

      # Run one more time and validate the trained ensemble.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

      # Forth round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_forth, ensemble_proto)

  def testTrainRegressorNonInMemory(self):
    tf.compat.v1.reset_default_graph()
    expected_first, expected_second, expected_third = (
        self._get_expected_ensembles_for_regression())
    with tf.Graph().as_default(), self.cached_session() as sess:
      train_op, ensemble_serialized = self._get_train_op_and_ensemble(
          boosted_trees._create_regression_head(label_dimension=1),
          run_config.RunConfig(),
          is_classification=False,
          train_in_memory=False)
      _, serialized = sess.run([train_op, ensemble_serialized])
      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)

      # Run one more time and validate the trained ensemble.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

  def testTrainRegressorNotInMemoryWithCenterBias(self):
    tf.compat.v1.reset_default_graph()
    expected_first, expected_second, expected_third, expected_forth = (
        self._get_expected_ensembles_for_regression_with_bias())
    with tf.Graph().as_default(), self.cached_session() as sess:
      train_op, ensemble_serialized = self._get_train_op_and_ensemble(
          boosted_trees._create_regression_head(label_dimension=1),
          run_config.RunConfig(),
          is_classification=False,
          train_in_memory=False,
          center_bias=True)
      # 3 iterations to center the bias (because we are using regularization).
      for _ in range(3):
        _, serialized = sess.run([train_op, ensemble_serialized])

      # Validate the trained ensemble.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_first, ensemble_proto)

      # Run one more time and validate the trained ensemble.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_second, ensemble_proto)

      # Third round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_third, ensemble_proto)

      # Forth round training and validation.
      _, serialized = sess.run([train_op, ensemble_serialized])
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      ensemble_proto.ParseFromString(serialized)
      self.assertProtoEquals(expected_forth, ensemble_proto)


if __name__ == '__main__':
  googletest.main()
