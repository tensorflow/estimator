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
"""Tests for multi_head.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow_estimator.python.estimator.util import tf_keras
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.head import head_utils as test_lib
from tensorflow_estimator.python.estimator.head import multi_head as multi_head_lib
from tensorflow_estimator.python.estimator.head import multi_label_head
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys


@test_util.run_all_in_graph_and_eager_modes
class MultiHeadTest(tf.test.TestCase):

  def test_no_heads(self):
    with self.assertRaisesRegexp(ValueError,
                                 r'Must specify heads\. Given: \[\]'):
      multi_head_lib.MultiHead(heads=[])

  def test_head_name_missing(self):
    head1 = multi_label_head.MultiLabelHead(n_classes=2, name='head1')
    head2 = multi_label_head.MultiLabelHead(n_classes=3)
    with self.assertRaisesRegexp(ValueError,
                                 r'All given heads must have name specified\.'):
      multi_head_lib.MultiHead([head1, head2])

  def test_head_weights_wrong_size(self):
    head1 = multi_label_head.MultiLabelHead(n_classes=2, name='head1')
    head2 = multi_label_head.MultiLabelHead(n_classes=3, name='head2')
    with self.assertRaisesRegexp(
        ValueError, r'heads and head_weights must have the same size\. '
        r'Given len\(heads\): 2. Given len\(head_weights\): 1\.'):
      multi_head_lib.MultiHead([head1, head2], head_weights=[1.])

  def test_name(self):
    head1 = multi_label_head.MultiLabelHead(n_classes=2, name='head1')
    head2 = multi_label_head.MultiLabelHead(n_classes=3, name='head2')
    multi_head = multi_head_lib.MultiHead([head1, head2])
    self.assertEqual('head1_head2', multi_head.name)

  def test_predict_two_heads_logits_dict(self):
    """Tests predict with logits as dict."""
    head1 = multi_label_head.MultiLabelHead(n_classes=2, name='head1')
    head2 = multi_label_head.MultiLabelHead(n_classes=3, name='head2')
    multi_head = multi_head_lib.MultiHead([head1, head2])

    logits = {
        'head1': np.array([[-1., 1.], [-1.5, 1.]], dtype=np.float32),
        'head2': np.array([[2., -2., 2.], [-3., 2., -2.]], dtype=np.float32)
    }
    expected_probabilities = {
        'head1': tf.math.sigmoid(logits['head1']),
        'head2': tf.math.sigmoid(logits['head2']),
    }
    pred_keys = prediction_keys.PredictionKeys

    predictions = multi_head.predictions(logits)
    self.assertAllClose(logits['head1'],
                        self.evaluate(predictions[('head1', pred_keys.LOGITS)]))
    self.assertAllClose(logits['head2'],
                        self.evaluate(predictions[('head2', pred_keys.LOGITS)]))
    self.assertAllClose(
        expected_probabilities['head1'],
        self.evaluate(predictions[('head1', pred_keys.PROBABILITIES)]))
    self.assertAllClose(
        expected_probabilities['head2'],
        self.evaluate(predictions[('head2', pred_keys.PROBABILITIES)]))
    if tf.executing_eagerly():
      return

    spec = multi_head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=ModeKeys.PREDICT,
        logits=logits)
    self.assertItemsEqual((test_lib._DEFAULT_SERVING_KEY, 'predict', 'head1',
                           'head1/classification', 'head1/predict', 'head2',
                           'head2/classification', 'head2/predict'),
                          spec.export_outputs.keys())
    # Assert predictions and export_outputs.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      predictions = sess.run(spec.predictions)
      self.assertAllClose(logits['head1'],
                          predictions[('head1', pred_keys.LOGITS)])
      self.assertAllClose(logits['head2'],
                          predictions[('head2', pred_keys.LOGITS)])
      self.assertAllClose(expected_probabilities['head1'],
                          predictions[('head1', pred_keys.PROBABILITIES)])
      self.assertAllClose(expected_probabilities['head2'],
                          predictions[('head2', pred_keys.PROBABILITIES)])

      self.assertAllClose(
          expected_probabilities['head1'],
          sess.run(spec.export_outputs[test_lib._DEFAULT_SERVING_KEY].scores))
      self.assertAllClose(expected_probabilities['head1'],
                          sess.run(spec.export_outputs['head1'].scores))
      self.assertAllClose(expected_probabilities['head2'],
                          sess.run(spec.export_outputs['head2'].scores))
      self.assertAllClose(
          expected_probabilities['head1'],
          sess.run(
              spec.export_outputs['predict'].outputs['head1/probabilities']))
      self.assertAllClose(
          expected_probabilities['head2'],
          sess.run(
              spec.export_outputs['predict'].outputs['head2/probabilities']))
      self.assertAllClose(
          expected_probabilities['head1'],
          sess.run(
              spec.export_outputs['head1/predict'].outputs['probabilities']))
      self.assertAllClose(
          expected_probabilities['head2'],
          sess.run(
              spec.export_outputs['head2/predict'].outputs['probabilities']))

  def test_predict_two_heads_logits_tensor(self):
    """Tests predict with logits as Tensor."""
    head1 = multi_label_head.MultiLabelHead(n_classes=2, name='head1')
    head2 = multi_label_head.MultiLabelHead(n_classes=3, name='head2')
    multi_head = multi_head_lib.MultiHead([head1, head2])

    logits = np.array([[-1., 1., 2., -2., 2.], [-1.5, 1., -3., 2., -2.]],
                      dtype=np.float32)
    expected_logits1 = np.array([[-1., 1.], [-1.5, 1.]], dtype=np.float32)
    expected_logits2 = np.array([[2., -2., 2.], [-3., 2., -2.]],
                                dtype=np.float32)
    expected_probabilities = {
        'head1': tf.math.sigmoid(expected_logits1),
        'head2': tf.math.sigmoid(expected_logits2),
    }
    pred_keys = prediction_keys.PredictionKeys

    predictions = multi_head.predictions(logits)
    self.assertAllClose(expected_logits1,
                        self.evaluate(predictions[('head1', pred_keys.LOGITS)]))
    self.assertAllClose(expected_logits2,
                        self.evaluate(predictions[('head2', pred_keys.LOGITS)]))
    self.assertAllClose(
        expected_probabilities['head1'],
        self.evaluate(predictions[('head1', pred_keys.PROBABILITIES)]))
    self.assertAllClose(
        expected_probabilities['head2'],
        self.evaluate(predictions[('head2', pred_keys.PROBABILITIES)]))
    if tf.executing_eagerly():
      return

    spec = multi_head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=ModeKeys.PREDICT,
        logits=logits)
    self.assertItemsEqual((test_lib._DEFAULT_SERVING_KEY, 'predict', 'head1',
                           'head1/classification', 'head1/predict', 'head2',
                           'head2/classification', 'head2/predict'),
                          spec.export_outputs.keys())
    # Assert predictions and export_outputs.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      predictions = sess.run(spec.predictions)
      self.assertAllClose(expected_logits1,
                          predictions[('head1', pred_keys.LOGITS)])
      self.assertAllClose(expected_logits2,
                          predictions[('head2', pred_keys.LOGITS)])
      self.assertAllClose(expected_probabilities['head1'],
                          predictions[('head1', pred_keys.PROBABILITIES)])
      self.assertAllClose(expected_probabilities['head2'],
                          predictions[('head2', pred_keys.PROBABILITIES)])

      self.assertAllClose(
          expected_probabilities['head1'],
          sess.run(spec.export_outputs[test_lib._DEFAULT_SERVING_KEY].scores))
      self.assertAllClose(expected_probabilities['head1'],
                          sess.run(spec.export_outputs['head1'].scores))
      self.assertAllClose(expected_probabilities['head2'],
                          sess.run(spec.export_outputs['head2'].scores))

  def test_predict_two_heads_logits_tensor_multi_dim(self):
    """Tests predict with multi-dimensional logits of shape [2, 2, 5]."""
    head1 = regression_head.RegressionHead(label_dimension=2, name='head1')
    head2 = regression_head.RegressionHead(label_dimension=3, name='head2')
    multi_head = multi_head_lib.MultiHead([head1, head2])

    logits = np.array([[[-1., 1., 2., -2., 2.], [-1., 1., 2., -2., 2.]],
                       [[-1.5, 1., -3., 2., -2.], [-1.5, 1., -3., 2., -2.]]],
                      dtype=np.float32)
    expected_logits1 = np.array(
        [[[-1., 1.], [-1., 1.]], [[-1.5, 1.], [-1.5, 1.]]], dtype=np.float32)
    expected_logits2 = np.array(
        [[[2., -2., 2.], [2., -2., 2.]], [[-3., 2., -2.], [-3., 2., -2.]]],
        dtype=np.float32)
    pred_keys = prediction_keys.PredictionKeys

    predictions = multi_head.predictions(logits)
    self.assertAllClose(
        expected_logits1,
        self.evaluate(predictions[('head1', pred_keys.PREDICTIONS)]))
    self.assertAllClose(
        expected_logits2,
        self.evaluate(predictions[('head2', pred_keys.PREDICTIONS)]))
    if tf.executing_eagerly():
      return

    spec = multi_head.create_estimator_spec(
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=ModeKeys.PREDICT,
        logits=logits)
    self.assertItemsEqual(
        (test_lib._DEFAULT_SERVING_KEY, 'predict', 'head1', 'head1/regression',
         'head1/predict', 'head2', 'head2/regression', 'head2/predict'),
        spec.export_outputs.keys())
    # Assert predictions and export_outputs.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      predictions = sess.run(spec.predictions)
      self.assertAllClose(expected_logits1,
                          predictions[('head1', pred_keys.PREDICTIONS)])
      self.assertAllClose(expected_logits2,
                          predictions[('head2', pred_keys.PREDICTIONS)])

      self.assertAllClose(
          expected_logits1,
          sess.run(spec.export_outputs[test_lib._DEFAULT_SERVING_KEY].value))
      self.assertAllClose(expected_logits1,
                          sess.run(spec.export_outputs['head1'].value))
      self.assertAllClose(expected_logits2,
                          sess.run(spec.export_outputs['head2'].value))

  def test_eval_two_heads_with_weights(self):
    head1 = multi_label_head.MultiLabelHead(n_classes=2, name='head1')
    head2 = multi_label_head.MultiLabelHead(n_classes=3, name='head2')
    multi_head = multi_head_lib.MultiHead([head1, head2], head_weights=[1., 2.])

    logits = {
        'head1':
            np.array([[-10., 10.], [-15., 10.]], dtype=np.float32),
        'head2':
            np.array([[20., -20., 20.], [-30., 20., -20.]], dtype=np.float32),
    }
    labels = {
        'head1': np.array([[1, 0], [1, 1]], dtype=np.int64),
        'head2': np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int64),
    }
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # For large logits, sigmoid cross entropy loss is approximated as:
    # loss = labels * (logits < 0) * (-logits) +
    #        (1 - labels) * (logits > 0) * logits =>
    # head1: expected_unweighted_loss = [[10., 10.], [15., 0.]]
    # loss = ((10 + 10) / 2 + (15 + 0) / 2) / 2 = 8.75
    # head2: expected_unweighted_loss = [[20., 20., 20.], [30., 0., 0]]
    # loss = ((20 + 20 + 20) / 3 + (30 + 0 + 0) / 3) / 2 = 15
    expected_loss_head1 = 8.75
    expected_loss_head2 = 15.
    expected_loss = 1. * expected_loss_head1 + 2. * expected_loss_head2
    tol = 1e-3
    keys = metric_keys.MetricKeys
    expected_metrics = {
        keys.LOSS + '/head1': expected_loss_head1,
        keys.LOSS + '/head2': expected_loss_head2,
        # Average loss over examples.
        keys.LOSS_MEAN + '/head1': expected_loss_head1,
        keys.LOSS_MEAN + '/head2': expected_loss_head2,
        # auc and auc_pr cannot be reliably calculated for only 4-6 samples, but
        # this assert tests that the algorithm remains consistent.
        keys.AUC + '/head1': 0.1667,
        keys.AUC + '/head2': 0.3333,
        keys.AUC_PR + '/head1': 0.60228,
        keys.AUC_PR + '/head2': 0.40152,
    }

    if tf.executing_eagerly():
      loss = multi_head.loss(
          labels, logits, features=features, mode=ModeKeys.EVAL)
      self.assertIsNotNone(loss)
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)

      eval_metrics = multi_head.metrics()
      updated_metrics = multi_head.update_metrics(eval_metrics, features,
                                                  logits, labels)
      self.assertItemsEqual(expected_metrics.keys(), updated_metrics.keys())
      self.assertAllClose(
          expected_metrics,
          {k: updated_metrics[k].result() for k in updated_metrics},
          rtol=tol,
          atol=tol)
      return

    spec = multi_head.create_estimator_spec(
        features=features, mode=ModeKeys.EVAL, logits=logits, labels=labels)
    # Assert spec contains expected tensors.
    self.assertIsNotNone(spec.loss)
    self.assertItemsEqual(expected_metrics.keys(), spec.eval_metric_ops.keys())
    self.assertIsNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    test_lib._assert_no_hooks(self, spec)
    # Assert predictions, loss, and metrics.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNone(spec.scaffold.summary_op)
      value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
      update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
      loss, _ = sess.run((spec.loss, update_ops))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      # Check results of value ops (in `metrics`).
      self.assertAllClose(
          expected_metrics, {k: value_ops[k].eval() for k in value_ops},
          rtol=tol,
          atol=tol)

  def test_train_loss_one_head(self):
    head1 = multi_label_head.MultiLabelHead(n_classes=2, name='head1')
    multi_head = multi_head_lib.MultiHead([head1])

    logits = {'head1': np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)}
    labels = {'head1': np.array([[1, 0], [1, 1]], dtype=np.int64)}
    loss = multi_head.loss(
        labels=labels,
        logits=logits,
        features={'x': np.array(((42,),), dtype=np.int32)},
        mode=ModeKeys.TRAIN)
    tol = 1e-3
    # Unreduced loss of the head is [[(10 + 10) / 2], (15 + 0) / 2]
    # (averaged over classes, averaged over examples).
    # loss = sum(unreduced_loss) / 2 = sum([10, 7.5]) / 2 = 8.75
    self.assertAllClose(8.75, self.evaluate(loss), rtol=tol, atol=tol)

  def test_train_loss_two_heads_with_weights(self):
    # Use different example weighting for each head weighting.
    weights1 = np.array([[1.], [2.]], dtype=np.float32)
    weights2 = np.array([[2.], [3.]])
    head1 = multi_label_head.MultiLabelHead(
        n_classes=2, name='head1', weight_column='weights1')
    head2 = multi_label_head.MultiLabelHead(
        n_classes=3, name='head2', weight_column='weights2')
    multi_head = multi_head_lib.MultiHead([head1, head2], head_weights=[1., 2.])

    logits = {
        'head1':
            np.array([[-10., 10.], [-15., 10.]], dtype=np.float32),
        'head2':
            np.array([[20., -20., 20.], [-30., 20., -20.]], dtype=np.float32),
    }
    labels = {
        'head1': np.array([[1, 0], [1, 1]], dtype=np.int64),
        'head2': np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int64),
    }
    training_loss = multi_head.loss(
        logits=logits,
        labels=labels,
        features={
            'x': np.array(((42,),), dtype=np.int32),
            'weights1': weights1,
            'weights2': weights2
        },
        mode=ModeKeys.TRAIN)
    tol = 1e-3
    # loss of the first head is [[(10 + 10) / 2], [(15 + 0) / 2]]
    # = [10, 7.5]
    # training_loss = (1 * 10 + 2 * 7.5) / 2 = 12.5
    # head-weighted unreduced_loss = 1 * [10, 7.5]
    # loss of the second head is [[(20 + 20 + 20) / 3], [(30 + 0 + 0) / 3]]
    # = [20, 10]
    # training_loss = (2 * 20 + 3 * 10) / 2 = 35
    # head-weighted unreduced_loss = 2 * [20, 10]
    # head-weighted training_loss = 1 * 12.5 + 2 * 35 = 82.5
    self.assertAllClose(82.5, self.evaluate(training_loss), rtol=tol, atol=tol)

  def test_train_loss_logits_tensor(self):
    """Tests loss with logits Tensor."""
    weights1 = np.array([[1.], [2.]], dtype=np.float32)
    weights2 = np.array([[2.], [3.]])
    head1 = multi_label_head.MultiLabelHead(
        n_classes=2, name='head1', weight_column='weights1')
    head2 = multi_label_head.MultiLabelHead(
        n_classes=3, name='head2', weight_column='weights2')
    multi_head = multi_head_lib.MultiHead([head1, head2], head_weights=[1., 2.])

    logits = np.array(
        [[-10., 10., 20., -20., 20.], [-15., 10., -30., 20., -20.]],
        dtype=np.float32)
    labels = {
        'head1': np.array([[1, 0], [1, 1]], dtype=np.int64),
        'head2': np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int64),
    }
    training_loss = multi_head.loss(
        logits=logits,
        labels=labels,
        features={
            'x': np.array(((42,),), dtype=np.int32),
            'weights1': weights1,
            'weights2': weights2
        },
        mode=ModeKeys.TRAIN)
    tol = 1e-3
    # loss of the first head is [[(10 + 10) / 2], [(15 + 0) / 2]]
    # = [10, 7.5]
    # training_loss = (1 * 10 + 2 * 7.5) / 2 = 12.5
    # head-weighted unreduced_loss = 1 * [10, 7.5]
    # loss of the second head is [[(20 + 20 + 20) / 3], [(30 + 0 + 0) / 3]]
    # = [20, 10]
    # training_loss = (2 * 20 + 3 * 10) / 2 = 35
    # head-weighted unreduced_loss = 2 * [20, 10]
    # head-weighted training_loss = 1 * 12.5 + 2 * 35 = 82.5
    self.assertAllClose(82.5, self.evaluate(training_loss), rtol=tol, atol=tol)

  def test_train_loss_logits_tensor_wrong_shape(self):
    """Tests loss with a logits Tensor of the wrong shape."""
    weights1 = np.array([[1.], [2.]], dtype=np.float32)
    weights2 = np.array([[2.], [3.]])
    head1 = multi_label_head.MultiLabelHead(
        n_classes=2, name='head1', weight_column='weights1')
    head2 = multi_label_head.MultiLabelHead(
        n_classes=3, name='head2', weight_column='weights2')
    multi_head = multi_head_lib.MultiHead([head1, head2], head_weights=[1., 2.])

    # logits tensor is 2x6 instead of 2x5
    logits = np.array(
        [[-10., 10., 20., -20., 20., 70.], [-15., 10., -30., 20., -20., 80.]],
        dtype=np.float32)
    labels = {
        'head1': np.array([[1, 0], [1, 1]], dtype=np.int64),
        'head2': np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int64),
    }
    with self.assertRaisesRegexp(ValueError, r'Could not split logits'):
      multi_head.loss(
          features={
              'x': np.array(((42,),), dtype=np.int32),
              'weights1': weights1,
              'weights2': weights2
          },
          mode=ModeKeys.TRAIN,
          logits=logits,
          labels=labels)

  def test_train_loss_logits_tensor_multi_dim(self):
    """Tests loss with multi-dimensional logits of shape [2, 2, 5]."""
    head1 = regression_head.RegressionHead(label_dimension=2, name='head1')
    head2 = regression_head.RegressionHead(label_dimension=3, name='head2')
    multi_head = multi_head_lib.MultiHead([head1, head2])

    logits = np.array([[[-1., 1., 2., -2., 2.], [-1., 1., 2., -2., 2.]],
                       [[-1.5, 1.5, -2., 2., -2.], [-1.5, 1.5, -2., 2., -2.]]],
                      dtype=np.float32)
    labels = {
        'head1':
            np.array([[[1., 0.], [1., 0.]], [[1.5, 1.5], [1.5, 1.5]]],
                     dtype=np.float32),
        'head2':
            np.array(
                [[[0., 1., 0.], [0., 1., 0.]], [[2., 2., 0.], [2., 2., 0.]]],
                dtype=np.float32),
    }
    # Loss for the first head:
    # loss1 = ((1+1)^2 + (0-1)^2 + (1+1)^2 + (0-1)^2 +
    #          (1.5+1.5)^2 + (1.5-1.5)^2 + (1.5+1.5)^2 + (1.5-1.5)^2) / 8
    #       = 3.5
    # Loss for the second head:
    # loss2 = ((0-2)^2 + (1+2)^2 + (0-2)^2 + (0-2)^2 + (1+2)^2 + (0-2)^2 +
    #          (2+2)^2 + (2-2)^2 + (0+2)^2 + (2+2)^2 + (2-2)^2 + (0+2)^2) / 12
    #       = 6.167
    expected_training_loss = 3.5 + 6.167

    training_loss = multi_head.loss(
        logits=logits, labels=labels, features={}, mode=ModeKeys.TRAIN)
    tol = 1e-3
    self.assertAllClose(
        expected_training_loss,
        self.evaluate(training_loss),
        rtol=tol,
        atol=tol)

  def test_train_loss_logits_tensor_multi_dim_wrong_shape(self):
    """Tests loss with a multi-dimensional logits tensor of the wrong shape."""
    head1 = regression_head.RegressionHead(label_dimension=2, name='head1')
    head2 = regression_head.RegressionHead(label_dimension=3, name='head2')
    multi_head = multi_head_lib.MultiHead([head1, head2])

    # logits tensor is 2x2x4 instead of 2x2x5
    logits = np.array([[[-1., 1., 2., -2.], [-1., 1., 2., -2.]],
                       [[-1.5, 1.5, -2., 2.], [-1.5, 1.5, -2., 2.]]],
                      dtype=np.float32)
    labels = {
        'head1':
            np.array([[[1., 0.], [1., 0.]], [[1.5, 1.5], [1.5, 1.5]]],
                     dtype=np.float32),
        'head2':
            np.array(
                [[[0., 1., 0.], [0., 1., 0.]], [[2., 2., 0.], [2., 2., 0.]]],
                dtype=np.float32),
    }
    with self.assertRaisesRegexp(ValueError, r'Could not split logits'):
      multi_head.loss(
          features={}, mode=ModeKeys.TRAIN, logits=logits, labels=labels)

  def test_train_one_head(self):
    head1 = multi_label_head.MultiLabelHead(n_classes=2, name='head1')
    multi_head = multi_head_lib.MultiHead([head1])

    logits = {'head1': np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)}
    expected_probabilities = {
        'head1': tf.math.sigmoid(logits['head1']),
    }
    labels = {'head1': np.array([[1, 0], [1, 1]], dtype=np.int64)}
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # For large logits, sigmoid cross entropy loss is approximated as:
    # loss = labels * (logits < 0) * (-logits) +
    #        (1 - labels) * (logits > 0) * logits =>
    # expected_unweighted_loss = [[10., 10.], [15., 0.]]
    # loss = ((10 + 10) / 2 + (15 + 0) / 2) / 2 = 8.75
    expected_loss = 8.75
    tol = 1e-3
    loss = multi_head.loss(
        logits=logits, labels=labels, features=features, mode=ModeKeys.TRAIN)
    self.assertAllClose(expected_loss, self.evaluate(loss), rtol=tol, atol=tol)
    if tf.executing_eagerly():
      return

    expected_train_result = 'my_train_op'

    def _train_op_fn(loss):
      return tf.strings.join([
          tf.constant(expected_train_result),
          tf.strings.as_string(loss, precision=3)
      ])

    spec = multi_head.create_estimator_spec(
        features=features,
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)
    self.assertIsNotNone(spec.loss)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    test_lib._assert_no_hooks(self, spec)
    # Assert predictions, loss, train_op, and summaries.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str, predictions = sess.run(
          (spec.loss, spec.train_op, spec.scaffold.summary_op,
           spec.predictions))
      self.assertAllClose(
          logits['head1'],
          predictions[('head1', prediction_keys.PredictionKeys.LOGITS)])
      self.assertAllClose(
          expected_probabilities['head1'],
          predictions[('head1', prediction_keys.PredictionKeys.PROBABILITIES)])
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.3f}'.format(expected_train_result, expected_loss)),
          train_result)
      test_lib._assert_simple_summaries(
          self, {
              metric_keys.MetricKeys.LOSS: expected_loss,
              metric_keys.MetricKeys.LOSS + '/head1': expected_loss,
          }, summary_str, tol)

  def test_train_one_head_with_optimizer(self):
    head1 = multi_label_head.MultiLabelHead(n_classes=2, name='head1')
    multi_head = multi_head_lib.MultiHead([head1])

    logits = {'head1': np.array([[-10., 10.], [-15., 10.]], dtype=np.float32)}
    labels = {'head1': np.array([[1, 0], [1, 1]], dtype=np.int64)}
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # For large logits, sigmoid cross entropy loss is approximated as:
    # loss = labels * (logits < 0) * (-logits) +
    #        (1 - labels) * (logits > 0) * logits =>
    # expected_unweighted_loss = [[10., 10.], [15., 0.]]
    # loss = ((10 + 10) / 2 + (15 + 0) / 2) / 2 = 8.75
    expected_loss = 8.75
    tol = 1e-3
    loss = multi_head.loss(
        logits=logits, labels=labels, features=features, mode=ModeKeys.TRAIN)
    self.assertAllClose(expected_loss, self.evaluate(loss), rtol=tol, atol=tol)
    if tf.executing_eagerly():
      return

    expected_train_result = 'my_train_op'

    class _Optimizer(tf_keras.optimizers.Optimizer):

      def get_updates(self, loss, params):
        del params
        return [
            tf.strings.join([
                tf.constant(expected_train_result),
                tf.strings.as_string(loss, precision=3)
            ])
        ]

      def get_config(self):
        config = super(_Optimizer, self).get_config()
        return config

    spec = multi_head.create_estimator_spec(
        features=features,
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        optimizer=_Optimizer('my_optimizer'),
        trainable_variables=[tf.Variable([1.0, 2.0], dtype=tf.dtypes.float32)])

    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      loss, train_result = sess.run((spec.loss, spec.train_op))
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.3f}'.format(expected_train_result, expected_loss)),
          train_result)

  def test_train_two_heads_with_weights(self):
    head1 = multi_label_head.MultiLabelHead(n_classes=2, name='head1')
    head2 = multi_label_head.MultiLabelHead(n_classes=3, name='head2')
    multi_head = multi_head_lib.MultiHead([head1, head2], head_weights=[1., 2.])

    logits = {
        'head1':
            np.array([[-10., 10.], [-15., 10.]], dtype=np.float32),
        'head2':
            np.array([[20., -20., 20.], [-30., 20., -20.]], dtype=np.float32),
    }
    expected_probabilities = {
        'head1': tf.math.sigmoid(logits['head1']),
        'head2': tf.math.sigmoid(logits['head2']),
    }
    labels = {
        'head1': np.array([[1, 0], [1, 1]], dtype=np.int64),
        'head2': np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int64),
    }
    features = {'x': np.array(((42,),), dtype=np.int32)}
    # For large logits, sigmoid cross entropy loss is approximated as:
    # loss = labels * (logits < 0) * (-logits) +
    #        (1 - labels) * (logits > 0) * logits =>
    # head1: expected_unweighted_loss = [[10., 10.], [15., 0.]]
    # loss = ((10 + 10) / 2 + (15 + 0) / 2) / 2 = 8.75
    # head2: expected_unweighted_loss = [[20., 20., 20.], [30., 0., 0]]
    # loss = ((20 + 20 + 20) / 3 + (30 + 0 + 0) / 3) / 2 = 15
    # Average over classes, weighted sum over batch and heads.
    expected_loss_head1 = 8.75
    expected_loss_head2 = 15.0
    expected_loss = 1. * expected_loss_head1 + 2. * expected_loss_head2
    tol = 1e-3
    loss = multi_head.loss(
        logits=logits, labels=labels, features=features, mode=ModeKeys.TRAIN)
    self.assertAllClose(expected_loss, self.evaluate(loss), rtol=tol, atol=tol)
    if tf.executing_eagerly():
      return

    expected_train_result = 'my_train_op'

    def _train_op_fn(loss):
      return tf.strings.join([
          tf.constant(expected_train_result),
          tf.strings.as_string(loss, precision=3)
      ])

    spec = multi_head.create_estimator_spec(
        features=features,
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn)
    self.assertIsNotNone(spec.loss)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    test_lib._assert_no_hooks(self, spec)
    # Assert predictions, loss, train_op, and summaries.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str, predictions = sess.run(
          (spec.loss, spec.train_op, spec.scaffold.summary_op,
           spec.predictions))
      self.assertAllClose(
          logits['head1'],
          predictions[('head1', prediction_keys.PredictionKeys.LOGITS)])
      self.assertAllClose(
          expected_probabilities['head1'],
          predictions[('head1', prediction_keys.PredictionKeys.PROBABILITIES)])
      self.assertAllClose(
          logits['head2'],
          predictions[('head2', prediction_keys.PredictionKeys.LOGITS)])
      self.assertAllClose(
          expected_probabilities['head2'],
          predictions[('head2', prediction_keys.PredictionKeys.PROBABILITIES)])
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.3f}'.format(expected_train_result, expected_loss)),
          train_result)
      test_lib._assert_simple_summaries(
          self, {
              metric_keys.MetricKeys.LOSS: expected_loss,
              metric_keys.MetricKeys.LOSS + '/head1': expected_loss_head1,
              metric_keys.MetricKeys.LOSS + '/head2': expected_loss_head2,
          }, summary_str, tol)

  def test_train_with_regularization_losses(self):
    head1 = multi_label_head.MultiLabelHead(n_classes=2, name='head1')
    head2 = multi_label_head.MultiLabelHead(n_classes=3, name='head2')
    multi_head = multi_head_lib.MultiHead([head1, head2], head_weights=[1., 2.])

    logits = {
        'head1':
            np.array([[-10., 10.], [-15., 10.]], dtype=np.float32),
        'head2':
            np.array([[20., -20., 20.], [-30., 20., -20.]], dtype=np.float32),
    }
    expected_probabilities = {
        'head1': tf.math.sigmoid(logits['head1']),
        'head2': tf.math.sigmoid(logits['head2']),
    }
    labels = {
        'head1': np.array([[1, 0], [1, 1]], dtype=np.int64),
        'head2': np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int64),
    }
    features = {'x': np.array(((42,),), dtype=np.int32)}
    regularization_losses = [1.5, 0.5]

    # For large logits, sigmoid cross entropy loss is approximated as:
    # loss = labels * (logits < 0) * (-logits) +
    #        (1 - labels) * (logits > 0) * logits =>
    # head1: expected_unweighted_loss = [[10., 10.], [15., 0.]]
    # loss1 = ((10 + 10) / 2 + (15 + 0) / 2) / 2 = 8.75
    # head2: expected_unweighted_loss = [[20., 20., 20.], [30., 0., 0]]
    # loss2 = ((20 + 20 + 20) / 3 + (30 + 0 + 0) / 3) / 2 = 15
    # Average over classes, weighted sum over batch and heads.
    # weights = [1., 2.]
    # merged_training_loss = 1. * loss1 + 2. * loss2
    # training_loss = merged_training_loss + regularization_loss
    #               = 1. * loss1 + 2. * loss2 + sum([1.5, 0.5])
    expected_loss_head1 = 8.75
    expected_loss_head2 = 15.0
    expected_regularization_loss = 2.
    # training loss.
    expected_loss = (1. * expected_loss_head1 + 2. * expected_loss_head2 +
                     expected_regularization_loss)
    tol = 1e-3
    loss = multi_head.loss(
        logits=logits,
        labels=labels,
        features=features,
        mode=ModeKeys.TRAIN,
        regularization_losses=regularization_losses)
    self.assertAllClose(expected_loss, self.evaluate(loss), rtol=tol, atol=tol)
    if tf.executing_eagerly():
      return

    keys = metric_keys.MetricKeys
    expected_train_result = 'my_train_op'

    def _train_op_fn(loss):
      return tf.strings.join([
          tf.constant(expected_train_result),
          tf.strings.as_string(loss, precision=3)
      ])

    spec = multi_head.create_estimator_spec(
        features=features,
        mode=ModeKeys.TRAIN,
        logits=logits,
        labels=labels,
        train_op_fn=_train_op_fn,
        regularization_losses=regularization_losses)
    self.assertIsNotNone(spec.loss)
    self.assertEqual({}, spec.eval_metric_ops)
    self.assertIsNotNone(spec.train_op)
    self.assertIsNone(spec.export_outputs)
    test_lib._assert_no_hooks(self, spec)
    # Assert predictions, loss, train_op, and summaries.
    with self.cached_session() as sess:
      test_lib._initialize_variables(self, spec.scaffold)
      self.assertIsNotNone(spec.scaffold.summary_op)
      loss, train_result, summary_str, predictions = sess.run(
          (spec.loss, spec.train_op, spec.scaffold.summary_op,
           spec.predictions))
      self.assertAllClose(
          logits['head1'],
          predictions[('head1', prediction_keys.PredictionKeys.LOGITS)])
      self.assertAllClose(
          expected_probabilities['head1'],
          predictions[('head1', prediction_keys.PredictionKeys.PROBABILITIES)])
      self.assertAllClose(
          logits['head2'],
          predictions[('head2', prediction_keys.PredictionKeys.LOGITS)])
      self.assertAllClose(
          expected_probabilities['head2'],
          predictions[('head2', prediction_keys.PredictionKeys.PROBABILITIES)])
      self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
      self.assertEqual(
          six.b('{0:s}{1:.3f}'.format(expected_train_result, expected_loss)),
          train_result)
      test_lib._assert_simple_summaries(
          self, {
              keys.LOSS_REGULARIZATION: expected_regularization_loss,
              keys.LOSS: expected_loss,
              keys.LOSS + '/head1': expected_loss_head1,
              keys.LOSS + '/head2': expected_loss_head2,
          }, summary_str, tol)


@test_util.deprecated_graph_mode_only
class MultiHeadForEstimator(tf.test.TestCase):
  """Tests for create_estimator_spec running in Graph mode only."""

  def test_loss_reduction_must_be_same(self):
    """Tests the loss reduction must be the same for different heads."""
    head1 = multi_label_head.MultiLabelHead(
        n_classes=2,
        name='head1',
        loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    head2 = multi_label_head.MultiLabelHead(
        n_classes=3, name='head2', loss_reduction=tf.losses.Reduction.AUTO)
    multi_head = multi_head_lib.MultiHead([head1, head2])
    logits = {
        'head1':
            np.array([[-10., 10.], [-15., 10.]], dtype=np.float32),
        'head2':
            np.array([[20., -20., 20.], [-30., 20., -20.]], dtype=np.float32),
    }
    labels = {
        'head1': np.array([[1, 0], [1, 1]], dtype=np.int64),
        'head2': np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int64),
    }
    with self.assertRaisesRegexp(ValueError, 'must be the same'):
      multi_head.create_estimator_spec(
          features={'x': np.array(((42,),), dtype=np.int32)},
          mode=ModeKeys.TRAIN,
          logits=logits,
          labels=labels)


if __name__ == '__main__':
  tf.test.main()
