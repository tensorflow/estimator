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

"""Tests for util.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import training
from tensorflow_estimator.python.estimator import util


@test_util.deprecated_graph_mode_only
class UtilTest(test.TestCase, parameterized.TestCase):
  """Tests for miscellaneous Estimator utils."""

  def test_parse_input_fn_result_tuple(self):
    def _input_fn():
      features = constant_op.constant(np.arange(100))
      labels = constant_op.constant(np.arange(100, 200))
      return features, labels

    features, labels, hooks = util.parse_input_fn_result(_input_fn())

    with self.cached_session() as sess:
      vals = sess.run([features, labels])

    self.assertAllEqual(vals[0], np.arange(100))
    self.assertAllEqual(vals[1], np.arange(100, 200))
    self.assertEqual(hooks, [])

  @parameterized.named_parameters(('DatasetV1', dataset_ops.DatasetV1),
                                  ('DatasetV2', dataset_ops.DatasetV2))
  def test_parse_input_fn_result_dataset(self, dataset_class):

    def _input_fn():
      features = np.expand_dims(np.arange(100), 0)
      labels = np.expand_dims(np.arange(100, 200), 0)
      return dataset_class.from_tensor_slices((features, labels))

    features, labels, hooks = util.parse_input_fn_result(_input_fn())

    with training.MonitoredSession(hooks=hooks) as sess:
      vals = sess.run([features, labels])

    self.assertAllEqual(vals[0], np.arange(100))
    self.assertAllEqual(vals[1], np.arange(100, 200))
    self.assertIsInstance(hooks[0], util._DatasetInitializerHook)

  def test_parse_input_fn_result_mimic_dataset(self):

    class MimicIterator(object):

      @property
      def initializer(self):
        return {}

      def get_next(self):
        features = np.arange(100)
        labels = np.arange(100, 200)
        return ops.convert_to_tensor(features), ops.convert_to_tensor(labels)

    class MimicDataset(object):

      def make_initializable_iterator(self):
        return MimicIterator()

    features, labels, hooks = util.parse_input_fn_result(MimicDataset())

    with training.MonitoredSession(hooks=hooks) as sess:
      vals = sess.run([features, labels])

    self.assertAllEqual(vals[0], np.arange(100))
    self.assertAllEqual(vals[1], np.arange(100, 200))
    self.assertIsInstance(hooks[0], util._DatasetInitializerHook)

  def test_parse_input_fn_result_features_only(self):
    def _input_fn():
      return constant_op.constant(np.arange(100))

    features, labels, hooks = util.parse_input_fn_result(_input_fn())

    with self.cached_session() as sess:
      vals = sess.run([features])

    self.assertAllEqual(vals[0], np.arange(100))
    self.assertEqual(labels, None)
    self.assertEqual(hooks, [])

  @parameterized.named_parameters(('DatasetV1', dataset_ops.DatasetV1),
                                  ('DatasetV2', dataset_ops.DatasetV2))
  def test_parse_input_fn_result_features_only_dataset(self, dataset_class):

    def _input_fn():
      features = np.expand_dims(np.arange(100), 0)
      return dataset_class.from_tensor_slices(features)

    features, labels, hooks = util.parse_input_fn_result(_input_fn())

    with training.MonitoredSession(hooks=hooks) as sess:
      vals = sess.run([features])

    self.assertAllEqual(vals[0], np.arange(100))
    self.assertEqual(labels, None)
    self.assertIsInstance(hooks[0], util._DatasetInitializerHook)

  @parameterized.named_parameters(('DatasetV1', dataset_ops.DatasetV1),
                                  ('DatasetV2', dataset_ops.DatasetV2))
  def test_parse_input_fn_result_invalid(self, dataset_class):

    def _input_fn():
      features = np.expand_dims(np.arange(100), 0)
      labels = np.expand_dims(np.arange(100, 200), 0)
      return dataset_class.from_tensor_slices((features, labels, labels))

    with self.assertRaisesRegexp(ValueError, 'input_fn should return'):
      util.parse_input_fn_result(_input_fn())


if __name__ == '__main__':
  test.main()
