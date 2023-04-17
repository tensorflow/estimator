# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""estimator_export tests."""

import sys
import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_export
# pylint: disable=g-deprecated-tf-checker
from tensorflow_estimator.python.estimator import estimator_export


class TestClass(object):
  pass


class ValidateExportTest(tf.test.TestCase):
  """Tests for estimator_export class."""

  def setUp(self):
    super().setUp()
    self._modules = []

  def tearDown(self):
    super().tearDown()
    for name in self._modules:
      del sys.modules[name]
    self._modules = []
    if hasattr(TestClass, '_estimator_api_names'):
      del TestClass._estimator_api_names
    if hasattr(TestClass, '_estimator_api_names_v1'):
      del TestClass._estimator_api_names_v1

  def testExportClassInEstimator(self):
    estimator_export.estimator_export('estimator.TestClass')(TestClass)
    self.assertNotIn('_tf_api_names', TestClass.__dict__)
    self.assertEqual(['estimator.TestClass'], tf_export.get_v1_names(TestClass))

  @tf.compat.v1.test.mock.patch.object(
      logging, 'warning', autospec=True
  )
  def testExportDeprecated(self, mock_warning):
    export_decorator = estimator_export.estimator_export('estimator.TestClass')
    export_decorator(TestClass)

    # Deprecation should trigger a runtime warning
    TestClass()
    self.assertEqual(1, mock_warning.call_count)
    # Deprecation should only warn once, upon first call
    TestClass()
    self.assertEqual(1, mock_warning.call_count)


if __name__ == '__main__':
  tf.test.main()
