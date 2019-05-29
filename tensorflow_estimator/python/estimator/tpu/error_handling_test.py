# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Error Handling tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow_estimator.python.estimator.tpu import error_handling


class ErrorHandlingTest(test.TestCase):

  def catch_and_raise(self, error):
    er = error_handling.ErrorRendezvous(1)
    with er.catch_errors(source='infeed'):
      raise error
    er.raise_errors()

  def testInterestingError(self):
    with self.assertRaises(errors.InternalError):
      self.catch_and_raise(errors.InternalError('message', None, None))

  def testIgnoredError(self):
    """Expect no error to be raised."""
    self.catch_and_raise(errors.AbortedError('message', None, None))

if __name__ == '__main__':
  test.main()
