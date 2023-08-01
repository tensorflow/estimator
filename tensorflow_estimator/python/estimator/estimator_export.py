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
"""Utilities for exporting TensorFlow Estimator symbols to the API.

Exporting a function or a class:

To export a function or a class use the estimator_export decorator. For e.g.:
```python
@estimator_export('foo', 'bar.foo')
def foo(...):
  ...
```

If a function is assigned to a variable, you can export it by calling
estimator_export explicitly. For e.g.:
```python
foo = get_foo(...)
estimator_export('foo', 'bar.foo')(foo)
```


Exporting a constant
```python
foo = 1
estimator_export('consts.foo').export_constant(__name__, 'foo')
```
"""

from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_export


class estimator_export(tf_export.api_export):  # pylint: disable=invalid-name
  """Provides ways to export symbols to the TensorFlow Estimator API."""

  def __init__(self, *args, **kwargs):  # pylint: disable=g-doc-args
    """Export under the names *args (first one is considered canonical).

    All symbols exported by this decorator are exported under the `estimator`
    API name.

    Args:
      *args: API names in dot delimited format.
      **kwargs: Optional keyed arguments.
        v1: Names for the TensorFlow V1 API. If not set, we will use V2 API
          names both for TensorFlow V1 and V2 APIs.
    """
    super().__init__(*args, api_name=tf_export.ESTIMATOR_API_NAME, **kwargs)

  def __call__(self, func):
    """Calls this decorator.

    Args:
      func: decorated symbol (function or class).

    Returns:
      The input function with _tf_api_names attribute set and marked as
      deprecated.
    """
    func = deprecation.deprecated(None, 'Use tf.keras instead.')(func)
    return super().__call__(func)
