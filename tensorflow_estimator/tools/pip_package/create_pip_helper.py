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
"""Utils to help build and verify pip package for TensorFlow Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import fnmatch
import os

PIP_EXCLUDED_FILES = frozenset([
    'tensorflow_estimator/python/estimator/canned/optimizers_test_v2.py',
    'tensorflow_estimator/python/estimator/canned/dnn_test_fc_v2.py',
    'tensorflow_estimator/python/estimator/canned/dnn_test_fc_v1.py',
    'tensorflow_estimator/python/estimator/canned/v1/dnn_estimator_test_v1.py',
    'tensorflow_estimator/python/estimator/canned/v1/linear_test_v1.py',
    'tensorflow_estimator/python/estimator/canned/v1/dnn_linear_combined_estimator_test_v1.py',
    'tensorflow_estimator/python/estimator/canned/v1/dnn_linear_combined_test_v1.py',
    'tensorflow_estimator/python/estimator/canned/v1/baseline_estimator_test_v1.py',
    'tensorflow_estimator/python/estimator/canned/v1/linear_estimator_test_v1.py',
    'tensorflow_estimator/python/estimator/canned/v1/baseline_test_v1.py',
    'tensorflow_estimator/python/estimator/canned/v1/dnn_test_fc_v1_v1.py',
    'tensorflow_estimator/python/estimator/canned/v1/dnn_test_fc_v2_v1.py',
    'tensorflow_estimator/python/estimator/api/extractor_wrapper.py',
    'tensorflow_estimator/python/estimator/api/generator_wrapper.py',
    'tensorflow_estimator/tools/pip_package/setup.py',
    'tensorflow_estimator/tools/pip_package/create_pip_helper.py',
])

# Directories that should not have __init__.py files generated within them.
EXCLUDED_INIT_FILE_DIRECTORIES = frozenset(['tensorflow_estimator/tools'])


class PipPackagingError(Exception):
  pass


def create_init_files(pip_root):
  """Create __init__.py in pip directory tree.

  These files are auto-generated by Bazel when doing typical build/test, but
  do not get auto-generated by the pip build process. Currently, the entire
  directory tree is just python files, so its fine to just create all of the
  init files.

  Args:
    pip_root: Root directory of code being packaged into pip.

  Returns:
    True: contrib code is included in pip.
  """
  has_contrib = False
  for path, subdirs, _ in os.walk(pip_root):
    has_contrib = has_contrib or '/contrib/' in path
    for subdir in subdirs:
      init_file_path = os.path.join(path, subdir, '__init__.py')
      if any(excluded_path in init_file_path
             for excluded_path in EXCLUDED_INIT_FILE_DIRECTORIES):
        continue
      if not os.path.exists(init_file_path):
        # Create empty file
        open(init_file_path, 'w').close()
  return has_contrib


def verify_python_files_in_pip(pip_root, bazel_root, has_contrib):
  """Verifies all expected files are packaged into Pip.

  Args:
    pip_root: Root directory of code being packaged into pip.
    bazel_root: Root directory of Estimator Bazel workspace.
    has_contrib: Code from contrib/ should be included in pip.

  Raises:
    PipPackagingError: Missing file in pip.
  """
  for path, _, files in os.walk(bazel_root):
    if not has_contrib and '/contrib/' in path:
      continue
    python_files = set(fnmatch.filter(files, '*.py'))
    python_test_files = set(fnmatch.filter(files, '*test.py'))
    # We only care about python files in the pip package, see create_init_files.
    files = python_files - python_test_files
    for f in files:
      pip_path = os.path.join(pip_root, os.path.relpath(path, bazel_root), f)
      file_name = os.path.join(path, f)
      path_exists = os.path.exists(pip_path)
      file_excluded = file_name.lstrip('./') in PIP_EXCLUDED_FILES
      if not path_exists and not file_excluded:
        raise PipPackagingError(
            ('Pip package missing the file %s. If this is expected, add it '
             'to PIP_EXCLUDED_FILES in create_pip_helper.py. Otherwise, '
             'make sure it is a build dependency of the pip package') %
            file_name)
      if path_exists and file_excluded:
        raise PipPackagingError(
            ('File in PIP_EXCLUDED_FILES included in pip. %s' % file_name))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--bazel-root',
      type=str,
      required=True,
      help='Root directory of Estimator Bazel workspace.')
  parser.add_argument(
      '--pip-root',
      type=str,
      required=True,
      help='Root directory of code being packaged into pip.')

  args = parser.parse_args()
  has_contrib = create_init_files(args.pip_root)
  verify_python_files_in_pip(args.pip_root, args.bazel_root, has_contrib)


if __name__ == '__main__':
  main()
