#!/usr/bin/env bash
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
set -e

function is_absolute {
  [[ "$1" = /* ]] || [[ "$1" =~ ^[a-zA-Z]:[/\\].* ]]
}

function real_path() {
  is_absolute "$1" && echo "$1" || echo "$PWD/${1#./}"
}

function build_wheel() {
  TMPDIR="$1"
  DEST="$2"
  PROJECT_NAME="$3"

  mkdir -p "$TMPDIR"
  echo $(date) : "=== Preparing sources in dir: ${TMPDIR}"

  if [ ! -d bazel-bin/tensorflow_estimator ]; then
    echo "Could not find bazel-bin.  Did you run from the root of the build tree?"
    exit 1
  fi
  cp -r "bazel-bin/tensorflow_estimator/tools/pip_package/build_pip_package.runfiles/org_tensorflow_estimator/tensorflow_estimator" "$TMPDIR"
  cp tensorflow_estimator/tools/pip_package/setup.py "$TMPDIR"

  # Make sure init files exist.
  touch "${TMPDIR}/tensorflow_estimator/__init__.py"
  touch "${TMPDIR}/tensorflow_estimator/_api/__init__.py"
  touch "${TMPDIR}/tensorflow_estimator/_api/v1/__init__.py"
  touch "${TMPDIR}/tensorflow_estimator/_api/v2/__init__.py"
  touch "${TMPDIR}/tensorflow_estimator/python/__init__.py"
  touch "${TMPDIR}/tensorflow_estimator/python/estimator/__init__.py"
  touch "${TMPDIR}/tensorflow_estimator/python/estimator/api/__init__.py"
  touch "${TMPDIR}/tensorflow_estimator/python/estimator/canned/__init__.py"
  touch "${TMPDIR}/tensorflow_estimator/python/estimator/canned/linear_optimizer/__init__.py"
  touch "${TMPDIR}/tensorflow_estimator/python/estimator/canned/linear_optimizer/python/__init__.py"
  touch "${TMPDIR}/tensorflow_estimator/python/estimator/canned/linear_optimizer/python/utils/__init__.py"
  touch "${TMPDIR}/tensorflow_estimator/python/estimator/export/__init__.py"
  touch "${TMPDIR}/tensorflow_estimator/python/estimator/head/__init__.py"
  touch "${TMPDIR}/tensorflow_estimator/python/estimator/hooks/__init__.py"
  touch "${TMPDIR}/tensorflow_estimator/python/estimator/inputs/__init__.py"
  touch "${TMPDIR}/tensorflow_estimator/python/estimator/inputs/queues/__init__.py"

  # We should not have contrib/ directory in Estimator 2.*. If we don't have
  # contrib directory, then we shouldn't create __init__.py files under contrib.
  if [ -d "${TMPDIR}/tensorflow_estimator/contrib/" ]; then
    touch "${TMPDIR}/tensorflow_estimator/contrib/__init__.py"
    touch "${TMPDIR}/tensorflow_estimator/contrib/estimator/__init__.py"
    touch "${TMPDIR}/tensorflow_estimator/contrib/estimator/python/__init__.py"
    touch "${TMPDIR}/tensorflow_estimator/contrib/estimator/python/estimator/__init__.py"
  fi

  pushd ${TMPDIR} > /dev/null
  echo $(date) : "=== Building wheel"
  "${PYTHON_BIN_PATH:-python}" setup.py bdist_wheel --universal --project_name $PROJECT_NAME
  mkdir -p ${DEST}
  cp dist/* ${DEST}
  popd > /dev/null
  echo $(date) : "=== Output wheel file is in: ${DEST}"
  rm -rf "${TMPDIR}"
}

function main() {
  NIGHTLY_BUILD=0

  while true; do
    if [[ -z "$1" ]]; then
      break
    elif [[ "$1" == "--nightly" ]]; then
      NIGHTLY_BUILD=1
    elif [[ "$1" == "--project_name" ]]; then
      shift
      if [[ -z "$1" ]]; then
        break
      fi
      PROJECT_NAME="$1"
    else
      DSTDIR="$(real_path $1)"
    fi
    shift
  done

  if [[ -z ${PROJECT_NAME} ]]; then
    PROJECT_NAME="tensorflow_estimator"
    if [[ ${NIGHTLY_BUILD} == "1" ]]; then
      PROJECT_NAME="tf_estimator_nightly"
    fi
  fi

  SRCDIR="$(mktemp -d -t tmp.XXXXXXXXXX)"

  if [[ -z "$DSTDIR" ]]; then
    echo "No destination dir provided"
    exit 1
  fi



  build_wheel "$SRCDIR" "$DSTDIR" "$PROJECT_NAME"
}

main "$@"
