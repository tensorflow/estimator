workspace(name = "org_tensorflow_estimator")

# Use a custom python toolchain to make sure we always use the python binary
# provided by PYTHON_BIN_PATH.
# This is required due to https://github.com/bazelbuild/bazel/issues/7899,
# because --python_path will not work since Bazel 0.27
load("//third_party/py:python_configure.bzl", "python_configure")

python_configure(name = "local_config_py_toolchain")

register_toolchains("@local_config_py_toolchain//:py_toolchain")
