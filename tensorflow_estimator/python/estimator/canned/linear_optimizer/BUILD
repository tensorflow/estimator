# Placeholder: load py_library
load("//tensorflow_estimator:estimator.bzl", "py_test")

package(default_visibility = ["//tensorflow_estimator:__subpackages__"])

licenses(["notice"])

py_test(
    name = "sdca_test",
    size = "medium",
    srcs = ["python/sdca_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    deps = [
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:expect_proto_cpp_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator:linear",
    ],
)

py_library(
    name = "sdca_ops_py",
    srcs = [
        "__init__.py",
        "python/utils/sdca_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":sharded_mutable_dense_hashtable_py",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "sdca_ops_test",
    size = "medium",
    srcs = ["python/utils/sdca_ops_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    tags = [
        "no_gpu",
        "no_pip_gpu",
    ],
    deps = [
        ":sdca_ops_py",
        "//tensorflow_estimator/python/estimator:expect_proto_cpp_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "sharded_mutable_dense_hashtable_py",
    srcs = ["python/utils/sharded_mutable_dense_hashtable.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "sharded_mutable_dense_hashtable_test",
    size = "small",
    srcs = ["python/utils/sharded_mutable_dense_hashtable_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sharded_mutable_dense_hashtable_py",
        "//tensorflow_estimator/python/estimator:expect_proto_cpp_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)
