# Placeholder: load py_library
load("//tensorflow_estimator:estimator.bzl", "py_test")

package(default_visibility = ["//tensorflow_estimator:__subpackages__"])

licenses(["notice"])

py_library(
    name = "feature_keys",
    srcs = [
        "feature_keys.py",
    ],
    srcs_version = "PY3",
    deps = ["//tensorflow_estimator/python/estimator:expect_tensorflow_installed"],
)

py_library(
    name = "saved_model_utils",
    srcs = [
        "saved_model_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":feature_keys",
        ":head",
        ":model_utils",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_library(
    name = "model",
    srcs = [
        "model.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":feature_keys",
        ":math_utils",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_library(
    name = "estimators",
    srcs = [
        "estimators.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":ar_model",
        ":feature_keys",
        ":head",
        ":math_utils",
        ":saved_model_utils",
        ":state_management",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "estimators_test",
    srcs = [
        "estimators_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "notap",  # TODO(b/132129465): Re-enable.
    ],
    deps = [
        ":ar_model",
        ":estimators",
        ":feature_keys",
        ":saved_model_utils",
        "//tensorflow_estimator/python/estimator:estimator_py",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_library(
    name = "head",
    srcs = [
        "head.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":feature_keys",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "head_test",
    srcs = [
        "head_test.py",
    ],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    deps = [
        ":estimators",
        ":feature_keys",
        ":head",
        ":model",
        ":state_management",
        "//tensorflow_estimator/python/estimator:estimator_py",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_library(
    name = "model_utils",
    srcs = [
        "model_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":feature_keys",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_library(
    name = "state_management",
    srcs = [
        "state_management.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":feature_keys",
        ":math_utils",
        ":model",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_library(
    name = "ar_model",
    srcs = [
        "ar_model.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":feature_keys",
        ":model",
        ":model_utils",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "ar_model_test",
    srcs = [
        "ar_model_test.py",
    ],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    deps = [
        ":ar_model",
        ":estimators",
        ":feature_keys",
        "//tensorflow_estimator/python/estimator:estimator_py",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "ar_model_training_test",
    srcs = [
        "ar_model_training_test.py",
    ],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    deps = [
        ":ar_model",
        ":estimators",
        ":feature_keys",
        "//tensorflow_estimator/python/estimator:estimator_py",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_library(
    name = "math_utils",
    srcs = [
        "math_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":feature_keys",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "math_utils_test",
    srcs = [
        "math_utils_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":feature_keys",
        ":math_utils",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)
