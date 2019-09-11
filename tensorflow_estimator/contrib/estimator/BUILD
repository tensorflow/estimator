# INTERNAL TEST RULE PLACEHOLDER
load("//tensorflow_estimator:estimator.bzl", "py_test")

package(
    default_visibility = [
        "//tensorflow_estimator:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "estimator_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":boosted_trees",
        ":dnn_with_layer_annotations",
        ":export",
        ":exporter",
        ":extenders",
        ":head",
        ":logit_fns",
        ":multi_head",
        ":replicate_model_fn",
        ":rnn",
        ":saved_model_estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "boosted_trees",
    srcs = ["python/estimator/boosted_trees.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:boosted_trees",
    ],
)

py_test(
    name = "boosted_trees_test",
    size = "medium",
    srcs = ["python/estimator/boosted_trees_test.py"],
    python_version = "PY2",
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":boosted_trees",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator:numpy_io",
    ],
)

py_library(
    name = "dnn_with_layer_annotations",
    srcs = ["python/estimator/dnn_with_layer_annotations.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator:head",
        "//tensorflow_estimator/python/estimator:model_fn",
        "//tensorflow_estimator/python/estimator:optimizers",
    ],
)

py_test(
    name = "dnn_with_layer_annotations_test",
    size = "medium",
    srcs = ["python/estimator/dnn_with_layer_annotations_test.py"],
    python_version = "PY2",
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",  # b/67510291
    ],
    deps = [
        ":dnn_with_layer_annotations",
        "//tensorflow_estimator/python/estimator:dnn",
        "//tensorflow_estimator/python/estimator:dnn_testing_utils_v1",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_pandas_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator:export_export",
        "//tensorflow_estimator/python/estimator:numpy_io",
        "//tensorflow_estimator/python/estimator:pandas_io",
        "//tensorflow_estimator/python/estimator:prediction_keys",
    ],
)

py_library(
    name = "extenders",
    srcs = [
        "python/estimator/extenders.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator:model_fn",
        "//tensorflow_estimator/python/estimator:util",
    ],
)

py_test(
    name = "extenders_test",
    size = "medium",
    srcs = ["python/estimator/extenders_test.py"],
    python_version = "PY2",
    shard_count = 2,
    srcs_version = "PY2AND3",
    tags = [
        "no_oss",  # contrib
        "notap",  # contrib
        "notsan",  # b/62863147
    ],
    deps = [
        ":extenders",
        ":head",
        "//tensorflow_estimator/python/estimator:estimator_py",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator:linear",
    ],
)

py_library(
    name = "export",
    srcs = [
        "python/estimator/export.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:model_fn",
    ],
)

py_library(
    name = "exporter",
    srcs = [
        "python/estimator/exporter.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator:exporter",
    ],
)

py_test(
    name = "exporter_test",
    size = "medium",
    srcs = ["python/estimator/exporter_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":exporter",
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator:exporter",
    ],
)

py_library(
    name = "head",
    srcs = [
        "python/estimator/head.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator:export_output",
        "//tensorflow_estimator/python/estimator:head",
        "//tensorflow_estimator/python/estimator:metric_keys",
        "//tensorflow_estimator/python/estimator:model_fn",
        "//tensorflow_estimator/python/estimator:prediction_keys",
    ],
)

py_test(
    name = "head_test",
    size = "medium",
    srcs = ["python/estimator/head_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":head",
        "//tensorflow_estimator/python/estimator:estimator_py",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator:metric_keys",
        "//tensorflow_estimator/python/estimator:model_fn",
        "//tensorflow_estimator/python/estimator:prediction_keys",
    ],
)

py_library(
    name = "logit_fns",
    srcs = [
        "python/estimator/logit_fns.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:dnn",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator:linear",
    ],
)

py_test(
    name = "logit_fns_test",
    size = "small",
    srcs = ["python/estimator/logit_fns_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":logit_fns",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator:model_fn",
    ],
)

py_library(
    name = "multi_head",
    srcs = [
        "python/estimator/multi_head.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator:export_output",
        "//tensorflow_estimator/python/estimator:head",
        "//tensorflow_estimator/python/estimator:metric_keys",
        "//tensorflow_estimator/python/estimator:model_fn",
    ],
)

py_test(
    name = "multi_head_test",
    size = "small",
    srcs = ["python/estimator/multi_head_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":head",
        ":multi_head",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator:metric_keys",
        "//tensorflow_estimator/python/estimator:model_fn",
        "//tensorflow_estimator/python/estimator:prediction_keys",
    ],
)

py_library(
    name = "replicate_model_fn",
    srcs = [
        "python/estimator/replicate_model_fn.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator:export_output",
        "//tensorflow_estimator/python/estimator:model_fn",
        "//tensorflow_estimator/python/estimator:util",
    ],
)

py_library(
    name = "rnn",
    srcs = ["python/estimator/rnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":extenders",
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:binary_class_head",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator:multi_class_head",
        "//tensorflow_estimator/python/estimator:optimizers",
        "//tensorflow_estimator/python/estimator:sequential_head",
    ],
)

py_test(
    name = "rnn_test",
    size = "medium",
    srcs = ["python/estimator/rnn_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    tags = [
        "no_oss",  # contrib
        "no_pip",
        "noasan",  # times out
        "notap",  # contrib
        "notsan",
        "optonly",  # times out http://b/79220679
    ],
    deps = [
        ":head",
        ":rnn",
        "//tensorflow_estimator/python/estimator:estimator_py",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator:multi_class_head",
        "//tensorflow_estimator/python/estimator:numpy_io",
        "//tensorflow_estimator/python/estimator:parsing_utils",
        "//tensorflow_estimator/python/estimator:sequential_head",
    ],
)

py_library(
    name = "saved_model_estimator",
    srcs = ["python/estimator/saved_model_estimator.py"],
    deps = [
        "//tensorflow_estimator/python/estimator",
    ],
)
