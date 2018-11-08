package(
    default_visibility = [
        "//tensorflow_estimator:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

# INTERNAL TEST RULE PLACEHOLDER

py_library(
    name = "estimator_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":boosted_trees",
        ":dnn",
        ":dnn_with_layer_annotations",
        ":early_stopping",
        ":export",
        ":exporter",
        ":extenders",
        ":head",
        ":hooks",
        ":linear",
        ":logit_fns",
        ":multi_head",
        ":replicate_model_fn",
        ":rnn",
        ":saved_model_estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
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
        "//tensorflow_estimator/python/estimator:numpy_io",
    ],
)

py_library(
    name = "dnn",
    srcs = ["python/estimator/dnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:dnn",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "dnn_test",
    size = "medium",
    srcs = ["python/estimator/dnn_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
        "optonly",  # times out http://b/79220679
    ],
    deps = [
        ":dnn",
        ":head",
        "//tensorflow_estimator/python/estimator:dnn_testing_utils",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:export_export",
        "//tensorflow_estimator/python/estimator:numpy_io",
        "//tensorflow_estimator/python/estimator:prediction_keys",
    ],
)

py_library(
    name = "dnn_with_layer_annotations",
    srcs = ["python/estimator/dnn_with_layer_annotations.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:head",
        "//tensorflow_estimator/python/estimator:model_fn",
        "//tensorflow_estimator/python/estimator:optimizers",
    ],
)

py_test(
    name = "dnn_with_layer_annotations_test",
    size = "medium",
    srcs = ["python/estimator/dnn_with_layer_annotations_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",  # b/67510291
    ],
    deps = [
        ":dnn_with_layer_annotations",
        "//tensorflow_estimator/python/estimator:dnn",
        "//tensorflow_estimator/python/estimator:dnn_testing_utils",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_pandas_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
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
        "//tensorflow_estimator/python/estimator:model_fn",
        "//tensorflow_estimator/python/estimator:util",
    ],
)

py_test(
    name = "extenders_test",
    size = "medium",
    srcs = ["python/estimator/extenders_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],  # b/62863147
    deps = [
        ":extenders",
        ":head",
        "//tensorflow_estimator/python/estimator:estimator_py",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
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

py_test(
    name = "export_test",
    size = "medium",
    srcs = ["python/estimator/export_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],  # b/62863147
    deps = [
        ":export",
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:export_export",
        "//tensorflow_estimator/python/estimator:export_output",
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
        "//tensorflow_estimator/python/estimator:exporter",
    ],
)

py_test(
    name = "exporter_test",
    size = "medium",
    srcs = ["python/estimator/exporter_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":exporter",
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
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
    srcs_version = "PY2AND3",
    deps = [
        ":head",
        "//tensorflow_estimator/python/estimator:estimator_py",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:metric_keys",
        "//tensorflow_estimator/python/estimator:model_fn",
        "//tensorflow_estimator/python/estimator:prediction_keys",
    ],
)

py_library(
    name = "hooks",
    srcs = [
        "python/estimator/hooks.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:estimator_py",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "hooks_test",
    size = "medium",
    srcs = ["python/estimator/hooks_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":hooks",
        "//tensorflow_estimator/python/estimator:estimator_py",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_library(
    name = "linear",
    srcs = ["python/estimator/linear.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:linear",
    ],
)

py_test(
    name = "linear_test",
    size = "medium",
    srcs = ["python/estimator/linear_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":head",
        ":linear",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:export_export",
        "//tensorflow_estimator/python/estimator:linear_testing_utils",
        "//tensorflow_estimator/python/estimator:numpy_io",
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
        "//tensorflow_estimator/python/estimator:linear",
    ],
)

py_test(
    name = "logit_fns_test",
    size = "small",
    srcs = ["python/estimator/logit_fns_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":logit_fns",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
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
    srcs_version = "PY2AND3",
    deps = [
        ":head",
        ":multi_head",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
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
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:head",
        "//tensorflow_estimator/python/estimator:optimizers",
    ],
)

py_test(
    name = "rnn_test",
    size = "medium",
    srcs = ["python/estimator/rnn_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "noasan",  # times out
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
        "//tensorflow_estimator/python/estimator:numpy_io",
        "//tensorflow_estimator/python/estimator:parsing_utils",
    ],
)

py_library(
    name = "early_stopping",
    srcs = ["python/estimator/early_stopping.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "early_stopping_test",
    srcs = ["python/estimator/early_stopping_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":early_stopping",
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_library(
    name = "saved_model_estimator",
    srcs = ["python/estimator/saved_model_estimator.py"],
    deps = [
        ":export",
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:export",
        "//tensorflow_estimator/python/estimator:model_fn",
    ],
)

py_test(
    name = "saved_model_estimator_test",
    size = "medium",
    srcs = ["python/estimator/saved_model_estimator_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "notsan",
    ],
    deps = [
        ":export",
        ":saved_model_estimator",
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:export_export",
        "//tensorflow_estimator/python/estimator:export_output",
        "//tensorflow_estimator/python/estimator:model_fn",
    ],
)
