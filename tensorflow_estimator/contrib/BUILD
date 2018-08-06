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
    deps = [
        ":baseline",
        ":boosted_trees",
        ":dnn",
        ":dnn_linear_combined",
        ":early_stopping",
        ":export",
        ":extenders",
        ":head",
        ":hooks",
        ":linear",
        ":logit_fns",
        ":multi_head",
        ":replicate_model_fn",
        ":rnn",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
    ],
)

py_library(
    name = "baseline",
    srcs = ["python/estimator/baseline.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python:baseline",
        "//tensorflow_estimator/python:estimator",
    ],
)

py_test(
    name = "baseline_test",
    size = "small",
    srcs = ["python/estimator/baseline_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":baseline",
        ":head",
        "//tensorflow_estimator/python:expect_numpy_installed",
        "//tensorflow_estimator/python:expect_six_installed",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
        "//tensorflow_estimator/python:export_export",
        "//tensorflow_estimator/python:metric_keys",
        "//tensorflow_estimator/python:numpy_io",
    ],
)

py_library(
    name = "boosted_trees",
    srcs = ["python/estimator/boosted_trees.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python:boosted_trees",
        "//tensorflow_estimator/python:estimator",
    ],
)

py_test(
    name = "boosted_trees_test",
    size = "medium",
    srcs = ["python/estimator/boosted_trees_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":boosted_trees",
        "//tensorflow_estimator/python:expect_numpy_installed",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
        "//tensorflow_estimator/python:numpy_io",
    ],
)

py_library(
    name = "dnn",
    srcs = ["python/estimator/dnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python:dnn",
        "//tensorflow_estimator/python:estimator",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
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
        "//tensorflow_estimator/python:dnn_testing_utils",
        "//tensorflow_estimator/python:expect_numpy_installed",
        "//tensorflow_estimator/python:expect_six_installed",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
        "//tensorflow_estimator/python:export_export",
        "//tensorflow_estimator/python:numpy_io",
        "//tensorflow_estimator/python:prediction_keys",
    ],
)

py_library(
    name = "dnn_linear_combined",
    srcs = ["python/estimator/dnn_linear_combined.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python:dnn_linear_combined",
        "//tensorflow_estimator/python:estimator",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
    ],
)

py_test(
    name = "dnn_linear_combined_test",
    size = "medium",
    srcs = ["python/estimator/dnn_linear_combined_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":dnn_linear_combined",
        ":head",
        "//tensorflow_estimator/python:dnn_testing_utils",
        "//tensorflow_estimator/python:expect_numpy_installed",
        "//tensorflow_estimator/python:expect_six_installed",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
        "//tensorflow_estimator/python:export_export",
        "//tensorflow_estimator/python:linear_testing_utils",
        "//tensorflow_estimator/python:numpy_io",
        "//tensorflow_estimator/python:prediction_keys",
    ],
)

py_library(
    name = "extenders",
    srcs = [
        "python/estimator/extenders.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python:estimator",
        "//tensorflow_estimator/python:expect_six_installed",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
        "//tensorflow_estimator/python:model_fn",
        "//tensorflow_estimator/python:util",
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
        "//tensorflow_estimator/python:estimator_py",
        "//tensorflow_estimator/python:expect_numpy_installed",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
        "//tensorflow_estimator/python:linear",
    ],
)

py_library(
    name = "export",
    srcs = [
        "python/estimator/export.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python:model_fn",
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
        "//tensorflow_estimator/python:estimator",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
        "//tensorflow_estimator/python:export_export",
        "//tensorflow_estimator/python:export_output",
        "//tensorflow_estimator/python:model_fn",
    ],
)

py_library(
    name = "head",
    srcs = [
        "python/estimator/head.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python:expect_tensorflow_installed",
        "//tensorflow_estimator/python:export_output",
        "//tensorflow_estimator/python:head",
        "//tensorflow_estimator/python:metric_keys",
        "//tensorflow_estimator/python:model_fn",
        "//tensorflow_estimator/python:prediction_keys",
    ],
)

py_test(
    name = "head_test",
    size = "medium",
    srcs = ["python/estimator/head_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":head",
        "//tensorflow_estimator/python:expect_numpy_installed",
        "//tensorflow_estimator/python:expect_six_installed",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
        "//tensorflow_estimator/python:metric_keys",
        "//tensorflow_estimator/python:model_fn",
        "//tensorflow_estimator/python:prediction_keys",
    ],
)

py_library(
    name = "hooks",
    srcs = [
        "python/estimator/hooks.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python:estimator_py",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
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
        "//tensorflow_estimator/python:estimator_py",
        "//tensorflow_estimator/python:expect_numpy_installed",
        "//tensorflow_estimator/python:expect_six_installed",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
    ],
)

py_library(
    name = "linear",
    srcs = ["python/estimator/linear.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python:estimator",
        "//tensorflow_estimator/python:linear",
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
        "//tensorflow_estimator/python:expect_numpy_installed",
        "//tensorflow_estimator/python:expect_six_installed",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
        "//tensorflow_estimator/python:export_export",
        "//tensorflow_estimator/python:linear_testing_utils",
        "//tensorflow_estimator/python:numpy_io",
        "//tensorflow_estimator/python:prediction_keys",
    ],
)

py_library(
    name = "logit_fns",
    srcs = [
        "python/estimator/logit_fns.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python:dnn",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
        "//tensorflow_estimator/python:linear",
    ],
)

py_test(
    name = "logit_fns_test",
    size = "small",
    srcs = ["python/estimator/logit_fns_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":logit_fns",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
        "//tensorflow_estimator/python:model_fn",
    ],
)

py_library(
    name = "multi_head",
    srcs = [
        "python/estimator/multi_head.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python:expect_six_installed",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
        "//tensorflow_estimator/python:export_output",
        "//tensorflow_estimator/python:head",
        "//tensorflow_estimator/python:metric_keys",
        "//tensorflow_estimator/python:model_fn",
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
        "//tensorflow_estimator/python:expect_numpy_installed",
        "//tensorflow_estimator/python:expect_six_installed",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
        "//tensorflow_estimator/python:metric_keys",
        "//tensorflow_estimator/python:model_fn",
        "//tensorflow_estimator/python:prediction_keys",
    ],
)

py_library(
    name = "replicate_model_fn",
    srcs = [
        "python/estimator/replicate_model_fn.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python:expect_six_installed",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
        "//tensorflow_estimator/python:export_output",
        "//tensorflow_estimator/python:model_fn",
        "//tensorflow_estimator/python:util",
    ],
)

py_library(
    name = "rnn",
    srcs = ["python/estimator/rnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":extenders",
        "//tensorflow_estimator/python:estimator",
        "//tensorflow_estimator/python:expect_six_installed",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
        "//tensorflow_estimator/python:head",
        "//tensorflow_estimator/python:optimizers",
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
        "//tensorflow_estimator/python:expect_numpy_installed",
        "//tensorflow_estimator/python:expect_six_installed",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
        "//tensorflow_estimator/python:numpy_io",
        "//tensorflow_estimator/python:parsing_utils",
    ],
)

py_library(
    name = "early_stopping",
    srcs = ["python/estimator/early_stopping.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python:estimator",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
    ],
)

py_test(
    name = "early_stopping_test",
    srcs = ["python/estimator/early_stopping_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":early_stopping",
        "//tensorflow_estimator/python:estimator",
        "//tensorflow_estimator/python:expect_tensorflow_installed",
    ],
)
