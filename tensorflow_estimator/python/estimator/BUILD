load("//tensorflow_estimator:estimator.bzl", "py_test")

package(default_visibility = ["//tensorflow_estimator:internal"])

licenses(["notice"])  # Apache 2.0

py_test(
    name = "tf_estimator_doctest",
    srcs = ["tf_estimator_doctest.py"],
    python_version = "PY3",
    tags = [
        "no_oss_py2",
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [":estimator_py"],
)

py_library(
    name = "estimator_py",
    srcs = [
        "estimator_lib.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":base_head",
        ":baseline",
        ":basic_session_run_hooks",
        ":binary_class_head",
        ":boosted_trees",
        ":checkpoint_converter",
        ":dnn",
        ":dnn_linear_combined",
        ":early_stopping",
        ":estimator",
        ":export",
        ":exporter",
        ":extenders",
        ":fake_summary_writer",
        ":function",
        ":hooks",
        ":inputs",
        ":keras",
        ":kmeans",
        ":linear",
        ":mode_keys",
        ":model_fn",
        ":multi_class_head",
        ":multi_head",
        ":multi_label_head",
        ":parsing_utils",
        ":regression_head",
        ":rnn",
        ":run_config",
        ":saved_model_estimator",
        ":sequential_head",
        ":session_run_hook",
        ":training",
        "//tensorflow_estimator/python/estimator:expect_tensorboard_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator/canned/timeseries:estimators",
        "//tensorflow_estimator/python/estimator/tpu:tpu_estimator",
    ],
)

py_library(
    name = "exporter",
    srcs = ["exporter.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":gc",
        ":metric_keys",
        ":util",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "exporter_test",
    size = "medium",
    srcs = ["exporter_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":exporter",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "extenders",
    srcs = ["extenders.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":mode_keys",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "extenders_test",
    size = "medium",
    srcs = ["extenders_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = ["notsan"],  # b/62863147
    deps = [
        ":extenders",
        ":linear",
        ":run_config",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "gc",
    srcs = ["gc.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "gc_test",
    size = "small",
    srcs = ["gc_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":gc",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "hooks",
    srcs = ["hooks/hooks.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "hooks_test",
    srcs = ["hooks/hooks_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":estimator_py",
        ":hooks",
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "model_fn",
    srcs = ["model_fn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":mode_keys",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "model_fn_test",
    size = "small",
    srcs = ["model_fn_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":export_output",
        ":model_fn",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "mode_keys",
    srcs = ["mode_keys.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "training",
    srcs = ["training.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":exporter",
        ":run_config",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "training_test",
    size = "medium",
    srcs = ["training_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":dnn",
        ":estimator",
        ":exporter",
        ":inputs",
        ":run_config",
        ":training",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "run_config",
    srcs = ["run_config.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "run_config_test",
    size = "small",
    srcs = ["run_config_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":run_config",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "baseline",
    srcs = ["canned/baseline.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":head",
        ":head_utils",
        ":model_fn",
        ":optimizers",
        ":regression_head",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "baseline_test",
    size = "medium",
    srcs = ["canned/baseline_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "noasan",  # test flakily times out in asan mode.
        "notsan",  # b/67510291
        "optonly",  # flakily times out in fastbuild
    ],
    deps = [
        ":baseline",
        ":estimator",
        ":export_export",
        ":metric_keys",
        ":numpy_io",
        ":pandas_io",
        ":run_config",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_pandas_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "baseline_test_v1",
    size = "medium",
    srcs = ["canned/v1/baseline_test_v1.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "noasan",  # test flakily times out in asan mode.
        "notsan",  # b/67510291
        "optonly",  # flakily times out in fastbuild
    ],
    deps = [
        ":baseline",
        ":estimator",
        ":export_export",
        ":metric_keys",
        ":numpy_io",
        ":pandas_io",
        ":run_config",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_pandas_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "baseline_estimator_test",
    size = "medium",
    srcs = ["canned/baseline_estimator_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "noasan",  # test flakily times out in asan mode.
        "notsan",  # b/67510291
        "optonly",  # flakily times out in fastbuild
    ],
    deps = [
        ":baseline",
        ":estimator",
        ":export_export",
        ":metric_keys",
        ":numpy_io",
        ":regression_head",
        ":run_config",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "baseline_estimator_test_v1",
    size = "medium",
    srcs = ["canned/v1/baseline_estimator_test_v1.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "noasan",  # test flakily times out in asan mode.
        "notsan",  # b/67510291
        "optonly",  # flakily times out in fastbuild
    ],
    deps = [
        ":baseline",
        ":estimator",
        ":export_export",
        ":metric_keys",
        ":numpy_io",
        ":run_config",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "boosted_trees",
    srcs = ["canned/boosted_trees.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":boosted_trees_utils",
        ":estimator",
        ":head",
        ":mode_keys",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "boosted_trees_test",
    size = "medium",
    srcs = ["canned/boosted_trees_test.py"],
    python_version = "PY3",
    shard_count = 32,
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "no_oss",
        "notap",
        "optonly",
    ],
    deps = [
        ":boosted_trees",
        ":inputs",
        ":mode_keys",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//third_party/tensorflow/python:framework_test_lib",
        "//third_party/tensorflow/python:parsing_ops",
    ],
)

py_library(
    name = "boosted_trees_utils",
    srcs = ["canned/boosted_trees_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":head",
        ":model_fn",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "boosted_trees_utils_test",
    size = "medium",
    srcs = ["canned/boosted_trees_utils_test.py"],
    python_version = "PY3",
    shard_count = 2,
    srcs_version = "PY2AND3",
    tags = [
        "optonly",
    ],
    deps = [
        ":boosted_trees",
        ":inputs",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "kmeans",
    srcs = ["canned/kmeans.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":head",
        ":model_fn",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "kmeans_test",
    size = "medium",
    srcs = ["canned/kmeans_test.py"],
    python_version = "PY3",
    shard_count = 8,
    srcs_version = "PY2AND3",
    deps = [
        ":inputs",
        ":kmeans",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_library(
    name = "dnn",
    srcs = ["canned/dnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":head",
        ":head_utils",
        ":mode_keys",
        ":optimizers",
        ":regression_head",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "dnn_testing_utils",
    srcs = ["canned/dnn_testing_utils.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":estimator",
        ":head",
        ":metric_keys",
        ":mode_keys",
        ":model_fn",
        ":numpy_io",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "dnn_testing_utils_v1",
    srcs = ["canned/v1/dnn_testing_utils_v1.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":head",
        ":metric_keys",
        ":mode_keys",
        ":model_fn",
        ":numpy_io",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "dnn_test_fc_v1_v1",
    size = "medium",
    srcs = ["canned/v1/dnn_test_fc_v1_v1.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",  # b/67510291
    ],
    deps = [
        ":dnn",
        ":dnn_testing_utils_v1",
        ":export_export",
        ":numpy_io",
        ":pandas_io",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_pandas_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "dnn_test_fc_v2",
    size = "medium",
    srcs = ["canned/dnn_test_fc_v2.py"],
    python_version = "PY3",
    shard_count = 8,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",  # b/67510291
    ],
    deps = [
        ":dnn",
        ":dnn_testing_utils",
        ":export_export",
        ":numpy_io",
        ":pandas_io",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_pandas_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "dnn_test_fc_v2_v1",
    size = "medium",
    srcs = ["canned/v1/dnn_test_fc_v2_v1.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",  # b/67510291
    ],
    deps = [
        ":dnn",
        ":dnn_testing_utils_v1",
        ":export_export",
        ":numpy_io",
        ":pandas_io",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_pandas_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "dnn_estimator_test",
    size = "medium",
    srcs = ["canned/dnn_estimator_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
        "optonly",  # times out http://b/79220679
    ],
    deps = [
        ":dnn",
        ":dnn_testing_utils",
        ":export_export",
        ":multi_class_head",
        ":numpy_io",
        ":prediction_keys",
        ":regression_head",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "dnn_estimator_test_v1",
    size = "medium",
    srcs = ["canned/v1/dnn_estimator_test_v1.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
        "optonly",  # times out http://b/79220679
    ],
    deps = [
        ":dnn",
        ":dnn_testing_utils_v1",
        ":export_export",
        ":head",
        ":numpy_io",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "dnn_linear_combined",
    srcs = ["canned/dnn_linear_combined.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dnn",
        ":estimator",
        ":head",
        ":head_utils",
        ":linear",
        ":model_fn",
        ":optimizers",
        ":regression_head",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "dnn_linear_combined_estimator_test",
    size = "medium",
    srcs = ["canned/dnn_linear_combined_estimator_test.py"],
    python_version = "PY3",
    shard_count = 3,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":dnn_linear_combined",
        ":dnn_testing_utils",
        ":export_export",
        ":linear_testing_utils",
        ":numpy_io",
        ":prediction_keys",
        ":regression_head",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "dnn_linear_combined_estimator_test_v1",
    size = "medium",
    srcs = ["canned/v1/dnn_linear_combined_estimator_test_v1.py"],
    python_version = "PY3",
    shard_count = 3,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":dnn_linear_combined",
        ":dnn_testing_utils_v1",
        ":export_export",
        ":head",
        ":linear_testing_utils_v1",
        ":numpy_io",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "dnn_linear_combined_test",
    size = "medium",
    srcs = ["canned/dnn_linear_combined_test.py"],
    python_version = "PY3",
    shard_count = 32,
    srcs_version = "PY2AND3",
    tags = [
        "no_oss",  # TODO(b/143323557)
        "no_pip",
        "notsan",  # TODO(b/67510291)
    ],
    deps = [
        ":dnn_linear_combined",
        ":dnn_testing_utils",
        ":export_export",
        ":linear_testing_utils",
        ":numpy_io",
        ":pandas_io",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_pandas_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "dnn_linear_combined_test_v1",
    size = "medium",
    srcs = ["canned/v1/dnn_linear_combined_test_v1.py"],
    python_version = "PY3",
    shard_count = 16,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",  # TODO(b/67510291)
    ],
    deps = [
        ":dnn_linear_combined",
        ":dnn_testing_utils_v1",
        ":export_export",
        ":linear_testing_utils_v1",
        ":numpy_io",
        ":pandas_io",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_pandas_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "checkpoint_converter",
    srcs = ["tools/checkpoint_converter.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "analytics_tools",
    srcs = ["tools/analytics.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "checkpoint_converter_test",
    srcs = ["tools/checkpoint_converter_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":checkpoint_converter",
        ":dnn",
        ":dnn_linear_combined",
        ":head",
        ":linear",
        ":numpy_io",
        ":regression_head",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_library(
    name = "util",
    srcs = [
        "util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = ["notsan"],  # b/67510291
    deps = [
        ":util",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "early_stopping",
    srcs = [
        "early_stopping.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":export_export",
        ":model_fn",
        ":run_config",
        ":util",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "early_stopping_test",
    srcs = [
        "early_stopping_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "notap",  # TODO(b/134928532): Reenable this test.
    ],
    deps = [
        ":early_stopping",
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
    ],
)

py_library(
    name = "estimator",
    srcs = [
        "estimator.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":export",
        ":mode_keys",
        ":model_fn",
        ":run_config",
        ":util",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "estimator_test",
    srcs = ["estimator_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = ["notsan"],  # b/67510291
    deps = [
        ":estimator",
        ":export",
        ":mode_keys",
        ":model_fn",
        ":numpy_io",
        ":run_config",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "parsing_utils",
    srcs = [
        "canned/parsing_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "parsing_utils_test",
    srcs = ["canned/parsing_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":parsing_utils",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "export_output",
    srcs = ["export/export_output.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_library(
    name = "export",
    srcs = [
        "export/export_lib.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":export_export",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_library(
    name = "export_export",
    srcs = [
        "export/export.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":util",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "export_test",
    size = "small",
    srcs = ["export/export_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":export_export",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "function",
    srcs = [
        "export/function.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":mode_keys",
        ":model_fn",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "function_test",
    size = "small",
    srcs = ["export/function_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":export",
        ":function",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_library(
    name = "head",
    srcs = ["canned/head.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":export_output",
        ":metric_keys",
        ":model_fn",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "head_test",
    size = "medium",
    srcs = ["canned/head_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "no_pip",
        "notap",  # b/148804861
    ],
    deps = [
        ":dnn_testing_utils_v1",
        ":head",
        ":metric_keys",
        ":mode_keys",
        ":model_fn",
        ":numpy_io",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "head_utils",
    srcs = ["head/head_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":binary_class_head",
        ":multi_class_head",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "base_head",
    srcs = ["head/base_head.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":export_output",
        ":head",
        ":metric_keys",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "base_head_test",
    size = "small",
    srcs = ["head/base_head_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":base_head_test_lib"],
)

py_library(
    name = "base_head_test_lib",
    testonly = True,
    srcs = ["head/base_head_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_head",
        ":binary_class_head",
        ":head_utils",
        ":mode_keys",
        ":model_fn",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "binary_class_head",
    srcs = ["head/binary_class_head.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_head",
        ":export_output",
        ":metric_keys",
        ":mode_keys",
        ":model_fn",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "binary_class_head_test",
    size = "medium",
    srcs = ["head/binary_class_head_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "no_pip",
        "notap",  # b/148804861
    ],
    deps = [
        ":binary_class_head",
        ":dnn",
        ":dnn_testing_utils",
        ":head_utils",
        ":metric_keys",
        ":mode_keys",
        ":model_fn",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "multi_head",
    srcs = ["head/multi_head.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_head",
        ":export_output",
        ":metric_keys",
        ":mode_keys",
        ":model_fn",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "multi_head_test",
    size = "medium",
    srcs = ["head/multi_head_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":head_utils",
        ":metric_keys",
        ":mode_keys",
        ":multi_head",
        ":multi_label_head",
        ":prediction_keys",
        ":regression_head",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "multi_class_head",
    srcs = ["head/multi_class_head.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_head",
        ":export_output",
        ":metric_keys",
        ":mode_keys",
        ":model_fn",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "multi_class_head_test",
    size = "medium",
    srcs = ["head/multi_class_head_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":dnn",
        ":head_utils",
        ":metric_keys",
        ":mode_keys",
        ":model_fn",
        ":multi_class_head",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "multi_label_head",
    srcs = ["head/multi_label_head.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_head",
        ":export_output",
        ":metric_keys",
        ":mode_keys",
        ":model_fn",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "multi_label_head_test",
    size = "medium",
    srcs = ["head/multi_label_head_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":dnn",
        ":head_utils",
        ":metric_keys",
        ":mode_keys",
        ":model_fn",
        ":multi_label_head",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "regression_head",
    srcs = ["head/regression_head.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_head",
        ":export_output",
        ":metric_keys",
        ":mode_keys",
        ":model_fn",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "regression_head_test",
    size = "medium",
    srcs = ["head/regression_head_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "notap",  # b/148804861
    ],
    deps = [
        ":head_utils",
        ":metric_keys",
        ":mode_keys",
        ":model_fn",
        ":numpy_io",
        ":prediction_keys",
        ":regression_head",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "sequential_head",
    srcs = ["head/sequential_head.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_head",
        ":mode_keys",
        ":multi_head",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "sequential_head_test",
    size = "medium",
    srcs = ["head/sequential_head_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":binary_class_head",
        ":head_utils",
        ":metric_keys",
        ":mode_keys",
        ":model_fn",
        ":multi_class_head",
        ":multi_head",
        ":prediction_keys",
        ":sequential_head",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_library(
    name = "inputs",
    srcs = ["inputs/inputs.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":numpy_io",
        ":pandas_io",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "linear",
    srcs = ["canned/linear.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":binary_class_head",
        ":estimator",
        ":head",
        ":head_utils",
        ":optimizers",
        ":regression_head",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
        "//tensorflow_estimator/python/estimator/canned/linear_optimizer:sdca_ops_py",
    ],
)

py_library(
    name = "linear_testing_utils",
    srcs = ["canned/linear_testing_utils.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":estimator",
        ":export_export",
        ":linear",
        ":metric_keys",
        ":numpy_io",
        ":pandas_io",
        ":run_config",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_pandas_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "linear_testing_utils_v1",
    srcs = ["canned/v1/linear_testing_utils_v1.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":export_export",
        ":linear",
        ":metric_keys",
        ":numpy_io",
        ":pandas_io",
        ":run_config",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_pandas_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "linear_estimator_test",
    size = "medium",
    srcs = ["canned/linear_estimator_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":export_export",
        ":linear",
        ":linear_testing_utils",
        ":numpy_io",
        ":prediction_keys",
        ":regression_head",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "linear_estimator_test_v1",
    size = "medium",
    srcs = ["canned/v1/linear_estimator_test_v1.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        ":export_export",
        ":head",
        ":linear",
        ":linear_testing_utils_v1",
        ":numpy_io",
        ":prediction_keys",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "linear_test",
    size = "medium",
    srcs = ["canned/linear_test.py"],
    python_version = "PY3",
    shard_count = 8,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",  # b/67510291
    ],
    deps = [
        ":linear",
        ":linear_testing_utils",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "linear_test_v1",
    size = "medium",
    srcs = ["canned/v1/linear_test_v1.py"],
    python_version = "PY3",
    shard_count = 8,
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",  # b/67510291
    ],
    deps = [
        ":linear",
        ":linear_testing_utils_v1",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "metric_keys",
    srcs = ["canned/metric_keys.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model_fn",
    ],
)

py_library(
    name = "numpy_io",
    srcs = ["inputs/numpy_io.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inputs_queues",
    ],
)

py_test(
    name = "numpy_io_test",
    size = "small",
    srcs = ["inputs/numpy_io_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":numpy_io",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "optimizers",
    srcs = ["canned/optimizers.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "optimizers_test",
    size = "small",
    srcs = ["canned/optimizers_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":optimizers",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "optimizers_test_v2",
    size = "small",
    srcs = ["canned/optimizers_test_v2.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":optimizers",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "object_checkpointing_test",
    size = "medium",
    srcs = ["object_checkpointing_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":model_fn",
        ":optimizers",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "pandas_io",
    srcs = ["inputs/pandas_io.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inputs_queues",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_pandas_installed",
    ],
)

py_test(
    name = "pandas_io_test",
    size = "small",
    srcs = ["inputs/pandas_io_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":pandas_io",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_pandas_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "prediction_keys",
    srcs = ["canned/prediction_keys.py"],
    srcs_version = "PY2AND3",
    deps = [],
)

py_library(
    name = "inputs_queues",
    srcs = [
        "inputs/queues/__init__.py",
        "inputs/queues/feeding_functions.py",
        "inputs/queues/feeding_queue_runner.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_pandas_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "feeding_functions_test",
    size = "small",
    srcs = [
        "inputs/queues/feeding_functions_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":inputs_queues",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_pandas_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "feeding_queue_runner_test",
    size = "small",
    srcs = ["inputs/queues/feeding_queue_runner_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":inputs_queues",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_pandas_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "keras",
    srcs = ["keras.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":export",
        ":mode_keys",
        ":model_fn",
        ":run_config",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "keras_test",
    size = "medium",
    srcs = ["keras_test.py"],
    python_version = "PY3",
    shard_count = 8,
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
        "notsan",  # b/67510291
    ],
    deps = [
        ":export",
        ":keras",
        ":mode_keys",
        ":numpy_io",
        ":run_config",
        "//tensorflow_estimator/python/estimator:expect_h5py_installed",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "keras_premade_model_test",
    size = "medium",
    srcs = ["keras_premade_model_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":export",
        ":keras",
        ":mode_keys",
        ":numpy_io",
        ":run_config",
        "//tensorflow_estimator/python/estimator:expect_h5py_installed",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "keras_distribute_strategy_test",
    srcs = ["keras_distribute_strategy_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":keras",
        ":run_config",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_library(
    name = "saved_model_estimator",
    srcs = ["canned/saved_model_estimator.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":estimator",
        ":export",
        ":mode_keys",
        ":model_fn",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "saved_model_estimator_test",
    size = "medium",
    srcs = ["canned/saved_model_estimator_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "notsan",
    ],
    deps = [
        ":estimator",
        ":export",
        ":mode_keys",
        ":model_fn",
        ":saved_model_estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "basic_session_run_hooks",
    srcs = ["hooks/basic_session_run_hooks.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "basic_session_run_hooks_test",
    size = "medium",
    srcs = ["hooks/basic_session_run_hooks_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":estimator_py",
        ":fake_summary_writer",
    ],
)

py_library(
    name = "session_run_hook",
    srcs = ["hooks/session_run_hook.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "fake_summary_writer",
    srcs = [
        "hooks/fake_summary_writer.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "rnn",
    srcs = ["canned/rnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":binary_class_head",
        ":estimator",
        ":multi_class_head",
        ":optimizers",
        ":sequential_head",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_test(
    name = "rnn_test",
    size = "medium",
    srcs = ["canned/rnn_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_oss",  # b/140934549
        "no_pip",
        "noasan",  # times out
        "notsan",
        "optonly",  # times out http://b/79220679
    ],
    deps = [
        ":export",
        ":head",
        ":metric_keys",
        ":multi_class_head",
        ":numpy_io",
        ":parsing_utils",
        ":prediction_keys",
        ":rnn",
        ":sequential_head",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_keras_installed",
    ],
)

py_library(
    name = "expect_absl_installed",
    # This is a dummy rule used as a absl dependency in open-source.
    # We expect absl to already be installed on the system, e.g. via
    # `pip install absl`
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_numpy_installed",
    # This is a dummy rule used as a numpy dependency in open-source.
    # We expect numpy to already be installed on the system, e.g. via
    # `pip install numpy`
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_pandas_installed",
    # This is a dummy rule used as a numpy dependency in open-source.
    # We expect pandas to already be installed on the system, e.g. via
    # `pip install pandas`
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_h5py_installed",
    # This is a dummy rule used as a numpy dependency in open-source.
    # We expect h5py to already be installed on the system, e.g. via
    # `pip install h5py'
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_six_installed",
    # This is a dummy rule used as a numpy dependency in open-source.
    # We expect six to already be installed on the system, e.g. via
    # `pip install six`
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_tensorboard_installed",
    # This is a dummy rule used as a tensorboard dependency in open-source.
    # We expect tensorboard to already be installed on the system, e.g. via
    # `pip install tensorboard`.
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_tensorflow_installed",
    # This is a dummy rule used as a numpy dependency in open-source.
    # We expect tensorflow to already be installed on the system, e.g. via
    # `pip install tensorflow` or `pip install tensorflow_gpu`
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_tensorflow_keras_installed",
    # This is a dummy rule used as a numpy dependency in open-source.
    # We expect tensorflow to already be installed on the system, e.g. via
    # `pip install tensorflow` or `pip install tensorflow_gpu`
    visibility = ["//visibility:public"],
)

# The following targets are emulating cuda_py_test from //third_party/tensorflow:tensorflow.google.bzl
# cuda_py_test cannot be used directly because the bzl file cannot be imported into tensorflow_estimator

py_test(
    name = "distribute_strategy_estimator_integration_test",
    size = "medium",
    srcs = ["distribute_strategy_estimator_integration_test.py"],
    main = "distribute_strategy_estimator_integration_test.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "tf_integration_test",
    ],
    deps = [
        ":estimator_py",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "distribute_strategy_estimator_integration_test_gpu",
    size = "medium",
    srcs = ["distribute_strategy_estimator_integration_test.py"],
    main = "distribute_strategy_estimator_integration_test.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "cuda",
        "gpu",
        "multi_and_single_gpu",
        "requires-gpu-sm35",
        "tf_integration_test",
    ],
    deps = [
        ":estimator_py",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "distribute_strategy_estimator_training_test",
    size = "medium",
    srcs = ["distribute_strategy_estimator_training_test.py"],
    main = "distribute_strategy_estimator_training_test.py",
    python_version = "PY3",
    shard_count = 48,
    srcs_version = "PY2AND3",
    tags = [
        "no_oss",  # b/140933379
        # TODO(b/118768923): Re-enable {a,m,t}san test.
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        ":estimator_py",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "distribute_strategy_estimator_training_test_gpu",
    size = "medium",
    srcs = ["distribute_strategy_estimator_training_test.py"],
    main = "distribute_strategy_estimator_training_test.py",
    python_version = "PY3",
    shard_count = 48,
    srcs_version = "PY2AND3",
    tags = [
        # TODO(b/118768923): Re-enable {a,m,t}san test.
        "noasan",
        "nomsan",
        "notsan",
        "cuda",
        "requires-gpu-sm35",
        "gpu",
        "multi_and_single_gpu",
    ],
    deps = [
        ":estimator_py",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "canned_estimator_ds_integration_test",
    size = "medium",
    srcs = ["canned/canned_estimator_ds_integration_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":estimator_py",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "canned_estimator_ds_integration_test_gpu",
    size = "medium",
    srcs = ["canned/canned_estimator_ds_integration_test.py"],
    main = "canned/canned_estimator_ds_integration_test.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "cuda",
        "gpu",
        "multi_and_single_gpu",
        "requires-gpu-sm35",
        "tf_integration_test",
    ],
    deps = [
        ":estimator_py",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_numpy_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)
