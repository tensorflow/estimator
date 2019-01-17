package(default_visibility = ["//tensorflow_estimator:__subpackages__"])

licenses(["notice"])  # Apache 2.0

# TODO(b/112597278): Investigate and remove stale BUILD tags.
# The no_oss tag was added as the estimator Kokoro test fails on certain
# core TF imports.

py_library(
    name = "feature_keys",
    srcs = [
        "feature_keys.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow_estimator/python/estimator:expect_tensorflow_installed"],
)

# py_library(
#     name = "saved_model_utils",
#     srcs = [
#         "saved_model_utils.py",
#     ],
#     srcs_version = "PY2AND3",
#     deps = [
#         ":feature_keys",
#         ":head",
#         ":model_utils",
#         "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
#         "//third_party/tensorflow/contrib/timeseries/python/timeseries:input_pipeline",
#     ],
# )

py_library(
    name = "model",
    srcs = [
        "model.py",
    ],
    srcs_version = "PY2AND3",
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
    srcs_version = "PY2AND3",
    deps = [
        ":ar_model",
        ":feature_keys",
        ":head",
        ":math_utils",
        ":state_management",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

# py_test(
#     name = "estimators_test",
#     timeout = "long",
#     srcs = [
#         "estimators_test.py",
#     ],
#     shard_count = 3,
#     srcs_version = "PY2AND3",
#     tags = [
#         "no_mac",
#         "no_oss",
#         "nomsan",  # Takes too long to run.
#         "notsan",  # b/67865658
#         "optonly",  # Takes too long to run without optimization.
#     ],
#     deps = [
#         ":ar_model",
#         ":estimators",
#         ":feature_keys",
#         ":saved_model_utils",
#         "//tensorflow_estimator/python/estimator:expect_numpy_installed",
#         "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
#         "//third_party/tensorflow/contrib/timeseries/python/timeseries:input_pipeline",
#     ],
# )

py_library(
    name = "head",
    srcs = [
        "head.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":feature_keys",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

# py_test(
#     name = "head_test",
#     size = "large",
#     srcs = [
#         "head_test.py",
#     ],
#     shard_count = 4,
#     srcs_version = "PY2AND3",
#     tags = ["no_oss"],
#     deps = [
#         ":estimators",
#         ":feature_keys",
#         ":head",
#         ":model",
#         ":state_management",
#         "//tensorflow_estimator/python/estimator:expect_absl_installed",
#         "//tensorflow_estimator/python/estimator:expect_numpy_installed",
#         "//tensorflow_estimator/python/estimator:expect_six_installed",
#         "//third_party/py/tensorflow",
#         "//third_party/tensorflow/contrib/timeseries/examples:lstm",
#         "//third_party/tensorflow/contrib/timeseries/python/timeseries:input_pipeline",
#     ],
# )

py_library(
    name = "model_utils",
    srcs = [
        "model_utils.py",
    ],
    srcs_version = "PY2AND3",
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
    srcs_version = "PY2AND3",
    deps = [
        ":feature_keys",
        ":math_utils",
        ":model",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

# py_library(
#     name = "test_utils",
#     srcs = [
#         "test_utils.py",
#     ],
#     srcs_version = "PY2AND3",
#     tags = [
#         "no_oss",
#         "no_pip",
#     ],
#     deps = [
#         ":estimators",
#         ":feature_keys",
#         ":state_management",
#         "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
#         "//third_party/tensorflow/contrib/timeseries/python/timeseries:input_pipeline",
#     ],
# )

py_library(
    name = "ar_model",
    srcs = [
        "ar_model.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":feature_keys",
        ":model",
        ":model_utils",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

# py_test(
#     name = "ar_model_test",
#     timeout = "long",  # Moderate but for asan
#     srcs = [
#         "ar_model_test.py",
#     ],
#     srcs_version = "PY2AND3",
#     tags = [
#         "no_oss",
#         "notsan",
#     ],
#     deps = [
#         ":ar_model",
#         ":estimators",
#         ":feature_keys",
#         ":test_utils",
#         "//tensorflow_estimator/python/estimator:expect_numpy_installed",
#         "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
#         "//third_party/tensorflow/contrib/timeseries/python/timeseries:input_pipeline",
#     ],
# )

py_library(
    name = "math_utils",
    srcs = [
        "math_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":feature_keys",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

# py_test(
#     name = "math_utils_test",
#     srcs = [
#         "math_utils_test.py",
#     ],
#     srcs_version = "PY2AND3",
#     tags = [
#         "no_oss",
#         "no_windows",  # TODO: needs investigation on Windows
#     ],
#     deps = [
#         ":feature_keys",
#         ":math_utils",
#         "//tensorflow_estimator/python/estimator:expect_numpy_installed",
#         "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
#         "//third_party/tensorflow/contrib/timeseries/python/timeseries:input_pipeline",
#     ],
# )
