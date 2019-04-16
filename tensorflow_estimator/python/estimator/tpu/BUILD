# Description: TPUEstimator

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_estimator:internal",
        "//third_party/tensorflow:__subpackages__",
    ],
)

py_library(
    name = "tpu_estimator",
    srcs = [
        "_tpu_estimator_embedding.py",
        "error_handling.py",
        "tpu_config.py",
        "tpu_context.py",
        "tpu_estimator.py",
        "util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:expect_six_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//tensorflow_estimator/python/estimator:export_output",
        "//tensorflow_estimator/python/estimator:model_fn",
        "//tensorflow_estimator/python/estimator:run_config",
    ],
)

py_test(
    name = "tpu_config_test",
    size = "small",
    srcs = ["tpu_config_test.py"],
    deps = [
        ":tpu_estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "tpu_estimator_signals_test",
    size = "small",
    srcs = ["tpu_estimator_signals_test.py"],
    # TODO(jhseu): Remove. Fails in OSS on Python 3.
    tags = ["no_oss"],
    deps = [
        ":tpu_estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)
