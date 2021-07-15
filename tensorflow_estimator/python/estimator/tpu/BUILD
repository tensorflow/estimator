# Description: TPUEstimator

# INTERNAL TEST RULE PLACEHOLDER
load("//tensorflow_estimator:estimator.bzl", "py_test", "tpu_py_test")

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_estimator:internal",
        "//third_party/tensorflow:__subpackages__",
    ],
    features = [
        "static_initialization_order_fiasco",  # TODO(b/191807649)
    ],
)

py_library(
    name = "tpu_estimator",
    srcs = [
        "_tpu_estimator_embedding.py",
        "error_handling.py",
        "iteration_count_estimator.py",
        "tpu_config.py",
        "tpu_context.py",
        "tpu_estimator.py",
        "util.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_estimator/python/estimator",
        "//tensorflow_estimator/python/estimator:analytics_tools",
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
    python_version = "PY3",
    deps = [
        ":tpu_estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "error_handling_test",
    size = "small",
    srcs = ["error_handling_test.py"],
    python_version = "PY3",
    deps = [
        ":tpu_estimator",
    ],
)

py_test(
    name = "tpu_estimator_signals_test",
    size = "small",
    srcs = ["tpu_estimator_signals_test.py"],
    python_version = "PY3",
    # TODO(jhseu): Remove. Fails in OSS on Python 3.
    tags = [
        "no_oss",
    ],
    deps = [
        ":tpu_estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

tpu_py_test(
    name = "tpu_estimator_test",
    size = "medium",
    timeout = "long",
    srcs = ["tpu_estimator_test.py"],
    args = [
        "--test_num_shards=2",
    ],
    disable_experimental = True,
    disable_tfrt = True,  # TFRT does not support estimator
    malloc = "//third_party/tcmalloc:tcmalloc_or_debug",
    shard_count = 2,
    srcs_version = "PY3",
    deps = [
        ":tpu_estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//third_party/py/absl/flags",
    ],
)

tpu_py_test(
    name = "tpu_estimator_embedding_test",
    size = "medium",
    timeout = "long",
    srcs = [
        "tpu_estimator_embedding_test.py",
    ],
    args = [
        "--test_num_shards=2",
    ],
    # TODO(b/140117863): Hanging, then timeout
    disable_experimental = True,
    disable_tfrt = True,  # TFRT does not support estimator
    malloc = "//third_party/tcmalloc:tcmalloc_or_debug",
    shard_count = 4,
    srcs_version = "PY3",
    deps = [
        ":tpu_estimator",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//third_party/py/absl/flags",
    ],
)

tpu_py_test(
    name = "tpu_estimator_evaluation_test",
    size = "medium",
    timeout = "long",
    srcs = ["tpu_estimator_evaluation_test.py"],
    args = [
        "--test_num_shards=2",
    ],
    disable_experimental = True,
    disable_tfrt = True,  # TFRT does not support estimator
    malloc = "//third_party/tcmalloc:tcmalloc_or_debug",
    shard_count = 2,
    srcs_version = "PY3",
    deps = [
        ":tpu_estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//third_party/py/absl/flags",
    ],
)

tpu_py_test(
    name = "tpu_estimator_export_test",
    size = "medium",
    srcs = ["tpu_estimator_export_test.py"],
    args = [
        "--test_num_shards=2",
    ],
    disable_experimental = True,
    disable_tfrt = True,  # TFRT does not support estimator
    malloc = "//third_party/tcmalloc:tcmalloc_or_debug",
    shard_count = 2,
    srcs_version = "PY3",
    deps = [
        ":tpu_estimator",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

tpu_py_test(
    name = "tpu_estimator_gradients_test",
    size = "medium",
    srcs = [
        "tpu_estimator_gradients_test.py",
    ],
    args = [
        "--test_num_shards=2",
        "--xla_jf_conv_full_precision=true",
    ],
    # TODO(b/140117863): Fatal error from hardware
    disable_experimental = True,
    disable_mlir_bridge = False,
    disable_tfrt = True,  # TFRT does not support estimator
    malloc = "//third_party/tcmalloc:tcmalloc_or_debug",
    shard_count = 2,
    srcs_version = "PY3",
    deps = [
        ":tpu_estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

tpu_py_test(
    name = "tpu_estimator_input_v2_test",
    size = "medium",
    srcs = ["tpu_estimator_input_v2_test.py"],
    disable_experimental = True,
    disable_tfrt = True,  # TFRT does not support estimator
    malloc = "//third_party/tcmalloc:tcmalloc_or_debug",
    srcs_version = "PY3",
    deps = [
        ":tpu_estimator",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

tpu_py_test(
    name = "tpu_estimator_integration_test",
    size = "medium",
    srcs = ["tpu_estimator_integration_test.py"],
    args = [
        "--test_num_shards=2",
    ],
    disable_experimental = True,
    disable_tfrt = True,  # TFRT does not support estimator
    malloc = "//third_party/tcmalloc:tcmalloc_or_debug",
    srcs_version = "PY3",
    deps = [
        ":tpu_estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

tpu_py_test(
    name = "tpu_estimator_model_parallelism_test",
    size = "medium",
    srcs = ["tpu_estimator_model_parallelism_test.py"],
    args = [
    ],
    disable_experimental = True,
    disable_tfrt = True,  # TFRT does not support estimator
    malloc = "//third_party/tcmalloc:tcmalloc_or_debug",
    srcs_version = "PY3",
    deps = [
        ":tpu_estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

py_test(
    name = "autotuning_iterations_per_loop_test",
    size = "small",
    srcs = ["autotuning_iterations_per_loop_test.py"],
    python_version = "PY3",
    deps = [
        ":tpu_estimator",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
    ],
)

tpu_py_test(
    name = "tpu_enqueue_sequence_test",
    size = "medium",
    srcs = ["tpu_enqueue_sequence_test.py"],
    disable_tfrt = True,  # TFRT does not support estimator
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_estimator",
        "//tensorflow_estimator/python/estimator:expect_absl_installed",
        "//tensorflow_estimator/python/estimator:expect_tensorflow_installed",
        "//third_party/tensorflow/contrib/summary",
    ],
)
