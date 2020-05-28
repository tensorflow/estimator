package(default_visibility = ["//tensorflow_estimator:internal"])

# Description:
#  Tools for building the TensorFlow pip package.

COMMON_PIP_DEPS = [
    "//tensorflow_estimator",
    # Need to include testing libraries in pip package so our pip
    # release tests can run. (see py_test rule in estimator.bzl for more context).
    # Essentially, everything needed to run the test (except the test file itself)
    # must be contained in the pip package since we strip away all deps.
    "//tensorflow_estimator/python/estimator:dnn_testing_utils",
    "//tensorflow_estimator/python/estimator:dnn_testing_utils_v1",
    "//tensorflow_estimator/python/estimator:linear_testing_utils",
    "//tensorflow_estimator/python/estimator:linear_testing_utils_v1",
]

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = COMMON_PIP_DEPS,
)
