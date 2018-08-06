package(default_visibility = ["//tensorflow_estimator:internal"])

# Description:
#  Tools for building the TensorFlow pip package.

COMMON_PIP_DEPS = [
    "//tensorflow_estimator/contrib:estimator_py",
    "//tensorflow_estimator/python:estimator_py",
    "//tensorflow_estimator/python/api:estimator_python_api_gen",
]

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = COMMON_PIP_DEPS,
)
