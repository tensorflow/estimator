package(default_visibility = ["//tensorflow_estimator:internal"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow_estimator/python/estimator/api:api_gen.bzl", "gen_api_init_files")
load("//tensorflow_estimator/python/estimator/api:api_gen.bzl", "ESTIMATOR_API_INIT_FILES_V1")
load("//tensorflow_estimator/python/estimator/api:api_gen.bzl", "ESTIMATOR_API_INIT_FILES_V2")

exports_files(
    [
        "create_python_api_wrapper.py",
    ],
)

gen_api_init_files(
    name = "estimator_python_api_gen",
    api_version = 1,
    output_files = ESTIMATOR_API_INIT_FILES_V1,
    output_package = "tensorflow_estimator.python.estimator.api",
)

gen_api_init_files(
    name = "estimator_python_api_gen_compat_v1",
    api_version = 1,
    output_dir = "_v1/",
    output_files = ESTIMATOR_API_INIT_FILES_V1,
    output_package = "tensorflow_estimator.python.estimator.api._v1",
)

gen_api_init_files(
    name = "estimator_python_api_gen_compat_v2",
    api_version = 2,
    output_dir = "_v2/",
    output_files = ESTIMATOR_API_INIT_FILES_V2,
    output_package = "tensorflow_estimator.python.estimator.api._v2",
)
