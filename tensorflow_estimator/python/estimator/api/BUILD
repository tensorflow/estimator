package(default_visibility = ["//tensorflow_estimator:internal"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow_estimator/python/estimator/api:api_gen.bzl", "gen_api_init_files")
load("//tensorflow_estimator/python/estimator/api:api_gen.bzl", "ESTIMATOR_API_INIT_FILES")

gen_api_init_files(
    name = "estimator_python_api_gen",
    api_name = "estimator",
    output_files = ESTIMATOR_API_INIT_FILES,
    output_package = "tensorflow_estimator.python.estimator.api",
    package = "tensorflow_estimator.python.estimator",
    package_dep = "//tensorflow_estimator/python/estimator:estimator_py",
)
