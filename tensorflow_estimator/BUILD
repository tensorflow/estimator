# Description: Tensorflow Estimator.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow_estimator:internal"])

exports_files(["LICENSE"])

load("//tensorflow_estimator/python/estimator/api:api_gen.bzl", "gen_api_init_files")
load("//tensorflow_estimator/python/estimator/api:api_gen.bzl", "ESTIMATOR_API_INIT_FILES_V1")
load("//tensorflow_estimator/python/estimator/api:api_gen.bzl", "ESTIMATOR_API_INIT_FILES_V2")

# TODO(mikecase): Clean up. Remove all non estimator packages.
package_group(
    name = "internal",
    packages = [
        "//learning/brain/...",
        "//tensorflow_estimator/...",
        "//third_party/tensorflow/contrib/...",
        "//third_party/tensorflow/python/estimator/...",
    ],
)

py_library(
    name = "tensorflow_estimator",
    srcs = [
        ":estimator_python_api_gen",
        ":estimator_python_api_gen_compat_v1",
        ":estimator_python_api_gen_compat_v2",
        # Old API files. Delete once TensorFlow is updated to import from new location.
        "//tensorflow_estimator/python/estimator/api:estimator_python_api_gen",
        "//tensorflow_estimator/python/estimator/api:estimator_python_api_gen_compat_v1",
        "//tensorflow_estimator/python/estimator/api:estimator_python_api_gen_compat_v2",
    ],
    deps = [
        "//tensorflow_estimator/contrib/estimator:estimator_py",
        "//tensorflow_estimator/python/estimator:estimator_py",
    ],
)

gen_api_init_files(
    name = "estimator_python_api_gen",
    api_version = 1,
    output_files = ESTIMATOR_API_INIT_FILES_V1,
    output_package = "tensorflow_estimator",
)

gen_api_init_files(
    name = "estimator_python_api_gen_compat_v1",
    api_version = 1,
    output_dir = "_api/v1/",
    output_files = ESTIMATOR_API_INIT_FILES_V1,
    output_package = "tensorflow_estimator._api.v1",
)

gen_api_init_files(
    name = "estimator_python_api_gen_compat_v2",
    api_version = 2,
    output_dir = "_api/v2/",
    output_files = ESTIMATOR_API_INIT_FILES_V2,
    output_package = "tensorflow_estimator._api.v2",
)
