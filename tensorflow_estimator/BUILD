# Description: Tensorflow Estimator.
load(
    "//tensorflow_estimator/python/estimator/api:api_gen.bzl",
    "ESTIMATOR_API_INIT_FILES_V1",
    "ESTIMATOR_API_INIT_FILES_V2",
    "gen_api_init_files",
)

licenses(["notice"])

package(default_visibility = ["//tensorflow_estimator:internal"])

exports_files(["LICENSE"])

# TODO(mikecase): Clean up. Remove all non estimator packages.
package_group(
    name = "internal",
    packages = [
        "//learning/brain/...",
        "//learning/deepmind/research/...",
        "//learning/tfx/models/uplift/estimators/...",
        "//nlp/nlx/i18n/pangloss/...",
        "//tensorflow_estimator/...",
        "//third_party/py/tensorflow_privacy/...",
        "//third_party/tensorflow/python/estimator/...",
    ],
)

# This flag specifies whether Estimator 2.0 API should be built instead
# of 1.* API. Note that Estimator 2.0 API is currently under development.
config_setting(
    name = "api_version_2",
    define_values = {"estimator_api_version": "2"},
)

config_setting(
    name = "no_estimator_py_deps",
    define_values = {"no_estimator_py_deps": "true"},
    visibility = ["//visibility:public"],
)

py_library(
    name = "tensorflow_estimator",
    srcs = [
        ":root_init_gen",
        ":estimator_python_api_gen_compat_v1",
        ":estimator_python_api_gen_compat_v2",
        # Old API files. Delete once TensorFlow is updated to import from new location.
        "//tensorflow_estimator/python/estimator/api:estimator_python_api_gen",
        "//tensorflow_estimator/python/estimator/api:estimator_python_api_gen_compat_v1",
        "//tensorflow_estimator/python/estimator/api:estimator_python_api_gen_compat_v2",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_estimator/python/estimator:estimator_py",
    ],
)

genrule(
    name = "root_init_gen",
    srcs = select({
        "api_version_2": [":estimator_python_api_gen_compat_v2"],
        "//conditions:default": [":estimator_python_api_gen_compat_v1"],
    }),
    outs = ["__init__.py"],
    cmd = select({
        "api_version_2": "cp $(@D)/_api/v2/v2.py $(OUTS)",
        "//conditions:default": "cp $(@D)/_api/v1/v1.py $(OUTS)",
    }),
)

gen_api_init_files(
    name = "estimator_python_api_gen_compat_v1",
    api_version = 1,
    output_dir = "_api/v1/",
    output_files = ESTIMATOR_API_INIT_FILES_V1,
    output_package = "tensorflow_estimator._api.v1",
    root_file_name = "v1.py",
)

gen_api_init_files(
    name = "estimator_python_api_gen_compat_v2",
    api_version = 2,
    output_dir = "_api/v2/",
    output_files = ESTIMATOR_API_INIT_FILES_V2,
    output_package = "tensorflow_estimator._api.v2",
    root_file_name = "v2.py",
)
