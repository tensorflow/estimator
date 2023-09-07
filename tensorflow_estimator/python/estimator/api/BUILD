# Placeholder: load aliased py_binary
load("//tensorflow_estimator/python/estimator/api:api_gen.bzl", "ESTIMATOR_API_INIT_FILES_V1", "ESTIMATOR_API_INIT_FILES_V2", "generate_apis")

package(default_visibility = ["//tensorflow_estimator:internal"])

licenses(["notice"])

# This flag specifies whether Estimator 2.0 API should be built instead
# of 1.* API. Note that Estimator 2.0 API is currently under development.
config_setting(
    name = "api_version_2",
    define_values = {"estimator_api_version": "2"},
)

py_binary(
    name = "extractor_wrapper",
    srcs = ["extractor_wrapper.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_estimator/python/estimator:expect_absl_installed",  # absl:app
    ],
)

py_binary(
    name = "generator_wrapper",
    srcs = ["generator_wrapper.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_estimator/python/estimator:expect_absl_installed",  # absl:app
    ],
)

genrule(
    name = "estimator_python_api_gen",
    srcs = select({
        "api_version_2": ["_v2/v2.py"],
        "//conditions:default": ["_v1/v1.py"],
    }),
    outs = ["__init__.py"],
    cmd = select({
        "api_version_2": "cp $(location :_v2/v2.py) $(OUTS)",
        "//conditions:default": "cp $(location :_v1/v1.py) $(OUTS)",
    }),
)

generate_apis(
    name = "estimator_python_api_gen_compat_v1",
    api_version = 1,
    output_dir = "_v1/",
    output_files = ESTIMATOR_API_INIT_FILES_V1,
    output_package = "tensorflow_estimator.python.estimator.api._v1",
    root_file_name = "v1.py",
    visibility = ["//visibility:public"],
)

generate_apis(
    name = "estimator_python_api_gen_compat_v2",
    api_version = 2,
    output_dir = "_v2/",
    output_files = ESTIMATOR_API_INIT_FILES_V2,
    output_package = "tensorflow_estimator.python.estimator.api._v2",
    root_file_name = "v2.py",
    visibility = ["//visibility:public"],
)
