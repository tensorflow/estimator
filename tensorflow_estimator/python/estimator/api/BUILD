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

# This flag specifies whether Estimator 2.0 API should be built instead
# of 1.* API. Note that Estimator 2.0 API is currently under development.
config_setting(
    name = "api_version_2",
    define_values = {"estimator_api_version": "2"},
)

genrule(
    name = "estimator_python_api_gen",
    srcs = select({
        "api_version_2": [":estimator_python_api_gen_compat_v2"],
        "//conditions:default": [":estimator_python_api_gen_compat_v1"],
    }),
    outs = ["__init__.py"],
    cmd = select({
        "api_version_2": "cp $(@D)/_v2/v2.py $(OUTS)",
        "//conditions:default": "cp $(@D)/_v1/v1.py $(OUTS)",
    }),
    visibility = ["//visibility:public"],
)

gen_api_init_files(
    name = "estimator_python_api_gen_compat_v1",
    api_version = 1,
    output_dir = "_v1/",
    output_files = ESTIMATOR_API_INIT_FILES_V1,
    output_package = "tensorflow_estimator.python.estimator.api._v1",
    root_file_name = "v1.py",
)

gen_api_init_files(
    name = "estimator_python_api_gen_compat_v2",
    api_version = 2,
    output_dir = "_v2/",
    output_files = ESTIMATOR_API_INIT_FILES_V2,
    output_package = "tensorflow_estimator.python.estimator.api._v2",
    root_file_name = "v2.py",
)
