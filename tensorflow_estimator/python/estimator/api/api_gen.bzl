"""Targets for generating TensorFlow Python API __init__.py files.

This bzl file is copied with slight modifications from
tensorflow/python/estimator/api/api_gen.bzl
so that we can avoid needing to depend on TF source code in Bazel build.

It should be noted that because this file is executed during the build,
and it imports TensorFlow code, that installing TensorFlow python package
is required to Bazel build Estimator.
"""

# keep sorted
ESTIMATOR_API_INIT_FILES = [
    # BEGIN GENERATED ESTIMATOR FILES
    "__init__.py",
    "estimator/__init__.py",
    "estimator/export/__init__.py",
    "estimator/inputs/__init__.py",
    # END GENERATED ESTIMATOR FILES
]

# Creates a genrule that generates a directory structure with __init__.py
# files that import all exported modules (i.e. modules with tf_export
# decorators).
#
# Args:
#   name: name of genrule to create.
#   output_files: List of __init__.py files that should be generated.
#     This list should include file name for every module exported using
#     tf_export. For e.g. if an op is decorated with
#     @tf_export('module1.module2', 'module3'). Then, output_files should
#     include module1/module2/__init__.py and module3/__init__.py.
#   root_init_template: Python init file that should be used as template for
#     root __init__.py file. "# API IMPORTS PLACEHOLDER" comment inside this
#     template will be replaced with root imports collected by this genrule.
#   srcs: genrule sources. If passing root_init_template, the template file
#     must be included in sources.
#   api_name: Name of the project that you want to generate API files for
#     (e.g. "tensorflow" or "estimator").
#   package: Python package containing the @tf_export decorators you want to
#     process
#   package_dep: Python library target containing your package.

def gen_api_init_files(
        name,
        output_files = ESTIMATOR_API_INIT_FILES,
        output_package = "tensorflow_estimator.python.estimator.api",
        package = "tensorflow_estimator.python.estimator",
        package_dep = "//tensorflow_estimator/python/estimator:estimator_py",
        srcs = [],
        api_name = "estimator"):
    api_gen_binary_target = "create_estimator_api"
    native.py_binary(
        name = api_gen_binary_target,
        srcs = ["//tensorflow_estimator/python/estimator/api:create_python_api_wrapper.py"],
        main = "//tensorflow_estimator/python/estimator/api:create_python_api_wrapper.py",
        srcs_version = "PY2AND3",
        visibility = ["//visibility:public"],
        deps = [
            package_dep,
        ],
    )

    native.genrule(
        name = name,
        outs = output_files,
        cmd = (
            "$(location :" + api_gen_binary_target + ") " +
            " --apidir=$(@D) --apiname=" + api_name +
            " --package=" + package +
            " --output_package=" + output_package + " $(OUTS)"
        ),
        srcs = srcs,
        tools = [":" + api_gen_binary_target],
        visibility = ["//visibility:public"],
    )
