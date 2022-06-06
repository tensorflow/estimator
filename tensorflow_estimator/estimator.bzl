"""Estimator common skylark macros."""

# Macro to run Estimator py_tests against pip installation.
def py_test(deps = [], **kwargs):
    native.py_test(
        deps = select({
            "//conditions:default": deps,
            "//tensorflow_estimator:no_estimator_py_deps": [],
        }),
        **kwargs
    )

def tpu_py_test(**kwargs):
    # Skip the tpu test for Estimator oss.
    pass

# We are never indexing generated code in the OSS build, but still
# return a select() for consistency.
def if_indexing_source_code(
        if_true,  # @unused
        if_false):
    """Return a select() on whether or not we are building for source code indexing."""
    return select({
        "//conditions:default": if_false,
    })
