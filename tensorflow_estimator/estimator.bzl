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
