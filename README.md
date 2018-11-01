-----------------
| **`Documentation`** |
|-----------------|
| [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/python/tf/estimator) |

TensorFlow Estimator is a high-level TensorFlow API that greatly simplifies machine learning programming.
Estimators encapsulate training, evaluation, prediction, and exporting for your model.

## Getting Started

See our Estimator [getting started guide](https://www.tensorflow.org/guide/estimators) for an introduction to the Estimator APIs.

## Installation

`tf.Estimator` is installed when you install the TensorFlow pip package. See [Installing TensorFlow](https://www.tensorflow.org/get_started/os_setup.html) for instructions.

## Developing

If you want to build TensorFlow Estimator locally, you will need to [install Bazel](https://docs.bazel.build/versions/master/install.html) and [install TensorFlow]((https://www.tensorflow.org/get_started/os_setup.html)).

```sh
# To build TensorFlow Estimator whl file.
bazel build //tensorflow_estimator/tools/pip_package:build_pip_package
bazel-bin/tensorflow_estimator/tools/pip_package/build_pip_package /tmp/estimator_pip
pip install /tmp/estimator_pip/tensorflow_estimator-1.10.0-py2-none-any.whl

# To run all Estimator tests
bazel test //tensorflow_estimator/...
```

## Contribution guidelines

**If you want to contribute to TensorFlow Estimator, be sure to review the [contribution
guidelines](CONTRIBUTING.md).**

**We use [GitHub issues](https://github.com/tensorflow/estimator/issues) for
tracking requests and bugs. So please see
[TensorFlow Discuss](https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss) for general questions
and discussion, and please direct specific questions to [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow).**

The TensorFlow project strives to abide by generally accepted best practices in open-source software development:

## License

[Apache License 2.0](LICENSE)
