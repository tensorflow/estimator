-----------------
| **`Documentation`** |
|-----------------|
| [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/python/tf/estimator) |

TensorFlow Estimator is a high-level TensorFlow API that greatly simplifies machine learning programming.
Estimators encapsulate training, evaluation, prediction, and exporting for your model.

## Getting Started

See our Estimator
[getting started guide](https://www.tensorflow.org/guide/estimator) for an
introduction to the Estimator APIs.

## Installation

`tf.estimator` is installed when you install the TensorFlow pip package. See
[Installing TensorFlow](https://www.tensorflow.org/install) for instructions.

## Developing

If you want to build TensorFlow Estimator locally, you will need to
[install Bazel](https://docs.bazel.build/versions/master/install.html) and
[install TensorFlow](https://www.tensorflow.org/install/pip).

```sh
# To build TensorFlow Estimator whl file.
bazel build //tensorflow_estimator/tools/pip_package:build_pip_package
bazel-bin/tensorflow_estimator/tools/pip_package/build_pip_package /tmp/estimator_pip

# To run all Estimator tests
bazel test //tensorflow_estimator/...
```

## Contribution guidelines

If you want to contribute to TensorFlow Estimator, be sure to review the [contribution
guidelines](CONTRIBUTING.md).

**Note that this repository is included as a component of the main TensorFlow
package, and any issues encountered while using Estimators should be filed under
[TensorFlow GitHub Issues](https://github.com/tensorflow/tensorflow/issues),
as we do not separately track issues in this repository. You can link this
repository in any issues created as necessary.**

Please see
[TensorFlow Discuss](https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss) for general questions
and discussion and please direct specific questions to
[Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow).

## License

[Apache License 2.0](LICENSE)
