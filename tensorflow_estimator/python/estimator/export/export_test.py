# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for export."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from google.protobuf import text_format

from tensorflow.core.example import example_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow_estimator.python.estimator.export import export


class LabeledTensorMock(object):
  """Mock class emulating LabeledTensor."""

  def __init__(self):
    self.tensor = tf.constant([1])


def _convert_labeled_tensor_mock_to_tensor(value, *args, **kwargs):
  return ops.internal_convert_to_tensor(value.tensor, *args, **kwargs)


tf.register_tensor_conversion_function(LabeledTensorMock,
                                       _convert_labeled_tensor_mock_to_tensor)


class ServingInputReceiverTest(tf.test.TestCase):

  def test_serving_input_receiver_constructor(self):
    """Tests that no errors are raised when input is expected."""
    features = {
        "feature0": tf.constant([0]),
        u"feature1": tf.constant([1]),
        "feature2": tf.sparse.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 1]),
        # ints are allowed only in the `features` dict
        42: tf.constant([3]),
    }
    receiver_tensors = {
        "example0": tf.constant(["test0"], name="example0"),
        u"example1": tf.constant(["test1"], name="example1"),
    }
    export.ServingInputReceiver(features, receiver_tensors)

  def test_serving_input_receiver_features_invalid(self):
    receiver_tensors = {
        "example0": tf.constant(["test0"], name="example0"),
        u"example1": tf.constant(["test1"], name="example1"),
    }

    with self.assertRaisesRegexp(ValueError, "features must be defined"):
      export.ServingInputReceiver(
          features=None, receiver_tensors=receiver_tensors)

    with self.assertRaisesRegexp(ValueError,
                                 "feature keys must be strings or ints"):
      export.ServingInputReceiver(
          features={42.2: tf.constant([1])}, receiver_tensors=receiver_tensors)

    with self.assertRaisesRegexp(
        ValueError, "feature feature1 must be a Tensor, SparseTensor, or "
        "RaggedTensor."):
      export.ServingInputReceiver(
          features={"feature1": [1]}, receiver_tensors=receiver_tensors)

  def test_serving_input_receiver_receiver_tensors_invalid(self):
    features = {
        "feature0": tf.constant([0]),
        u"feature1": tf.constant([1]),
        "feature2": tf.sparse.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 1]),
    }

    with self.assertRaisesRegexp(ValueError,
                                 "receiver_tensors must be defined"):
      export.ServingInputReceiver(features=features, receiver_tensors=None)

    with self.assertRaisesRegexp(ValueError,
                                 "receiver_tensor keys must be strings"):
      export.ServingInputReceiver(
          features=features,
          receiver_tensors={1: tf.constant(["test"], name="example0")})

    with self.assertRaisesRegexp(ValueError,
                                 "receiver_tensor example1 must be a Tensor"):
      export.ServingInputReceiver(
          features=features, receiver_tensors={"example1": [1]})

  def test_single_feature_single_receiver(self):
    feature = tf.constant(5)
    receiver_tensor = tf.constant(["test"])
    input_receiver = export.ServingInputReceiver(feature, receiver_tensor)
    # single feature is automatically named
    feature_key, = input_receiver.features.keys()
    self.assertEqual("feature", feature_key)
    # single receiver is automatically named
    receiver_key, = input_receiver.receiver_tensors.keys()
    self.assertEqual("input", receiver_key)

  def test_multi_feature_single_receiver(self):
    features = {"foo": tf.constant(5), "bar": tf.constant(6)}
    receiver_tensor = tf.constant(["test"])
    _ = export.ServingInputReceiver(features, receiver_tensor)

  def test_multi_feature_multi_receiver(self):
    features = {"foo": tf.constant(5), "bar": tf.constant(6)}
    receiver_tensors = {"baz": tf.constant(5), "qux": tf.constant(6)}
    _ = export.ServingInputReceiver(features, receiver_tensors)

  def test_feature_wrong_type(self):
    feature = "not a tensor"
    receiver_tensor = tf.constant(["test"])
    with self.assertRaises(ValueError):
      _ = export.ServingInputReceiver(feature, receiver_tensor)

  def test_feature_labeled_tensor(self):
    feature = LabeledTensorMock()
    receiver_tensor = tf.constant(["test"])
    _ = export.ServingInputReceiver(feature, receiver_tensor)

  def test_receiver_wrong_type(self):
    feature = tf.constant(5)
    receiver_tensor = "not a tensor"
    with self.assertRaises(ValueError):
      _ = export.ServingInputReceiver(feature, receiver_tensor)


class UnsupervisedInputReceiverTest(tf.test.TestCase):

  # Since this is basically a wrapper around ServingInputReceiver, we only
  # have a simple sanity check to ensure that it works.

  def test_unsupervised_input_receiver_constructor(self):
    """Tests that no errors are raised when input is expected."""
    features = {
        "feature0":
            tf.constant([0]),
        u"feature1":
            tf.constant([1]),
        "feature2":
            tf.sparse.SparseTensor(
                indices=[[0, 0]], values=[1], dense_shape=[1, 1]),
        42:  # ints are allowed only in the `features` dict
            tf.constant([3]),
    }
    receiver_tensors = {
        "example0": tf.constant(["test0"], name="example0"),
        u"example1": tf.constant(["test1"], name="example1"),
    }
    export.UnsupervisedInputReceiver(features, receiver_tensors)


class SupervisedInputReceiverTest(tf.test.TestCase):

  def test_input_receiver_constructor(self):
    """Tests that no errors are raised when input is expected."""
    features = {
        "feature0":
            tf.constant([0]),
        u"feature1":
            tf.constant([1]),
        "feature2":
            tf.sparse.SparseTensor(
                indices=[[0, 0]], values=[1], dense_shape=[1, 1]),
        42:  # ints are allowed in the `features` dict
            tf.constant([3]),
    }
    labels = {
        "classes": tf.constant([0] * 100),
        43:  # ints are allowed in the `labels` dict
            tf.constant([3]),
    }

    receiver_tensors = {
        "example0": tf.constant(["test0"], name="example0"),
        u"example1": tf.constant(["test1"], name="example1"),
    }
    export.SupervisedInputReceiver(features, labels, receiver_tensors)

  def test_input_receiver_raw_values(self):
    """Tests that no errors are raised when input is expected."""
    features = {
        "feature0":
            tf.constant([0]),
        u"feature1":
            tf.constant([1]),
        "feature2":
            tf.sparse.SparseTensor(
                indices=[[0, 0]], values=[1], dense_shape=[1, 1]),
        42:  # ints are allowed in the `features` dict
            tf.constant([3]),
    }

    labels = {
        "classes": tf.constant([0] * 100),
        43:  # ints are allowed in the `labels` dict
            tf.constant([3]),
    }

    receiver_tensors = {
        "example0": tf.constant(["test0"], name="example0"),
        u"example1": tf.constant(["test1"], name="example1"),
    }
    rec = export.SupervisedInputReceiver(features["feature2"], labels,
                                         receiver_tensors)
    self.assertIsInstance(rec.features, tf.sparse.SparseTensor)

    rec = export.SupervisedInputReceiver(features, labels["classes"],
                                         receiver_tensors)
    self.assertIsInstance(rec.labels, tf.Tensor)

  def test_input_receiver_features_invalid(self):
    features = tf.constant([0] * 100)
    labels = tf.constant([0])
    receiver_tensors = {
        "example0": tf.constant(["test0"], name="example0"),
        u"example1": tf.constant(["test1"], name="example1"),
    }

    with self.assertRaisesRegexp(ValueError, "features must be defined"):
      export.SupervisedInputReceiver(
          features=None, labels=labels, receiver_tensors=receiver_tensors)

    with self.assertRaisesRegexp(ValueError,
                                 "feature keys must be strings or ints"):
      export.SupervisedInputReceiver(
          features={1.11: tf.constant([1])},
          labels=labels,
          receiver_tensors=receiver_tensors)

    with self.assertRaisesRegexp(ValueError,
                                 "label keys must be strings or ints"):
      export.SupervisedInputReceiver(
          features=features,
          labels={1.11: tf.constant([1])},
          receiver_tensors=receiver_tensors)

    with self.assertRaisesRegexp(
        ValueError, "feature feature1 must be a Tensor, SparseTensor, or "
        "RaggedTensor."):
      export.SupervisedInputReceiver(
          features={"feature1": [1]},
          labels=labels,
          receiver_tensors=receiver_tensors)

    with self.assertRaisesRegexp(ValueError,
                                 "feature must be a Tensor, SparseTensor, "
                                 "or RaggedTensor."):
      export.SupervisedInputReceiver(
          features=[1], labels=labels, receiver_tensors=receiver_tensors)

    with self.assertRaisesRegexp(ValueError,
                                 "label must be a Tensor, SparseTensor, "
                                 "or RaggedTensor."):
      export.SupervisedInputReceiver(
          features=features, labels=100, receiver_tensors=receiver_tensors)

  def test_input_receiver_receiver_tensors_invalid(self):
    features = {
        "feature0":
            tf.constant([0]),
        u"feature1":
            tf.constant([1]),
        "feature2":
            tf.sparse.SparseTensor(
                indices=[[0, 0]], values=[1], dense_shape=[1, 1]),
    }
    labels = tf.constant([0])

    with self.assertRaisesRegexp(ValueError,
                                 "receiver_tensors must be defined"):
      export.SupervisedInputReceiver(
          features=features, labels=labels, receiver_tensors=None)

    with self.assertRaisesRegexp(ValueError,
                                 "receiver_tensor keys must be strings"):
      export.SupervisedInputReceiver(
          features=features,
          labels=labels,
          receiver_tensors={1: tf.constant(["test"], name="example0")})

    with self.assertRaisesRegexp(ValueError,
                                 "receiver_tensor example1 must be a Tensor"):
      export.SupervisedInputReceiver(
          features=features, labels=labels, receiver_tensors={"example1": [1]})

  def test_single_feature_single_receiver(self):
    feature = tf.constant(5)
    label = tf.constant(5)
    receiver_tensor = tf.constant(["test"])
    input_receiver = export.SupervisedInputReceiver(feature, label,
                                                    receiver_tensor)

    # single receiver is automatically named
    receiver_key, = input_receiver.receiver_tensors.keys()
    self.assertEqual("input", receiver_key)

  def test_multi_feature_single_receiver(self):
    features = {"foo": tf.constant(5), "bar": tf.constant(6)}
    labels = {"value": tf.constant(5)}
    receiver_tensor = tf.constant(["test"])
    _ = export.SupervisedInputReceiver(features, labels, receiver_tensor)

  def test_multi_feature_multi_receiver(self):
    features = {"foo": tf.constant(5), "bar": tf.constant(6)}
    labels = {"value": tf.constant(5)}
    receiver_tensors = {"baz": tf.constant(5), "qux": tf.constant(6)}
    _ = export.SupervisedInputReceiver(features, labels, receiver_tensors)

  def test_feature_labeled_tensor(self):
    feature = LabeledTensorMock()
    label = tf.constant(5)
    receiver_tensor = tf.constant(["test"])
    _ = export.SupervisedInputReceiver(feature, label, receiver_tensor)


class ExportTest(tf.test.TestCase):

  # Calling serving_input_receiver_fn requires graph mode.
  @test_util.deprecated_graph_mode_only
  def test_build_parsing_serving_input_receiver_fn(self):
    feature_spec = {
        "int_feature": tf.io.VarLenFeature(tf.dtypes.int64),
        "float_feature": tf.io.VarLenFeature(tf.dtypes.float32)
    }
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    with tf.Graph().as_default():
      serving_input_receiver = serving_input_receiver_fn()
      self.assertEqual(
          set(["int_feature", "float_feature"]),
          set(serving_input_receiver.features.keys()))
      self.assertEqual(
          set(["examples"]),
          set(serving_input_receiver.receiver_tensors.keys()))

      example = example_pb2.Example()
      text_format.Parse(
          "features: { "
          "  feature: { "
          "    key: 'int_feature' "
          "    value: { "
          "      int64_list: { "
          "        value: [ 21, 2, 5 ] "
          "      } "
          "    } "
          "  } "
          "  feature: { "
          "    key: 'float_feature' "
          "    value: { "
          "      float_list: { "
          "        value: [ 525.25 ] "
          "      } "
          "    } "
          "  } "
          "} ", example)

      with self.cached_session() as sess:
        sparse_result = sess.run(
            serving_input_receiver.features,
            feed_dict={
                serving_input_receiver.receiver_tensors["examples"].name: [
                    example.SerializeToString()
                ]
            })
        self.assertAllEqual([[0, 0], [0, 1], [0, 2]],
                            sparse_result["int_feature"].indices)
        self.assertAllEqual([21, 2, 5], sparse_result["int_feature"].values)
        self.assertAllEqual([[0, 0]], sparse_result["float_feature"].indices)
        self.assertAllEqual([525.25], sparse_result["float_feature"].values)

  # Calling serving_input_receiver_fn requires graph mode.
  @test_util.deprecated_graph_mode_only
  def test_build_raw_serving_input_receiver_fn_name(self):
    """Test case for issue #12755."""
    f = {
        "feature":
            tf.compat.v1.placeholder(
                name="feature", shape=[32], dtype=tf.dtypes.float32)
    }
    serving_input_receiver_fn = export.build_raw_serving_input_receiver_fn(f)
    v = serving_input_receiver_fn()
    self.assertIsInstance(v, export.ServingInputReceiver)

  # Calling serving_input_receiver_fn requires graph mode.
  @test_util.deprecated_graph_mode_only
  def test_build_raw_serving_input_receiver_fn_without_shape(self):
    """Test case for issue #21178."""
    f = {
        "feature_1": tf.compat.v1.placeholder(tf.dtypes.float32),
        "feature_2": tf.compat.v1.placeholder(tf.dtypes.int32)
    }
    serving_input_receiver_fn = export.build_raw_serving_input_receiver_fn(f)
    v = serving_input_receiver_fn()
    self.assertIsInstance(v, export.ServingInputReceiver)
    self.assertEqual(tensor_shape.unknown_shape(),
                     v.receiver_tensors["feature_1"].shape)
    self.assertEqual(tensor_shape.unknown_shape(),
                     v.receiver_tensors["feature_2"].shape)

  def test_build_raw_serving_input_receiver_fn(self):
    features = {
        "feature_1": tf.constant(["hello"]),
        "feature_2": tf.constant([42])
    }
    serving_input_receiver_fn = export.build_raw_serving_input_receiver_fn(
        features)
    with tf.Graph().as_default():
      serving_input_receiver = serving_input_receiver_fn()
      self.assertEqual(
          set(["feature_1", "feature_2"]),
          set(serving_input_receiver.features.keys()))
      self.assertEqual(
          set(["feature_1", "feature_2"]),
          set(serving_input_receiver.receiver_tensors.keys()))
      self.assertEqual(
          tf.dtypes.string,
          serving_input_receiver.receiver_tensors["feature_1"].dtype)
      self.assertEqual(
          tf.dtypes.int32,
          serving_input_receiver.receiver_tensors["feature_2"].dtype)

  def test_build_raw_supervised_input_receiver_fn(self):
    features = {
        "feature_1": tf.constant(["hello"]),
        "feature_2": tf.constant([42])
    }
    labels = {"foo": tf.constant([5]), "bar": tf.constant([6])}
    input_receiver_fn = export.build_raw_supervised_input_receiver_fn(
        features, labels)
    with tf.Graph().as_default():
      input_receiver = input_receiver_fn()
      self.assertEqual(
          set(["feature_1", "feature_2"]), set(input_receiver.features.keys()))
      self.assertEqual(set(["foo", "bar"]), set(input_receiver.labels.keys()))
      self.assertEqual(
          set(["feature_1", "feature_2", "foo", "bar"]),
          set(input_receiver.receiver_tensors.keys()))
      self.assertEqual(tf.dtypes.string,
                       input_receiver.receiver_tensors["feature_1"].dtype)
      self.assertEqual(tf.dtypes.int32,
                       input_receiver.receiver_tensors["feature_2"].dtype)

  def test_build_raw_supervised_input_receiver_fn_raw_tensors(self):
    features = {
        "feature_1": tf.constant(["hello"]),
        "feature_2": tf.constant([42])
    }
    labels = {"foo": tf.constant([5]), "bar": tf.constant([6])}
    input_receiver_fn1 = export.build_raw_supervised_input_receiver_fn(
        features["feature_1"], labels)
    input_receiver_fn2 = export.build_raw_supervised_input_receiver_fn(
        features["feature_1"], labels["foo"])
    with tf.Graph().as_default():
      input_receiver = input_receiver_fn1()
      self.assertIsInstance(input_receiver.features, tf.Tensor)
      self.assertEqual(set(["foo", "bar"]), set(input_receiver.labels.keys()))
      self.assertEqual(
          set(["input", "foo", "bar"]),
          set(input_receiver.receiver_tensors.keys()))

      input_receiver = input_receiver_fn2()
      self.assertIsInstance(input_receiver.features, tf.Tensor)
      self.assertIsInstance(input_receiver.labels, tf.Tensor)
      self.assertEqual(
          set(["input", "label"]), set(input_receiver.receiver_tensors.keys()))

  def test_build_raw_supervised_input_receiver_fn_batch_size(self):
    features = {
        "feature_1": tf.constant(["hello"]),
        "feature_2": tf.constant([42])
    }
    labels = {"foo": tf.constant([5]), "bar": tf.constant([6])}
    input_receiver_fn = export.build_raw_supervised_input_receiver_fn(
        features, labels, default_batch_size=10)
    with tf.Graph().as_default():
      input_receiver = input_receiver_fn()
      self.assertEqual([10], input_receiver.receiver_tensors["feature_1"].shape)
      self.assertEqual([10], input_receiver.features["feature_1"].shape)

  def test_build_raw_supervised_input_receiver_fn_overlapping_keys(self):
    features = {
        "feature_1": tf.constant(["hello"]),
        "feature_2": tf.constant([42])
    }
    labels = {"feature_1": tf.constant([5]), "bar": tf.constant([6])}
    with self.assertRaises(ValueError):
      export.build_raw_supervised_input_receiver_fn(features, labels)

  def test_build_supervised_input_receiver_fn_from_input_fn(self):

    def dummy_input_fn():
      return ({
          "x": tf.constant([[1], [1]]),
          "y": tf.constant(["hello", "goodbye"])
      }, tf.constant([[1], [1]]))

    input_receiver_fn = export.build_supervised_input_receiver_fn_from_input_fn(
        dummy_input_fn)

    with tf.Graph().as_default():
      input_receiver = input_receiver_fn()
      self.assertEqual(set(["x", "y"]), set(input_receiver.features.keys()))
      self.assertIsInstance(input_receiver.labels, tf.Tensor)
      self.assertEqual(
          set(["x", "y", "label"]), set(input_receiver.receiver_tensors.keys()))

  def test_build_supervised_input_receiver_fn_from_input_fn_args(self):

    def dummy_input_fn(feature_key="x"):
      return ({
          feature_key: tf.constant([[1], [1]]),
          "y": tf.constant(["hello", "goodbye"])
      }, {
          "my_label": tf.constant([[1], [1]])
      })

    input_receiver_fn = export.build_supervised_input_receiver_fn_from_input_fn(
        dummy_input_fn, feature_key="z")

    with tf.Graph().as_default():
      input_receiver = input_receiver_fn()
      self.assertEqual(set(["z", "y"]), set(input_receiver.features.keys()))
      self.assertEqual(set(["my_label"]), set(input_receiver.labels.keys()))
      self.assertEqual(
          set(["z", "y", "my_label"]),
          set(input_receiver.receiver_tensors.keys()))


class TensorServingReceiverTest(tf.test.TestCase):

  def test_tensor_serving_input_receiver_constructor(self):
    features = tf.constant([0])
    receiver_tensors = {
        "example0": tf.constant(["test0"], name="example0"),
        u"example1": tf.constant(["test1"], name="example1"),
    }
    r = export.TensorServingInputReceiver(features, receiver_tensors)
    self.assertIsInstance(r.features, tf.Tensor)
    self.assertIsInstance(r.receiver_tensors, dict)

  def test_tensor_serving_input_receiver_sparse(self):
    features = tf.sparse.SparseTensor(
        indices=[[0, 0]], values=[1], dense_shape=[1, 1])
    receiver_tensors = {
        "example0": tf.constant(["test0"], name="example0"),
        u"example1": tf.constant(["test1"], name="example1"),
    }
    r = export.TensorServingInputReceiver(features, receiver_tensors)
    self.assertIsInstance(r.features, tf.sparse.SparseTensor)
    self.assertIsInstance(r.receiver_tensors, dict)

  def test_serving_input_receiver_features_invalid(self):
    receiver_tensors = {
        "example0": tf.constant(["test0"], name="example0"),
        u"example1": tf.constant(["test1"], name="example1"),
    }

    with self.assertRaisesRegexp(ValueError, "features must be defined"):
      export.TensorServingInputReceiver(
          features=None, receiver_tensors=receiver_tensors)

    with self.assertRaisesRegexp(ValueError, "feature must be a Tensor"):
      export.TensorServingInputReceiver(
          features={"1": tf.constant([1])}, receiver_tensors=receiver_tensors)

  def test_serving_input_receiver_receiver_tensors_invalid(self):
    features = tf.constant([0])

    with self.assertRaisesRegexp(ValueError,
                                 "receiver_tensors must be defined"):
      export.TensorServingInputReceiver(
          features=features, receiver_tensors=None)

    with self.assertRaisesRegexp(ValueError,
                                 "receiver_tensor keys must be strings"):
      export.TensorServingInputReceiver(
          features=features,
          receiver_tensors={1: tf.constant(["test"], name="example0")})

    with self.assertRaisesRegexp(ValueError,
                                 "receiver_tensor example1 must be a Tensor"):
      export.TensorServingInputReceiver(
          features=features, receiver_tensors={"example1": [1]})


if __name__ == "__main__":
  tf.test.main()
