# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for sequence embedding features using TPU and TPUEstimator."""

import os
from typing import Dict, List, Text, Tuple
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.contrib import summary as contrib_summary


class TPUEnqueueSequenceTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    temp_dir = self.get_temp_dir()
    self._model_dir = os.path.join(temp_dir, 'model_dir')
    self._summary_dir = os.path.join(temp_dir, 'summaries')

    os.mkdir(self._model_dir)
    os.mkdir(self._summary_dir)

  # The key in the dataset which holds the sparse IDs. TPUEstimator will pass
  # the embeddings in the features dictionary arg of model_fn after performing
  # the embedding lookups.
  _KEY = 'SparseIDs'

  # The names of the summaries which hold the activations/sequence lengths.
  _SUMMARY_ACTIVATIONS = 'summary_activations'
  _SUMMARY_SEQUENCE_LENGTHS = 'summary_sequence_lengths'

  def get_activations_and_sequence_lengths(
      self,
      embedding_weights: List[List[float]],
      sparse_ids: tf.SparseTensorValue,
      batch_size: int,
      max_sequence_length: int,
      dimension: int,
      combiner: Text = 'mean',
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Gets the activations and seq lengths for a batch of sparse IDs.

    This method uses TPUEstimator and the Feature Column API to get embedding
    activations for a batch of sparse of sparse IDs using a specified set of
    embedding weights.

    Args:
      embedding_weights: The embedding weights as a 2D list of floats.  The
        outer list length is the vocabulary size of the embedding table.  The
        inner list length is the dimension of the embedding weights.
      sparse_ids: The embedding IDs to lookup. This is a 2D SparseTensorValue of
        shape [batch_size, max_sequence_length].
      batch_size: The size of the first dimension of sparse_ids.
      max_sequence_length:  The size of the second dimension of sparse_ids.
      dimension: The embedding dimension size (number of floats for each
        embedding ID).
      combiner: The embedding column combiner (used for multivalent features).

    Returns:
      A tuple containing:
        activations:  The activations for the specified sparse_ids.
          type=float32, shape=[batch_size, max_sequence_length, dimension]
        sequence_lengths: The sequence length of each example.
          type=int64. shape=[batch_size].
    """

    vocab_size = len(embedding_weights)
    categorical_column = (
        tf.feature_column.sequence_categorical_column_with_identity(
            key=self._KEY,
            num_buckets=vocab_size,
        ))

    # Create embedding column initialized with weights provided by caller.
    embedding_column = tf.tpu.experimental.embedding_column(
        categorical_column,
        dimension=dimension,
        max_sequence_length=max_sequence_length,
        initializer=tf.constant_initializer(embedding_weights),
        combiner=combiner,
    )

    # Add an SGD optimizer. This choice is arbitrary for computing activations.
    # It's only required to avoid an undefined gradients error.
    embedding_opt = tf.tpu.experimental.StochasticGradientDescentParameters(.1)
    embedding_config_spec = tf.estimator.tpu.experimental.EmbeddingConfigSpec(
        feature_columns=[embedding_column],
        optimization_parameters=embedding_opt,
    )

    def _input_fn(params: Dict[Text, int]) -> tf.data.Dataset:
      """Creates a batched dataset containing the sparse_ids as a feature."""
      # Convert sparse IDs to batched dataset.
      sparse_ids_dataset = tf.data.Dataset.range(1).map(
          lambda x: {self._KEY: tf.SparseTensor.from_value(sparse_ids)})

      # Unbatch and rebatch the dataset based on the batch_size param from
      # TPUEstimator. This is necessary for shape validation performed internal
      # to TPUEstimator.
      return sparse_ids_dataset.unbatch().repeat().batch(params['batch_size'])

    def _host_call(
        concat_activations: tf.Tensor,
        concat_sequence_lengths: tf.Tensor,
    ) -> List[tf.Operation]:
      """Stores the activations and sequence lengths into a summary.

      TPUEstimator will concat the activations and sequence lengths from the
      minibatches on each core along axis=0 and pass them to this host call.
      This host call writes them to a file using the TF summary APIs.

      Args:
        concat_activations: The activations for the global batch. 2D
          Tensor(type=float32, shape=[batch_size, max_sequence_length]).
        concat_sequence_lengths:  The sequence lengths for the global batch. 2D
          Tensor(type=int64, shape=[batch_size, max_sequence_length]).

      Returns:
        A list of summary ops for TPUEstimator to run on the host.
      """
      with contrib_summary.create_file_writer(self._summary_dir).as_default():
        with contrib_summary.always_record_summaries():
          contrib_summary.generic(
              self._SUMMARY_ACTIVATIONS,
              concat_activations,
          )
          contrib_summary.generic(self._SUMMARY_SEQUENCE_LENGTHS,
                                  concat_sequence_lengths)
          return contrib_summary.all_summary_ops()

    def _model_fn(
        features: Dict[Text, tf.Tensor],
        params: Dict[Text, int],
        mode: tf.estimator.ModeKeys,
    ) -> tf.estimator.tpu.TPUEstimatorSpec:
      """A model which writes activations and sequence lengths to a file.

      This method creates a model to extract the activations and sequence
      lengths on each TPU core and pass them to a host call which writes them
      to a file.

      The model also applies an optimizer to the activations simply to avoid an
      undefined gradients error.

      Args:
        features: A dictionary mapping keys to tensor inputs.
        params: Parameters passed by TPUEstimator.
        mode: Mode can be (TRAIN, EVAL, PREDICT).

      Returns:
        A TPUEstimatorSpec which holds the training_op that TPUEstimator will
        run on TPU and the host_call that TPUEstimator will run on the host.
      """
      del params
      input_layer = tf.keras.experimental.SequenceFeatures([embedding_column])
      activations, sequence_lengths = input_layer(features)
      opt = tf.tpu.CrossShardOptimizer(tf.train.GradientDescentOptimizer(0.1))
      loss = tf.reduce_sum(activations)
      train_op = opt.minimize(loss, global_step=tf.train.get_global_step())

      return tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          host_call=(_host_call, [activations, sequence_lengths]),
      )

    tpu_config = tf.estimator.tpu.TPUConfig(
        per_host_input_for_training=(
            tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2),)
    run_config = tf.estimator.tpu.RunConfig(
        session_config=tf.ConfigProto(isolate_session_state=True),
        tpu_config=tpu_config,
    )
    estimator = tf.estimator.tpu.TPUEstimator(
        model_fn=_model_fn,
        model_dir=self._model_dir,
        use_tpu=True,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        config=run_config,
        embedding_config_spec=embedding_config_spec,
    )

    # Train for 1 step and store the activations as summaries.
    estimator.train(_input_fn, steps=1)

    # Read the event summaries and decode the activation tensors.
    output = {}
    for filename in tf.io.gfile.listdir(self._summary_dir):
      filepath = os.path.join(os.path.join(self._summary_dir, filename))
      for event in tf.train.summary_iterator(filepath):
        for v in event.summary.value:
          decoded = tf.io.decode_raw(v.tensor.tensor_content, v.tensor.dtype)
          shape = tf.TensorShape(v.tensor.tensor_shape)
          output[v.tag] = tf.reshape(decoded, shape)
    return (output[self._SUMMARY_ACTIVATIONS],
            output[self._SUMMARY_SEQUENCE_LENGTHS])

  def test_non_contiguous_sequence(self):
    """Tests embedding lookups for non-contiguous sparse IDs.

    A "non-contiguous sequence" is a sequence which has missing values followed
    by actual values.
    """
    batch_size = 4
    max_sequence_length = 3
    dimension = 2
    embedding_weights = np.float32([
        [-5., -5.],  # embedding ID = 0
        [10., 11.],  # embedding ID = 1
        [20., 21.],  # embedding ID = 2
        [30., 31.],  # embedding ID = 3
        [40., 41.],  # embedding ID = 4
        [50., 51.],  # embedding ID = 5
    ])

    # The sparse_ids are indexes into the embedding_weights for each
    # (example, sequence_index).
    sparse_ids = tf.SparseTensorValue(
        indices=[[0, 0], [1, 0], [1, 1], [2, 0], [2, 2]],
        values=[
            1,  # Example 0, sequence_index 0
            2,  # Example 1, sequence_index 0
            3,  # Example 1, sequence_index 1
            4,  # Example 2, sequence_index 0
            5,  # Example 2, sequence_index 2
        ],
        dense_shape=[batch_size, max_sequence_length],
    )

    activations, sequence_lengths = self.get_activations_and_sequence_lengths(
        embedding_weights,
        sparse_ids,
        batch_size,
        max_sequence_length,
        dimension,
    )

    self.assertAllEqual(
        [
            [  # Example 0
                [10, 11],  # Sequence Index = 0
                [0., 0.],  # Sequence Index = 1
                [0., 0.],  # Sequence Index = 2
            ],
            [  # Example 1
                [20, 21],  # Sequence Index = 0
                [30, 31],  # Sequence Index = 1
                [0., 0.],  # Sequence Index = 2
            ],
            [  # Example 2
                [40, 41],  # Sequence Index = 0
                [0., 0.],  # Sequence Index = 1 (Missing value mid-sequence)
                [50, 51],  # Sequence Index = 2
            ],
            [  # Example 3
                [0., 0.],  # Sequence Index = 0
                [0., 0.],  # Sequence Index = 1
                [0., 0.],  # Sequence Index = 2
            ],
        ],
        activations)
    self.assertAllEqual(
        [
            1,  # Example 0
            2,  # Example 1
            3,  # Example 2
            0,  # Example 3
        ],
        sequence_lengths,
    )

  def test_non_contiguous_sequence_with_length_gt_max_sequence_length(self):
    """Tests non contiguous sequence which has length > max_sequence_length.

    A "non-contiguous sequence" is a sequence which has missing values followed
    by actual values.

    Additionally, this test has a sequence with length > max_sequence_length. In
    this case, we expect the sequence to be truncated from the right.
    """
    batch_size = 4
    max_sequence_length = 3
    dimension = 1
    embedding_weights = np.float32([
        [-5.],  # embedding ID = 0
        [10.],  # embedding ID = 1
        [20.],  # embedding ID = 2
        [30.],  # embedding ID = 3
        [40.],  # embedding ID = 4
        [50.],  # embedding ID = 5
    ])

    # The sparse_ids are indexes into the embedding_weights for each
    # (example, sequence_index).  Sequence indexes larger than max_sequence
    # length will be truncated.
    sparse_ids = tf.SparseTensorValue(
        indices=[[0, 0], [1, 0], [1, 1], [2, 0], [2, 2], [2, 3]],
        values=[
            1,  # Example 0, sequence_index 0
            2,  # Example 1, sequence_index 0
            3,  # Example 1, sequence_index 1
            4,  # Example 2, sequence_index 0
            5,  # Example 2, sequence_index 2
            6,  # Example 2, sequence_index 3
        ],
        dense_shape=[batch_size, max_sequence_length + 1],
    )

    activations, sequence_lengths = self.get_activations_and_sequence_lengths(
        embedding_weights,
        sparse_ids,
        batch_size,
        max_sequence_length,
        dimension,
    )

    self.assertAllEqual(
        [
            [  # Example 0
                [10],  # Sequence Index = 0
                [0.],  # Sequence Index = 1
                [0.],  # Sequence Index = 2
            ],
            [  # Example 1
                [20],  # Sequence Index = 0
                [30],  # Sequence Index = 1
                [0.],  # Sequence Index = 2
            ],
            [  # Example 2 (Truncated)
                [40],  # Sequence Index = 0
                [0.],  # Sequence Index = 1 (Missing value mid-sequence)
                [50],  # Sequence Index = 2
            ],
            [  # Example 3
                [0.],  # Sequence Index = 0
                [0.],  # Sequence Index = 1
                [0.],  # Sequence Index = 2
            ],
        ],
        activations)

    self.assertAllEqual(
        [
            1,  # Example 0
            2,  # Example 1
            3,  # Example 2
            0,  # Example 3
        ],
        sequence_lengths,
    )

  @parameterized.named_parameters(
      ('sum_combiner', 'sum'),
      ('mean_combiner', 'mean'),
  )
  def test_multivalent_sequence_features(self, combiner: Text):
    """Tests multivalent sequence embedding features.

    Args:
      combiner: The combiner used to reduce multivalent features.  A multivalent
        sequence can have many IDs per sequence index.  The input for
        multivalent sequence features is a 3D SparseTensor (instead of a 2D
        SparseTensor for univalent sequence features).  The last dimension
        represents the index that will be reduced (using the combiner).
    """
    batch_size = 4
    max_sequence_length = 3
    dimension = 1
    embedding_weights = np.float32([
        [-5.],  # embedding ID = 0
        [10.],  # embedding ID = 1
        [20.],  # embedding ID = 2
        [30.],  # embedding ID = 3
        [40.],  # embedding ID = 4
        [50.],  # embedding ID = 5
    ])

    # For multivalent sequence features, IDs are a 3D sparse tensor.
    # The outer dimension is batch, the middle dimension is sequence, and the
    # last dimension is the index.
    sparse_ids = tf.SparseTensorValue(
        indices=[
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 1, 0],
            [3, 0, 0],
            [3, 2, 0],
            [3, 2, 1],
            [3, 3, 0],
        ],
        values=[
            1,  # Example 0, sequence_index 0,  id_index 0.
            0,  # Example 0, sequence_index 0,  id_index 1.
            2,  # Example 1, sequence_index 0,  id_index 0.
            3,  # Example 1, sequence_index 1,  id_index 0.
            4,  # Example 3, sequence_index 0,  id_index 0.
            5,  # Example 3, sequence_index 2.  id_index 0.
            2,  # Example 3, sequence_index 2.  id_index 1.
            5,  # Example 3, sequence_index 3,  id_index 0.
        ],
        dense_shape=[batch_size, max_sequence_length + 1, 2],
    )

    activations, sequence_lengths = self.get_activations_and_sequence_lengths(
        embedding_weights,
        sparse_ids,
        batch_size,
        max_sequence_length,
        dimension,
        combiner=combiner,
    )

    self.assertAllEqual(
        [
            [  # Example 0
                [5 if combiner == 'sum' else 2.5],  # Sequence Index = 0.
                [0.],  # Sequence Index = 1.
                [0.],  # Sequence Index = 2.
            ],
            [  # Example 1
                [20],  # Sequence Index = 0.
                [30],  # Sequence Index = 1.
                [0.],  # Sequence Index = 2.
            ],
            [  # Example 2
                [0.],  # Sequence Index = 0.
                [0.],  # Sequence Index = 1.
                [0.],  # Sequence Index = 2.
            ],
            [  # Example 3
                [40],  # Sequence Index = 0.
                [0.],  # Sequence Index = 1.
                [70 if combiner == 'sum' else 35],  # Sequence Index = 2.
            ],
        ],
        activations,
    )

    self.assertAllEqual(
        [
            1,  # Example 0
            2,  # Example 1
            0,  # Example 2
            3,  # Example 3
        ],
        sequence_lengths,
    )


if __name__ == '__main__':
  tf.test.main()
