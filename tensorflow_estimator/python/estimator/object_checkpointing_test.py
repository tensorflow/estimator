# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Integration tests for Estimator + object checkpointing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
# pylint: disable=g-import-not-at-top
try:
  from tensorflow.python.training.tracking import util
except ImportError:
  # TODO(allenl): Remove this after cl/229814711 syncs
  from tensorflow.python.training.checkpointable import util
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import monitored_session
from tensorflow.python.training import training_util

from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib


class SubclassedModel(training.Model):

  def __init__(self):
    super(SubclassedModel, self).__init__()
    self.dense_one = core.Dense(5)
    self.dense_two = core.Dense(1)

  def call(self, inputs):
    return self.dense_two(self.dense_one(inputs))


class ObjectCheckpointingTest(test.TestCase):

  def _make_estimator(self, model_dir):

    def _model_fn(features, labels, mode):
      del labels
      model = SubclassedModel()
      optimizer = adam.Adam(0.01)
      checkpoint = util.Checkpoint(
          step=training_util.get_or_create_global_step(),
          optimizer=optimizer,
          model=model)
      # Make the save counter to satisfy the assert_consumed() assertion later
      checkpoint.save_counter  # pylint: disable=pointless-statement
      with backprop.GradientTape() as tape:
        output = model(features)
        loss = math_ops.reduce_sum(output)
      variables = model.trainable_variables
      gradients = tape.gradient(loss, variables)
      train_op = control_flow_ops.group(
          optimizer.apply_gradients(zip(gradients, variables)),
          checkpoint.step.assign_add(1))
      return model_fn_lib.EstimatorSpec(
          mode,
          loss=loss,
          train_op=train_op,
          predictions=dict(
              output=output,
              bias=array_ops.tile(
                  model.dense_two.bias[None, :],
                  [array_ops.shape(output)[0], 1]),
              step=array_ops.tile(
                  checkpoint.step[None],
                  [array_ops.shape(output)[0]])),
          scaffold=monitored_session.Scaffold(saver=checkpoint)
      )

    est = estimator_lib.EstimatorV2(model_fn=_model_fn, model_dir=model_dir)

    def _input_fn():
      return dataset_ops.Dataset.from_tensors([1.]).repeat().batch(10)

    return est, _input_fn

  def testTwoWayCompatibility(self):
    save_model_dir = os.path.join(self.get_temp_dir(), 'model_dir')
    save_est, input_fn = self._make_estimator(save_model_dir)

    save_est.train(input_fn, steps=3)

    model = SubclassedModel()
    optimizer = adam.Adam(0.01)
    checkpoint = util.Checkpoint(
        step=variables_lib.Variable(0, dtype=dtypes.int64),
        optimizer=optimizer,
        model=model)
    status = checkpoint.restore(
        checkpoint_management.latest_checkpoint(save_model_dir))
    self.assertEqual(3, self.evaluate(checkpoint.step))
    with backprop.GradientTape() as tape:
      output = model(constant_op.constant([[1.]]))
      loss = math_ops.reduce_sum(output)
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    status.assert_consumed()

    # The optimizer uses this for some reason...
    backend.clear_session()

    load_model_dir = os.path.join(self.get_temp_dir(), 'load_model_dir/')
    checkpoint.step.assign(40)
    checkpoint.model.dense_two.bias.assign([13.])
    checkpoint.save(load_model_dir)
    load_est, input_fn = self._make_estimator(load_model_dir)
    predictions = load_est.predict(input_fn)
    predictions = next(predictions)
    self.assertAllClose([13.], predictions['bias'])
    self.assertEqual(40, predictions['step'])


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
