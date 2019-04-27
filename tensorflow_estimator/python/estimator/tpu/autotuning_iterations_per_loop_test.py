# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Tests for auto-tuning iterations_per_loop using TPUStopWithAutoTunedStepHook."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from tensorflow.python.platform import test
from tensorflow.python.training import training_util
from tensorflow_estimator.python.estimator.tpu import iteration_count_estimator
from tensorflow_estimator.python.estimator.tpu import tpu_estimator
from tensorflow_estimator.python.estimator.tpu import util as util_lib


class IterationsPerLoopParsingTest(test.TestCase):

  def _parse_and_validate_iterations_per_loop(self, value, expected_value,
                                              expected_unit):
    d = util_lib.parse_iterations_per_loop(value)
    self.assertTrue(d)
    self.assertEqual(d.value, expected_value)
    self.assertEqual(d.unit, expected_unit)

  def _parse_and_validate_invalid_iterations_per_loop(self, value):
    with self.assertRaises(ValueError) as ve:
      self._parse_and_validate_iterations_per_loop(value, 0, '')
      self.assertTrue(
          ve.exception.message.startswith(
              'Invalid `iterations_per_loop` value.'))

  def test_parsing_iterations_per_loop(self):
    """Tests parsing valid and invalid `iterations_per_loop` values."""

    self._parse_and_validate_iterations_per_loop(1, 1, 'count')
    self._parse_and_validate_iterations_per_loop('1', 1, 'count')
    self._parse_and_validate_iterations_per_loop(2, 2, 'count')
    self._parse_and_validate_iterations_per_loop(10, 10, 'count')
    self._parse_and_validate_iterations_per_loop(123, 123, 'count')
    self._parse_and_validate_iterations_per_loop('123', 123, 'count')
    self._parse_and_validate_iterations_per_loop('1h', 3600, 'seconds')
    self._parse_and_validate_iterations_per_loop('1m', 60, 'seconds')
    self._parse_and_validate_iterations_per_loop('1s', 1, 'seconds')
    self._parse_and_validate_iterations_per_loop('10h', 10 * 3600, 'seconds')
    self._parse_and_validate_iterations_per_loop('10m', 10 * 60, 'seconds')
    self._parse_and_validate_iterations_per_loop('10s', 10, 'seconds')
    self._parse_and_validate_iterations_per_loop('100h', 100 * 3600, 'seconds')
    self._parse_and_validate_iterations_per_loop('1000m', 1000 * 60, 'seconds')
    self._parse_and_validate_iterations_per_loop('10800s', 10800, 'seconds')
    self._parse_and_validate_invalid_iterations_per_loop(+0)
    self._parse_and_validate_invalid_iterations_per_loop(0)
    self._parse_and_validate_invalid_iterations_per_loop(-0)
    self._parse_and_validate_invalid_iterations_per_loop(-0o12)
    self._parse_and_validate_invalid_iterations_per_loop('012')
    self._parse_and_validate_invalid_iterations_per_loop('001')
    self._parse_and_validate_invalid_iterations_per_loop('0')
    self._parse_and_validate_invalid_iterations_per_loop('01')
    self._parse_and_validate_invalid_iterations_per_loop('-1')
    self._parse_and_validate_invalid_iterations_per_loop('-0h')
    self._parse_and_validate_invalid_iterations_per_loop('0h')
    self._parse_and_validate_invalid_iterations_per_loop('0s')
    self._parse_and_validate_invalid_iterations_per_loop('0m')
    self._parse_and_validate_invalid_iterations_per_loop('-1h')
    self._parse_and_validate_invalid_iterations_per_loop('-1s')
    self._parse_and_validate_invalid_iterations_per_loop('-1m')


class IterationPredictorTest(test.TestCase):

  def setUp(self):
    self.estimator = iteration_count_estimator.IterationCountEstimator(
        capacity=5)

  def test_empty(self):
    """Tests on empty queue."""
    self.assertEqual(self.estimator._min_iterations, self.estimator.get(1))
    self.assertEqual(self.estimator._min_iterations, self.estimator.get(10))

  def test_reset(self):
    """Tests reset states."""
    self.assertEqual(0, self.estimator._sample_count)
    self.assertEqual(self.estimator._min_iterations, self.estimator.get(50))
    self.assertEqual(0, len(self.estimator._buffer_wheel))
    self.estimator._reset()
    self.assertEqual(0, self.estimator._sample_count)
    self.assertEqual(self.estimator._min_iterations, self.estimator.get(100))
    self.assertEqual(0, len(self.estimator._buffer_wheel))
    self.estimator.update(9, 1)
    self.assertEqual(1, self.estimator._sample_count)
    self.assertEqual(self.estimator._min_iterations, self.estimator.get(8))

  def test_invalid_update(self):
    """Tests reject invalid update."""
    self.estimator._reset()
    self.estimator.update(0, 0)
    self.assertEqual(0, len(self.estimator._buffer_wheel))
    with self.assertRaises(ValueError) as ve:
      self.assertEqual(self.estimator._min_iterations, self.estimator.get(-1))
      self.assertIn('Invalid `total_secs`', ve.message)
    with self.assertRaises(ValueError) as ve:
      self.assertEqual(self.estimator._min_iterations, self.estimator.get(0))
      self.assertIn('Invalid `total_secs`', ve.message)

  def test_zero_mean(self):
    """Tests getting estimate when the elapsed time mean value is zero."""
    self.estimator.update(0, 1)
    self.assertEqual(self.estimator._min_iterations, self.estimator.get(10))
    self.estimator.update(0, 1)
    self.estimator.update(0, 1)
    self.assertEqual(self.estimator._min_iterations, self.estimator.get(10))

  def test_diff_less_than_percentage(self):
    """Tests computing diff less than a percentage."""
    self.assertTrue(self.estimator._diff_less_than_percentage(5, 10, 50))
    self.assertTrue(self.estimator._diff_less_than_percentage(2.5, 10, 75))
    self.assertTrue(self.estimator._diff_less_than_percentage(10, 10, 5))
    self.assertTrue(self.estimator._diff_less_than_percentage(9.5, 10, 5))
    self.assertTrue(self.estimator._diff_less_than_percentage(9.6, 10, 5))
    self.assertFalse(self.estimator._diff_less_than_percentage(11, 10, 5))
    self.assertFalse(self.estimator._diff_less_than_percentage(20, 10, 5))
    self.assertTrue(self.estimator._diff_less_than_percentage(10.3, 10, 5))
    self.assertTrue(self.estimator._diff_less_than_percentage(10.5, 10, 5))
    self.assertFalse(self.estimator._diff_less_than_percentage(10.6, 10, 5))
    self.assertFalse(self.estimator._diff_less_than_percentage(1, 10, 5))
    self.assertFalse(self.estimator._diff_less_than_percentage(9, 10, 5))
    with self.assertRaises(ValueError) as ve:
      self.assertTrue(self.estimator._diff_less_than_percentage(0, 10, 5))
      self.assertIn('Invalid `actual` value', ve.message)
    with self.assertRaises(ValueError) as ve:
      self.assertTrue(self.estimator._diff_less_than_percentage(10, 0, 5))
      self.assertIn('Invalid `target` value.', ve.message)

  def test_mean_runtime_secs(self):
    """Tests computing mean of step time secs."""
    self.assertEqual(0.0, self.estimator._mean_runtime_secs())
    self.estimator.update(1, 5)
    self.assertEqual(1.0, self.estimator._mean_runtime_secs())
    self.estimator._reset()
    self.estimator.update(2, 3)
    self.estimator.update(2, 3)
    self.estimator.update(2, 3)
    self.assertEqual(2.0, self.estimator._mean_runtime_secs())
    self.estimator._reset()
    self.estimator.update(1, 3)
    self.estimator.update(2, 3)
    self.assertEqual((1.0 + 2.0) / 2, self.estimator._mean_runtime_secs())

  def test_mean_step_time_secs(self):
    """Tests computing mean of step time secs."""
    self.assertEqual(0.0, self.estimator._mean_step_time_secs())
    self.estimator.update(1, 5)
    self.assertEqual(1.0 / 5, self.estimator._mean_step_time_secs())
    self.estimator._reset()
    self.estimator.update(2, 3)
    self.estimator.update(2, 3)
    self.estimator.update(2, 3)
    self.assertEqual(2.0 / 3, self.estimator._mean_step_time_secs())
    self.estimator._reset()
    self.estimator.update(1, 3)
    self.estimator.update(2, 3)
    self.assertEqual((1.0 / 3 + 2.0 / 3) / 2,
                     self.estimator._mean_step_time_secs())

  def _test_std_step_time_secs(self):
    """Tests computing std deviation of the step time secs."""
    self.assertEqual(0.0, self.estimator._std_step_time_secs())
    self.estimator.update(1, 5)
    self.estimator.update(1, 5)
    self.assertEqual(0.0, self.estimator._std_step_time_secs())
    self.estimator.update(4, 5)
    self.assertAlmostEqual(0.283, self.estimator._std_step_time_secs(), 3)
    self.estimator.update(5, 5)
    self.assertAlmostEqual(0.357, self.estimator._std_step_time_secs(), 3)

  def test_buffer_capacity(self):
    """Tests to make sure wheel is kept at its capacity."""
    self.estimator._reset(capacity=3)
    self.assertEqual(0, len(self.estimator._buffer_wheel))
    self.assertEqual(3, self.estimator._capacity)
    for _ in range(0, self.estimator._capacity):
      self.estimator.update(1, 1)
    self.assertEqual(3, len(self.estimator._buffer_wheel))
    self.assertEqual(1.0, self.estimator._mean_runtime_secs())
    self.assertEqual(1.0, self.estimator._mean_step_time_secs())
    for _ in range(0, self.estimator._capacity):
      self.estimator.update(3, 2)
    self.assertEqual(3, len(self.estimator._buffer_wheel))
    self.assertEqual(3.0, self.estimator._mean_runtime_secs())
    self.assertEqual(1.5, self.estimator._mean_step_time_secs())

  def test_partial_wheel(self):
    """Tests getting estimate when the circular buffer is not full."""
    self.assertEqual(0, self.estimator._sample_count)
    self.estimator.update(5.0, 1)
    self.assertEqual(1, self.estimator._sample_count)
    self.assertEqual(5.0, self.estimator._mean_runtime_secs())
    self.assertEqual(5.0, self.estimator._mean_step_time_secs())
    self.assertEqual(2, self.estimator.get(10))
    self.estimator.update(5.0, 1)
    self.assertEqual(2, self.estimator._sample_count)
    self.assertEqual(5.0, self.estimator._mean_runtime_secs())
    self.assertEqual(5.0, self.estimator._mean_step_time_secs())
    self.assertEqual(3, self.estimator.get(15))
    self.estimator.update(5.0, 1)
    self.assertEqual(3, self.estimator._sample_count)
    self.assertEqual(5.0, self.estimator._mean_runtime_secs())
    self.assertEqual(5.0, self.estimator._mean_step_time_secs())
    self.assertEqual(2, self.estimator.get(10))

  def test_update_convergence(self):
    """Tests iterative search convergence."""
    for _ in range(0, self.estimator._capacity):
      self.estimator.update(2.0, 4)
    self.assertEqual(2, self.estimator._mean_runtime_secs())
    self.assertEqual(0.5, self.estimator._mean_step_time_secs())

    iterations = 4
    target_elapsed_time = 10
    actual_elapsed_time = 2
    secs_per_iterations = actual_elapsed_time / iterations
    for _ in range(0, 5):
      self.estimator.update(actual_elapsed_time, iterations)
      iterations = self.estimator.get(target_elapsed_time)
      actual_elapsed_time = iterations * secs_per_iterations
    self.assertLessEqual(abs(actual_elapsed_time - target_elapsed_time), 1)


class TPUStopAtStepHookTest(test.TestCase):

  def test_invalid_parameters_on_construction(self):
    """Tests invalid parameters on construction."""
    with self.assertRaises(ValueError) as ve:
      tpu_estimator._TPUStopAtStepHook(
          util_lib.IterationsPerLoopCounter(value=10, unit='count'),
          num_steps=None,
          final_step=None)
      self.assertEqual(ve.exception.message,
                       'One of num_steps or final_step must be specified.')

    with self.assertRaises(ValueError) as ve:
      tpu_estimator._TPUStopAtStepHook(
          util_lib.IterationsPerLoopCounter(value=10, unit='count'),
          num_steps=10,
          final_step=100)
      self.assertEqual(ve.exception.message,
                       'Only one of num_steps or final_step can be specified.')

    with self.assertRaises(ValueError) as ve:
      tpu_estimator._TPUStopAtStepHook(
          util_lib.IterationsPerLoopCounter(value=10, unit='secs'),
          num_steps=10,
          final_step=100)
      self.assertEqual(
          ve.exception.message,
          'Only `count` or `seconds` are accepted as the `iterations_per_loop` '
          'unit.')

  def _validate_hook_life_cycle(self, iterations_per_loop_counter, num_steps):
    """Test execute hook life-cycle.

    This test validates:
    - Correctly updating the iterations both for `iterations_per_loop_counter`
      specified as both `count` and `seconds`
    - Terminates the session.run() by signaling termination `request_stop()`
    - The computation of the final iterations count when the remaining step
      count is smaller than the iterations_per_loop_counter.value.

    Args:
      iterations_per_loop_counter: This is the number of train steps running in
        TPU before returning to CPU host for each `Session.run`. Can be
        specified as `count` or `seconds`.
      num_steps: Number of steps to execute.
    """
    with tf.Session() as sess:
      global_step_tensor = training_util.get_or_create_global_step(sess.graph)
      global_step_tensor.load(0, session=sess)
      self.assertEqual(sess.run(global_step_tensor), 0)

      default_iterations = 1
      hook = tpu_estimator._TPUStopAtStepHook(
          iterations_per_loop_counter, num_steps=num_steps)
      self.assertEqual(default_iterations, hook._next_iteration_count)
      self.assertEqual(num_steps, hook._num_steps)
      self.assertEqual(None, hook._final_step)
      self.assertEqual(iterations_per_loop_counter.value,
                       hook._iterations_per_loop_counter.value)
      self.assertEqual(iterations_per_loop_counter.unit,
                       hook._iterations_per_loop_counter.unit)

      def _step(hook, is_final, expected_iterations):
        hook.begin()
        hook.after_create_session(sess, None)

        class RunContextMock(object):

          def __init__(self, session):
            self.session = session
            self.stop = False

          def request_stop(self):
            self.stop = True

        class RunValues(object):

          def __init__(self, elapsed_time_secs):
            self.results = {'elapsed_time': elapsed_time_secs}

        run_context = RunContextMock(sess)
        run_values = RunValues(1)
        time.sleep(1.0)
        hook.after_run(run_context, run_values)
        if is_final:
          self.assertEqual(hook._next_iteration_count, expected_iterations)
          self.assertEqual(run_context.stop, is_final)
        else:
          self.assertLessEqual(
              abs(hook._next_iteration_count - expected_iterations), 1)

      # Estimates iterations when global_step < final_step.
      global_step = sess.run(training_util.get_global_step())
      self.assertEqual(global_step, 0)
      _step(hook, is_final=False, expected_iterations=3)

      # Estimates iterations when global_step < final_step.
      global_step_tensor.load(2, session=sess)
      _step(hook, is_final=False, expected_iterations=3)

      # Estimates iterations when global_step < final_step, and
      # (final_step - global_step) < estimated-iterations.
      global_step_tensor.load(4, session=sess)
      _step(hook, is_final=False, expected_iterations=1)

      # Estimates iterations when global_step == final_step.
      global_step_tensor.load(5, session=sess)
      _step(hook, is_final=True, expected_iterations=0)

  def test_hook_life_cycle(self):
    """Tests update iterations."""
    self._validate_hook_life_cycle(
        util_lib.IterationsPerLoopCounter(value=3, unit='seconds'), 5)
    self._validate_hook_life_cycle(
        util_lib.IterationsPerLoopCounter(value=3, unit='count'), 5)

  def _validate_initialization(self, iterations_per_loop_counter, num_steps):
    with tf.Session() as sess:
      global_step_tensor = training_util.get_or_create_global_step(sess.graph)
      global_step_tensor.load(0, session=sess)
      self.assertEqual(sess.run(global_step_tensor), 0)

      hook = tpu_estimator._TPUStopAtStepHook(
          iterations_per_loop_counter, num_steps=num_steps)
      self.assertEqual(1, hook._next_iteration_count)
      self.assertEqual(num_steps, hook._num_steps)
      self.assertEqual(None, hook._final_step)
      self.assertEqual(iterations_per_loop_counter.value,
                       hook._iterations_per_loop_counter.value)
      self.assertEqual(iterations_per_loop_counter.unit,
                       hook._iterations_per_loop_counter.unit)
      if iterations_per_loop_counter.unit == 'count':
        with self.assertRaises(AttributeError) as ve:
          _ = hook.iteration_count_estimator
          self.assertIn('object has no attribute', ve.message)
      else:
        self.assertIsInstance(hook._iteration_count_estimator,
                              iteration_count_estimator.IterationCountEstimator)

  def test_initialization(self):
    """Tests initialization.

    This test validates initialization of the Hook using both specifying
    `iterations_per_loop` as raw `count` and `seconds`.
    """
    self._validate_initialization(
        util_lib.IterationsPerLoopCounter(value=3, unit='seconds'), 3)
    self._validate_initialization(
        util_lib.IterationsPerLoopCounter(value=600, unit='seconds'), 1)
    self._validate_initialization(
        util_lib.IterationsPerLoopCounter(value=3600, unit='seconds'), 5)
    self._validate_initialization(
        util_lib.IterationsPerLoopCounter(value=3, unit='count'), 100)
    self._validate_initialization(
        util_lib.IterationsPerLoopCounter(value=100, unit='count'), 10)


if __name__ == '__main__':
  test.main()
