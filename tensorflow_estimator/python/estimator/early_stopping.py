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
"""Utilities for early stopping."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.util.tf_export import estimator_export
from tensorflow_estimator.python.estimator import estimator as estimator_lib


@estimator_export('estimator.experimental.make_early_stopping_hook')
def make_early_stopping_hook(estimator,
                             should_stop_fn,
                             run_every_secs=60,
                             run_every_steps=None):
  """Creates early-stopping hook.

  Returns a `SessionRunHook` that stops training when `should_stop_fn` returns
  `True`.

  Usage example:

  ```python
  estimator = ...
  hook = early_stopping.make_early_stopping_hook(
      estimator, should_stop_fn=make_stop_fn(...))
  train_spec = tf.estimator.TrainSpec(..., hooks=[hook])
  tf.estimator.train_and_evaluate(estimator, train_spec, ...)
  ```

  Caveat: Current implementation supports early-stopping both training and
  evaluation in local mode. In distributed mode, training can be stopped but
  evaluation (where it's a separate job) will indefinitely wait for new model
  checkpoints to evaluate, so you will need other means to detect and stop it.
  Early-stopping evaluation in distributed mode requires changes in
  `train_and_evaluate` API and will be addressed in a future revision.

  Args:
    estimator: A `tf.estimator.Estimator` instance.
    should_stop_fn: `callable`, function that takes no arguments and returns a
      `bool`. If the function returns `True`, stopping will be initiated by the
      chief.
    run_every_secs: If specified, calls `should_stop_fn` at an interval of
      `run_every_secs` seconds. Defaults to 60 seconds. Either this or
      `run_every_steps` must be set.
    run_every_steps: If specified, calls `should_stop_fn` every
      `run_every_steps` steps. Either this or `run_every_secs` must be set.

  Returns:
    A `SessionRunHook` that periodically executes `should_stop_fn` and initiates
    early stopping if the function returns `True`.

  Raises:
    TypeError: If `estimator` is not of type `tf.estimator.Estimator`.
    ValueError: If both `run_every_secs` and `run_every_steps` are set.
  """
  if not isinstance(estimator, estimator_lib.Estimator):
    raise TypeError('`estimator` must have type `tf.estimator.Estimator`. '
                    'Got: {}'.format(type(estimator)))

  if run_every_secs is not None and run_every_steps is not None:
    raise ValueError('Only one of `run_every_secs` and `run_every_steps` must '
                     'be set.')

  if estimator.config.is_chief:
    return _StopOnPredicateHook(should_stop_fn, run_every_secs, run_every_steps)
  else:
    return _CheckForStoppingHook()


def _get_or_create_stop_var():
  with variable_scope.variable_scope(
      name_or_scope='signal_early_stopping',
      values=[],
      reuse=variable_scope.AUTO_REUSE):
    return variable_scope.get_variable(
        name='STOP',
        shape=[],
        dtype=dtypes.bool,
        initializer=init_ops.constant_initializer(False),
        collections=[ops.GraphKeys.GLOBAL_VARIABLES],
        trainable=False)


class _StopOnPredicateHook(session_run_hook.SessionRunHook):
  """Hook that requests stop when `should_stop_fn` returns `True`."""

  def __init__(self, should_stop_fn, run_every_secs=60, run_every_steps=None):
    if not callable(should_stop_fn):
      raise TypeError('`should_stop_fn` must be callable.')

    self._should_stop_fn = should_stop_fn
    self._timer = basic_session_run_hooks.SecondOrStepTimer(
        every_secs=run_every_secs, every_steps=run_every_steps)
    self._global_step_tensor = None
    self._stop_var = None
    self._stop_op = None

  def begin(self):
    self._global_step_tensor = training_util.get_global_step()
    self._stop_var = _get_or_create_stop_var()
    self._stop_op = state_ops.assign(self._stop_var, True)

  def before_run(self, run_context):
    del run_context
    return session_run_hook.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    global_step = run_values.results
    if self._timer.should_trigger_for_step(global_step):
      self._timer.update_last_triggered_step(global_step)
      if self._should_stop_fn():
        tf_logging.info('Requesting early stopping at global step %d',
                        global_step)
        run_context.session.run(self._stop_op)
        run_context.request_stop()


class _CheckForStoppingHook(session_run_hook.SessionRunHook):
  """Hook that requests stop if stop is requested by `_StopOnPredicateHook`."""

  def __init__(self):
    self._stop_var = None

  def begin(self):
    self._stop_var = _get_or_create_stop_var()

  def before_run(self, run_context):
    del run_context
    return session_run_hook.SessionRunArgs(self._stop_var)

  def after_run(self, run_context, run_values):
    should_early_stop = run_values.results
    if should_early_stop:
      tf_logging.info('Early stopping requested, suspending run.')
      run_context.request_stop()
