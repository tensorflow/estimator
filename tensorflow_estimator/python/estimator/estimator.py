# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Base Estimator class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import os
import tempfile

import numpy as np
import six

from google.protobuf import message
from tensorflow.core.framework import summary_pb2
from tensorflow.python.client import session as tf_session
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.summary import summary
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import device_setter
from tensorflow.python.training import evaluation
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.training import warm_starting_util
from tensorflow.python.util import compat
from tensorflow.python.util import compat_internal
from tensorflow.python.util import function_utils
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import estimator_export
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator import util as estimator_util
from tensorflow_estimator.python.estimator.export import export as export_helpers


_VALID_MODEL_FN_ARGS = set(
    ['features', 'labels', 'mode', 'params', 'self', 'config'])


@estimator_export('estimator.Estimator', v1=[])
class EstimatorV2(object):
  """Estimator class to train and evaluate TensorFlow models.

  The `Estimator` object wraps a model which is specified by a `model_fn`,
  which, given inputs and a number of other parameters, returns the ops
  necessary to perform training, evaluation, or predictions.

  All outputs (checkpoints, event files, etc.) are written to `model_dir`, or a
  subdirectory thereof. If `model_dir` is not set, a temporary directory is
  used.

  The `config` argument can be passed `tf.estimator.RunConfig` object containing
  information about the execution environment. It is passed on to the
  `model_fn`, if the `model_fn` has a parameter named "config" (and input
  functions in the same manner). If the `config` parameter is not passed, it is
  instantiated by the `Estimator`. Not passing config means that defaults useful
  for local execution are used. `Estimator` makes config available to the model
  (for instance, to allow specialization based on the number of workers
  available), and also uses some of its fields to control internals, especially
  regarding checkpointing.

  The `params` argument contains hyperparameters. It is passed to the
  `model_fn`, if the `model_fn` has a parameter named "params", and to the input
  functions in the same manner. `Estimator` only passes params along, it does
  not inspect it. The structure of `params` is therefore entirely up to the
  developer.

  None of `Estimator`'s methods can be overridden in subclasses (its
  constructor enforces this). Subclasses should use `model_fn` to configure
  the base class, and may add methods implementing specialized functionality.

  @compatibility(eager)
  Calling methods of `Estimator` will work while eager execution is enabled.
  However, the `model_fn` and `input_fn` is not executed eagerly, `Estimator`
  will switch to graph mode before calling all user-provided functions (incl.
  hooks), so their code has to be compatible with graph mode execution. Note
  that `input_fn` code using `tf.data` generally works in both graph and eager
  modes.
  @end_compatibility
  """

  def __init__(self, model_fn, model_dir=None, config=None, params=None,
               warm_start_from=None):
    """Constructs an `Estimator` instance.

    See [estimators](https://tensorflow.org/guide/estimators) for more
    information.

    To warm-start an `Estimator`:

    ```python
    estimator = tf.estimator.DNNClassifier(
        feature_columns=[categorical_feature_a_emb, categorical_feature_b_emb],
        hidden_units=[1024, 512, 256],
        warm_start_from="/path/to/checkpoint/dir")
    ```

    For more details on warm-start configuration, see
    `tf.estimator.WarmStartSettings`.

    Args:
      model_fn: Model function. Follows the signature:

        * Args:

          * `features`: This is the first item returned from the `input_fn`
                 passed to `train`, `evaluate`, and `predict`. This should be a
                 single `tf.Tensor` or `dict` of same.
          * `labels`: This is the second item returned from the `input_fn`
                 passed to `train`, `evaluate`, and `predict`. This should be a
                 single `tf.Tensor` or `dict` of same (for multi-head models).
                 If mode is `tf.estimator.ModeKeys.PREDICT`, `labels=None` will
                 be passed. If the `model_fn`'s signature does not accept
                 `mode`, the `model_fn` must still be able to handle
                 `labels=None`.
          * `mode`: Optional. Specifies if this training, evaluation or
                 prediction. See `tf.estimator.ModeKeys`.
          * `params`: Optional `dict` of hyperparameters.  Will receive what
                 is passed to Estimator in `params` parameter. This allows
                 to configure Estimators from hyper parameter tuning.
          * `config`: Optional `estimator.RunConfig` object. Will receive what
                 is passed to Estimator as its `config` parameter, or a default
                 value. Allows setting up things in your `model_fn` based on
                 configuration such as `num_ps_replicas`, or `model_dir`.

        * Returns:
          `tf.estimator.EstimatorSpec`

      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into an estimator to
        continue training a previously saved model. If `PathLike` object, the
        path will be resolved. If `None`, the model_dir in `config` will be used
        if set. If both are set, they must be same. If both are `None`, a
        temporary directory will be used.
      config: `estimator.RunConfig` configuration object.
      params: `dict` of hyper parameters that will be passed into `model_fn`.
              Keys are names of parameters, values are basic python types.
      warm_start_from: Optional string filepath to a checkpoint or SavedModel to
                       warm-start from, or a `tf.estimator.WarmStartSettings`
                       object to fully configure warm-starting.  If the string
                       filepath is provided instead of a
                       `tf.estimator.WarmStartSettings`, then all variables are
                       warm-started, and it is assumed that vocabularies
                       and `tf.Tensor` names are unchanged.

    Raises:
      ValueError: parameters of `model_fn` don't match `params`.
      ValueError: if this is called via a subclass and if that class overrides
        a member of `Estimator`.
    """
    # We do not endorse Estimator child classes to override methods in
    # Estimator, other than a select few. You're on your own if you cleverly
    # override the method "_assert_members_are_not_overridden".
    self.__class__._assert_members_are_not_overridden(self)  # pylint: disable=protected-access

    self._config = maybe_overwrite_model_dir_and_session_config(config,
                                                                model_dir)

    # The distribute field contains an instance of tf.distribute.Strategy.
    self._train_distribution = self._config.train_distribute
    self._eval_distribution = self._config.eval_distribute
    # Model directory.
    self._model_dir = self._config.model_dir
    self._session_config = self._config.session_config
    logging.info('Using config: %s', str(vars(self._config)))

    self._device_fn = (
        self._config.device_fn or _get_replica_device_setter(self._config))

    if model_fn is None:
      raise ValueError('model_fn must be provided to Estimator.')
    _verify_model_fn_args(model_fn, params)
    self._model_fn = model_fn
    self._params = copy.deepcopy(params or {})

    # pylint: disable=protected-access
    self._warm_start_settings = _get_default_warm_start_settings(
        warm_start_from)
    # pylint: enable=protected-access

  @property
  def model_dir(self):
    return self._model_dir

  @property
  def config(self):
    return copy.deepcopy(self._config)

  @property
  def params(self):
    return copy.deepcopy(self._params)

  @property
  def model_fn(self):
    """Returns the `model_fn` which is bound to `self.params`.

    Returns:
      The `model_fn` with following signature:
        `def model_fn(features, labels, mode, config)`
    """

    def public_model_fn(features, labels, mode, config):
      return self._call_model_fn(features, labels, mode, config)

    return public_model_fn

  # TODO(ispir): support a list of names
  def get_variable_value(self, name):
    """Returns value of the variable given by name.

    Args:
      name: string or a list of string, name of the tensor.

    Returns:
      Numpy array - value of the tensor.

    Raises:
      ValueError: If the `Estimator` has not produced a checkpoint yet.
    """
    _check_checkpoint_available(self.model_dir)
    with context.graph_mode():
      return training.load_variable(self.model_dir, name)

  def get_variable_names(self):
    """Returns list of all variable names in this model.

    Returns:
      List of names.

    Raises:
      ValueError: If the `Estimator` has not produced a checkpoint yet.
    """
    _check_checkpoint_available(self.model_dir)
    with context.graph_mode():
      return [name for name, _ in training.list_variables(self.model_dir)]

  def latest_checkpoint(self):
    """Finds the filename of the latest saved checkpoint file in `model_dir`.

    Returns:
      The full path to the latest checkpoint or `None` if no checkpoint was
      found.
    """
    with context.graph_mode():
      return checkpoint_management.latest_checkpoint(self.model_dir)

  def train(self,
            input_fn,
            hooks=None,
            steps=None,
            max_steps=None,
            saving_listeners=None):
    """Trains a model given training data `input_fn`.

    Args:
      input_fn: A function that provides input data for training as minibatches.
        See [Premade Estimators](
        https://tensorflow.org/guide/premade_estimators#create_input_functions)
        for more information. The function should construct and return one of
        the following:  * A
        `tf.data.Dataset` object: Outputs of `Dataset` object must be a tuple
        `(features, labels)` with same constraints as below. * A tuple
        `(features, labels)`: Where `features` is a `tf.Tensor` or a dictionary
        of string feature name to `Tensor` and `labels` is a `Tensor` or a
        dictionary of string label name to `Tensor`. Both `features` and
        `labels` are consumed by `model_fn`. They should satisfy the expectation
        of `model_fn` from inputs.
      hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
        callbacks inside the training loop.
      steps: Number of steps for which to train the model. If `None`, train
        forever or train until `input_fn` generates the `tf.errors.OutOfRange`
        error or `StopIteration` exception. `steps` works incrementally. If you
        call two times `train(steps=10)` then training occurs in total 20 steps.
        If `OutOfRange` or `StopIteration` occurs in the middle, training stops
        before 20 steps. If you don't want to have incremental behavior please
        set `max_steps` instead. If set, `max_steps` must be `None`.
      max_steps: Number of total steps for which to train model. If `None`,
        train forever or train until `input_fn` generates the
        `tf.errors.OutOfRange` error or `StopIteration` exception. If set,
        `steps` must be `None`. If `OutOfRange` or `StopIteration` occurs in the
        middle, training stops before `max_steps` steps. Two calls to
        `train(steps=100)` means 200 training iterations. On the other hand, two
        calls to `train(max_steps=100)` means that the second call will not do
        any iteration since first call did all 100 steps.
      saving_listeners: list of `CheckpointSaverListener` objects. Used for
        callbacks that run immediately before or after checkpoint savings.

    Returns:
      `self`, for chaining.

    Raises:
      ValueError: If both `steps` and `max_steps` are not `None`.
      ValueError: If either `steps` or `max_steps <= 0`.
    """
    if self.config.task_type in (run_config.TaskType.EVALUATOR,
                                 run_config.TaskType.PS):
      raise ValueError(
          'Train has been called wrong configuration. Please use '
          'tf.estimator.train_and_evaluate which calls proper API according '
          'to given configuration. Current configuration: {}.'.format(
              self.config))

    with context.graph_mode():
      if (steps is not None) and (max_steps is not None):
        raise ValueError('Can not provide both steps and max_steps.')
      if steps is not None and steps <= 0:
        raise ValueError('Must specify steps > 0, given: {}'.format(steps))
      if max_steps is not None and max_steps <= 0:
        raise ValueError(
            'Must specify max_steps > 0, given: {}'.format(max_steps))

      if max_steps is not None:
        start_step = _load_global_step_from_checkpoint_dir(self._model_dir)
        if max_steps <= start_step:
          logging.info('Skipping training since max_steps has already saved.')
          return self

      hooks = _check_hooks_type(hooks)
      hooks.extend(self._convert_train_steps_to_hooks(steps, max_steps))

      saving_listeners = _check_listeners_type(saving_listeners)
      loss = self._train_model(input_fn, hooks, saving_listeners)
      logging.info('Loss for final step: %s.', loss)
      return self

  def _convert_train_steps_to_hooks(self, steps, max_steps):
    """Create hooks to run correct number of steps in training.

    Args:
      steps: number of steps to run during training.
      max_steps: maximum number of steps to be run during training. It'll be
        the maximum number of steps the model will train to after restoring
        from checkpoint even across multiple estimator.train calls.

    Returns:
      List of hooks to be passed to the estimator.
    """
    if steps is not None or max_steps is not None:
      if self._train_distribution:
        steps_per_run = getattr(
            self._train_distribution.extended, 'steps_per_run', 1)
        if steps_per_run > 1:
          return [basic_session_run_hooks._MultiStepStopAtStepHook(  # pylint: disable=protected-access
              steps, max_steps, steps_per_run)]
      return [training.StopAtStepHook(steps, max_steps)]
    else:
      return []

  def eval_dir(self, name=None):
    """Shows the directory name where evaluation metrics are dumped.

    Args:
      name: Name of the evaluation if user needs to run multiple evaluations on
        different data sets, such as on training data vs test data. Metrics for
        different evaluations are saved in separate folders, and appear
        separately in tensorboard.

    Returns:
      A string which is the path of directory contains evaluation metrics.
    """
    return os.path.join(self._model_dir, 'eval' if not name else
                        'eval_' + name)

  def evaluate(self, input_fn, steps=None, hooks=None, checkpoint_path=None,
               name=None):
    """Evaluates the model given evaluation data `input_fn`.

    For each step, calls `input_fn`, which returns one batch of data.
    Evaluates until:
    - `steps` batches are processed, or
    - `input_fn` raises an end-of-input exception (`tf.errors.OutOfRangeError`
    or
    `StopIteration`).

    Args:
      input_fn: A function that constructs the input data for evaluation. See
        [Premade Estimators](
        https://tensorflow.org/guide/premade#create_input_functions)
        for more information. The
        function should construct and return one of the following:  * A
        `tf.data.Dataset` object: Outputs of `Dataset` object must be a tuple
        `(features, labels)` with same constraints as below. * A tuple
        `(features, labels)`: Where `features` is a `tf.Tensor` or a dictionary
        of string feature name to `Tensor` and `labels` is a `Tensor` or a
        dictionary of string label name to `Tensor`. Both `features` and
        `labels` are consumed by `model_fn`. They should satisfy the expectation
        of `model_fn` from inputs.
      steps: Number of steps for which to evaluate model. If `None`, evaluates
        until `input_fn` raises an end-of-input exception.
      hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
        callbacks inside the evaluation call.
      checkpoint_path: Path of a specific checkpoint to evaluate. If `None`, the
        latest checkpoint in `model_dir` is used.  If there are no checkpoints
        in `model_dir`, evaluation is run with newly initialized `Variables`
        instead of ones restored from checkpoint.
      name: Name of the evaluation if user needs to run multiple evaluations on
        different data sets, such as on training data vs test data. Metrics for
        different evaluations are saved in separate folders, and appear
        separately in tensorboard.

    Returns:
      A dict containing the evaluation metrics specified in `model_fn` keyed by
      name, as well as an entry `global_step` which contains the value of the
      global step for which this evaluation was performed. For canned
      estimators, the dict contains the `loss` (mean loss per mini-batch) and
      the `average_loss` (mean loss per sample). Canned classifiers also return
      the `accuracy`. Canned regressors also return the `label/mean` and the
      `prediction/mean`.

    Raises:
      ValueError: If `steps <= 0`.
      ValueError: If no model has been trained, namely `model_dir`, or the
        given `checkpoint_path` is empty.
    """
    # pylint: disable=protected-access
    if (self._eval_distribution and
        hasattr(self._config, '_distribute_coordinator_mode') and
        self._config._distribute_coordinator_mode):
      return distribute_coordinator_training.estimator_evaluate(
          self,
          lambda est, s, eval_hooks: est._actual_eval(  # pylint: disable=g-long-lambda
              input_fn, strategy=s, steps=steps, hooks=eval_hooks,
              checkpoint_path=checkpoint_path, name=name),
          hooks)
    # pylint: enable=protected-access
    else:
      return self._actual_eval(
          input_fn,
          strategy=self._eval_distribution,
          steps=steps,
          hooks=hooks,
          checkpoint_path=checkpoint_path,
          name=name)

  def _actual_eval(self,
                   input_fn,
                   strategy=None,
                   steps=None,
                   hooks=None,
                   checkpoint_path=None,
                   name=None):
    """The method that does evaluation actually."""
    with context.graph_mode():
      hooks = _check_hooks_type(hooks)
      hooks.extend(self._convert_eval_steps_to_hooks(steps))

      # Check that model has been trained (if nothing has been set explicitly).
      if not checkpoint_path:
        latest_path = checkpoint_management.latest_checkpoint(self._model_dir)
        if not latest_path:
          logging.info('Could not find trained model in model_dir: {}, running '
                       'initialization to evaluate.'.format(self._model_dir))
        checkpoint_path = latest_path

      def _evaluate():
        (scaffold, update_op, eval_dict, all_hooks) = (
            self._evaluate_build_graph(input_fn, hooks, checkpoint_path))
        return self._evaluate_run(
            checkpoint_path=checkpoint_path,
            scaffold=scaffold,
            update_op=update_op,
            eval_dict=eval_dict,
            all_hooks=all_hooks,
            output_dir=self.eval_dir(name))

      with ops.Graph().as_default():
        if strategy:
          # We want to create the iterations variable outside the distribution
          # scope as that is just stored on the host and mainly used to drive
          # the loop and doesn't need to be a Mirrored/Device variable.
          training.get_or_create_steps_per_run_variable()
          with strategy.scope():
            return _evaluate()
        else:
          return _evaluate()

  def _convert_eval_steps_to_hooks(self, steps):
    """Create hooks to run correct number of steps in evaluation.

    Args:
      steps: number of steps to run during evaluation.

    Raises:
      ValueError: if steps is less than or equal to zero.

    Returns:
      List of hooks to be passed to the estimator.
    """
    if steps is None:
      return []

    if steps <= 0:
      raise ValueError('Must specify steps > 0, given: {}'.format(steps))

    # The hooks are declared as private in evaluation.py discourage the use
    # by other libraries or open source users. This should be the only usage
    # of the estimator evaluation hooks.
    if self._eval_distribution:
      steps_per_run = getattr(
          self._eval_distribution.extended, 'steps_per_run', 1)
      if steps_per_run > 1:
        return [evaluation._MultiStepStopAfterNEvalsHook(  # pylint: disable=protected-access
            num_evals=steps, steps_per_run=steps_per_run)]
    return [evaluation._StopAfterNEvalsHook(num_evals=steps)]  # pylint: disable=protected-access

  def predict(self,
              input_fn,
              predict_keys=None,
              hooks=None,
              checkpoint_path=None,
              yield_single_examples=True):
    """Yields predictions for given features.

    Please note that interleaving two predict outputs does not work. See:
    [issue/20506](
    https://github.com/tensorflow/tensorflow/issues/20506#issuecomment-422208517)

    Args:
      input_fn: A function that constructs the features. Prediction continues
        until `input_fn` raises an end-of-input exception
        (`tf.errors.OutOfRangeError` or `StopIteration`).
        See [Premade Estimators](
        https://tensorflow.org/guide/premade_estimators#create_input_functions)
        for more information. The function should construct and return one of
        the following:

          * A `tf.data.Dataset` object: Outputs of `Dataset` object must have
            same constraints as below.
          * features: A `tf.Tensor` or a dictionary of string feature name to
            `Tensor`. features are consumed by `model_fn`. They should satisfy
            the expectation of `model_fn` from inputs.
          * A tuple, in which case the first item is extracted as features.

      predict_keys: list of `str`, name of the keys to predict. It is used if
        the `tf.estimator.EstimatorSpec.predictions` is a `dict`. If
        `predict_keys` is used then rest of the predictions will be filtered
        from the dictionary. If `None`, returns all.
      hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
        callbacks inside the prediction call.
      checkpoint_path: Path of a specific checkpoint to predict. If `None`, the
        latest checkpoint in `model_dir` is used.  If there are no checkpoints
        in `model_dir`, prediction is run with newly initialized `Variables`
        instead of ones restored from checkpoint.
      yield_single_examples: If `False`, yields the whole batch as returned by
        the `model_fn` instead of decomposing the batch into individual
        elements. This is useful if `model_fn` returns some tensors whose first
        dimension is not equal to the batch size.

    Yields:
      Evaluated values of `predictions` tensors.

    Raises:
      ValueError: Could not find a trained model in `model_dir`.
      ValueError: If batch length of predictions is not the same and
        `yield_single_examples` is `True`.
      ValueError: If there is a conflict between `predict_keys` and
        `predictions`. For example if `predict_keys` is not `None` but
        `tf.estimator.EstimatorSpec.predictions` is not a `dict`.
    """
    with context.graph_mode():
      hooks = _check_hooks_type(hooks)
      # Check that model has been trained.
      if not checkpoint_path:
        checkpoint_path = checkpoint_management.latest_checkpoint(
            self._model_dir)
      if not checkpoint_path:
        logging.info('Could not find trained model in model_dir: {}, running '
                     'initialization to predict.'.format(self._model_dir))
      with ops.Graph().as_default() as g:
        random_seed.set_random_seed(self._config.tf_random_seed)
        self._create_and_assert_global_step(g)
        features, input_hooks = self._get_features_from_input_fn(
            input_fn, model_fn_lib.ModeKeys.PREDICT)
        estimator_spec = self._call_model_fn(
            features, None, model_fn_lib.ModeKeys.PREDICT, self.config)

        # Call to warm_start has to be after model_fn is called.
        self._maybe_warm_start(checkpoint_path)

        predictions = self._extract_keys(
            estimator_spec.predictions, predict_keys)
        all_hooks = list(input_hooks)
        all_hooks.extend(hooks)
        all_hooks.extend(list(estimator_spec.prediction_hooks or []))
        with training.MonitoredSession(
            session_creator=training.ChiefSessionCreator(
                checkpoint_filename_with_path=checkpoint_path,
                master=self._config.master,
                scaffold=estimator_spec.scaffold,
                config=self._session_config),
            hooks=all_hooks) as mon_sess:
          while not mon_sess.should_stop():
            preds_evaluated = mon_sess.run(predictions)
            if not yield_single_examples:
              yield preds_evaluated
            elif not isinstance(predictions, dict):
              for pred in preds_evaluated:
                yield pred
            else:
              for i in range(self._extract_batch_length(preds_evaluated)):
                yield {
                    key: value[i]
                    for key, value in six.iteritems(preds_evaluated)
                }

  def _assert_members_are_not_overridden(self):
    """Asserts members of `Estimator` are not overridden."""
    _assert_members_are_not_overridden(EstimatorV2, self)

  def export_saved_model(
      self, export_dir_base, serving_input_receiver_fn,
      assets_extra=None,
      as_text=False,
      checkpoint_path=None,
      experimental_mode=model_fn_lib.ModeKeys.PREDICT):
    # pylint: disable=line-too-long
    """Exports inference graph as a `SavedModel` into the given dir.

    For a detailed guide, see
    [Using SavedModel with Estimators](https://tensorflow.org/guide/saved_model#using_savedmodel_with_estimators).

    This method builds a new graph by first calling the
    `serving_input_receiver_fn` to obtain feature `Tensor`s, and then calling
    this `Estimator`'s `model_fn` to generate the model graph based on those
    features. It restores the given checkpoint (or, lacking that, the most
    recent checkpoint) into this graph in a fresh session.  Finally it creates
    a timestamped export directory below the given `export_dir_base`, and writes
    a `SavedModel` into it containing a single `tf.MetaGraphDef` saved from this
    session.

    The exported `MetaGraphDef` will provide one `SignatureDef` for each
    element of the `export_outputs` dict returned from the `model_fn`, named
    using
    the same keys.  One of these keys is always
    `tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY`,
    indicating which
    signature will be served when a serving request does not specify one.
    For each signature, the outputs are provided by the corresponding
    `tf.estimator.export.ExportOutput`s, and the inputs are always the input
    receivers provided by
    the `serving_input_receiver_fn`.

    Extra assets may be written into the `SavedModel` via the `assets_extra`
    argument.  This should be a dict, where each key gives a destination path
    (including the filename) relative to the assets.extra directory.  The
    corresponding value gives the full path of the source file to be copied.
    For example, the simple case of copying a single file without renaming it
    is specified as `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.

    The experimental_mode parameter can be used to export a single
    train/eval/predict graph as a `SavedModel`.
    See `experimental_export_all_saved_models` for full docs.

    Args:
      export_dir_base: A string containing a directory in which to create
        timestamped subdirectories containing exported `SavedModel`s.
      serving_input_receiver_fn: A function that takes no argument and returns a
        `tf.estimator.export.ServingInputReceiver` or
        `tf.estimator.export.TensorServingInputReceiver`.
      assets_extra: A dict specifying how to populate the assets.extra directory
        within the exported `SavedModel`, or `None` if no extra assets are
        needed.
      as_text: whether to write the `SavedModel` proto in text format.
      checkpoint_path: The checkpoint path to export.  If `None` (the default),
        the most recent checkpoint found within the model directory is chosen.
      experimental_mode: `tf.estimator.ModeKeys` value indicating with mode
        will be exported. Note that this feature is experimental.

    Returns:
      The string path to the exported directory.

    Raises:
      ValueError: if no `serving_input_receiver_fn` is provided, no
      `export_outputs` are provided, or no checkpoint can be found.
    """
    # pylint: enable=line-too-long
    if not serving_input_receiver_fn:
      raise ValueError('An input_receiver_fn must be defined.')

    input_receiver_fn_map = {experimental_mode: serving_input_receiver_fn}

    return self.experimental_export_all_saved_models(
        export_dir_base,
        input_receiver_fn_map,
        assets_extra=assets_extra,
        as_text=as_text,
        checkpoint_path=checkpoint_path)

  def experimental_export_all_saved_models(
      self, export_dir_base, input_receiver_fn_map,
      assets_extra=None,
      as_text=False,
      checkpoint_path=None):
    """Exports a `SavedModel` with `tf.MetaGraphDefs` for each requested mode.

    For each mode passed in via the `input_receiver_fn_map`,
    this method builds a new graph by calling the `input_receiver_fn` to obtain
    feature and label `Tensor`s. Next, this method calls the `Estimator`'s
    `model_fn` in the passed mode to generate the model graph based on
    those features and labels, and restores the given checkpoint
    (or, lacking that, the most recent checkpoint) into the graph.
    Only one of the modes is used for saving variables to the `SavedModel`
    (order of preference: `tf.estimator.ModeKeys.TRAIN`,
    `tf.estimator.ModeKeys.EVAL`, then
    `tf.estimator.ModeKeys.PREDICT`), such that up to three
    `tf.MetaGraphDefs` are saved with a single set of variables in a single
    `SavedModel` directory.

    For the variables and `tf.MetaGraphDefs`, a timestamped export directory
    below
    `export_dir_base`, and writes a `SavedModel` into it containing
    the `tf.MetaGraphDef` for the given mode and its associated signatures.

    For prediction, the exported `MetaGraphDef` will provide one `SignatureDef`
    for each element of the `export_outputs` dict returned from the `model_fn`,
    named using the same keys.  One of these keys is always
    `tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY`,
    indicating which
    signature will be served when a serving request does not specify one.
    For each signature, the outputs are provided by the corresponding
    `tf.estimator.export.ExportOutput`s, and the inputs are always the input
    receivers provided by
    the `serving_input_receiver_fn`.

    For training and evaluation, the `train_op` is stored in an extra
    collection,
    and loss, metrics, and predictions are included in a `SignatureDef` for the
    mode in question.

    Extra assets may be written into the `SavedModel` via the `assets_extra`
    argument.  This should be a dict, where each key gives a destination path
    (including the filename) relative to the assets.extra directory.  The
    corresponding value gives the full path of the source file to be copied.
    For example, the simple case of copying a single file without renaming it
    is specified as `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.

    Args:
      export_dir_base: A string containing a directory in which to create
        timestamped subdirectories containing exported `SavedModel`s.
      input_receiver_fn_map: dict of `tf.estimator.ModeKeys` to
        `input_receiver_fn` mappings, where the `input_receiver_fn` is a
        function that takes no arguments and returns the appropriate subclass of
        `InputReceiver`.
      assets_extra: A dict specifying how to populate the assets.extra directory
        within the exported `SavedModel`, or `None` if no extra assets are
        needed.
      as_text: whether to write the `SavedModel` proto in text format.
      checkpoint_path: The checkpoint path to export.  If `None` (the default),
        the most recent checkpoint found within the model directory is chosen.

    Returns:
      The string path to the exported directory.

    Raises:
      ValueError: if any `input_receiver_fn` is `None`, no `export_outputs`
        are provided, or no checkpoint can be found.
    """
    # TODO(b/65561022): Consider allowing multiple input_receiver_fns per mode.
    with context.graph_mode():
      if not checkpoint_path:
        # Locate the latest checkpoint
        checkpoint_path = checkpoint_management.latest_checkpoint(
            self._model_dir)
      if not checkpoint_path:
        raise ValueError("Couldn't find trained model at %s." % self._model_dir)

      final_export_dir = export_helpers.get_timestamped_export_dir(export_dir_base)
      if gfile.NeedsTempLocation(final_export_dir):
        current_export_dir = export_helpers.get_temp_export_dir(final_export_dir)

      builder = saved_model_builder.SavedModelBuilder(current_export_dir)

      save_variables = True
      # Note that the order in which we run here matters, as the first
      # mode we pass through will be used to save the variables. We run TRAIN
      # first, as that is also the mode used for checkpoints, and therefore
      # we are not likely to have vars in PREDICT that are not in the checkpoint
      # created by TRAIN.
      if input_receiver_fn_map.get(model_fn_lib.ModeKeys.TRAIN):
        self._add_meta_graph_for_mode(
            builder, input_receiver_fn_map, checkpoint_path,
            save_variables, mode=model_fn_lib.ModeKeys.TRAIN)
        save_variables = False
      if input_receiver_fn_map.get(model_fn_lib.ModeKeys.EVAL):
        self._add_meta_graph_for_mode(
            builder, input_receiver_fn_map, checkpoint_path,
            save_variables, mode=model_fn_lib.ModeKeys.EVAL)
        save_variables = False
      if input_receiver_fn_map.get(model_fn_lib.ModeKeys.PREDICT):
        self._add_meta_graph_for_mode(
            builder, input_receiver_fn_map, checkpoint_path,
            save_variables, mode=model_fn_lib.ModeKeys.PREDICT)
        save_variables = False

      if save_variables:
        raise ValueError('No valid modes for exporting found. Got {}.'.format(
            input_receiver_fn_map.keys()))

      builder.save(as_text)

      # Add the extra assets
      if assets_extra:
        assets_extra_path = os.path.join(compat.as_bytes(current_export_dir),
                                         compat.as_bytes('assets.extra'))
        for dest_relative, source in assets_extra.items():
          dest_absolute = os.path.join(compat.as_bytes(assets_extra_path),
                                       compat.as_bytes(dest_relative))
          dest_path = os.path.dirname(dest_absolute)
          gfile.MakeDirs(dest_path)
          gfile.Copy(source, dest_absolute)

      if gfile.NeedsTempLocation(final_export_dir):
        gfile.Rename(current_export_dir, final_export_dir)
      return final_export_dir

  def _add_meta_graph_for_mode(self,
                               builder,
                               input_receiver_fn_map,
                               checkpoint_path,
                               save_variables=True,
                               mode=model_fn_lib.ModeKeys.PREDICT,
                               export_tags=None,
                               check_variables=True):
    """Loads variables and adds them along with a `tf.MetaGraphDef` for saving.

    Args:
      builder: instance of `tf.saved_modle.builder.SavedModelBuilder` that will
        be used for saving.
      input_receiver_fn_map: dict of `tf.estimator.ModeKeys` to
        `input_receiver_fn` mappings, where the `input_receiver_fn` is a
        function that takes no argument and returns the appropriate subclass of
        `InputReceiver`.
      checkpoint_path: The checkpoint path to export.  If `None` (the default),
        the most recent checkpoint found within the model directory is chosen.
      save_variables: bool, whether variables should be saved. If `False`, just
        the `tf.MetaGraphDef` will be saved. Note that `save_variables` should
        only be `True` for the first call to this function, and the
        `SavedModelBuilder` will raise an error if that is not the case.
      mode: `tf.estimator.ModeKeys` value indicating which mode will be
        exported.
      export_tags: The set of tags with which to save `tf.MetaGraphDef`. If
        `None`, a default set will be selected to matched the passed mode.
      check_variables: bool, whether to check the checkpoint has all variables.

    Raises:
      ValueError: if `save_variables` is `True` and `check_variable` is `False`.
    """
    if export_tags is None:
      export_tags = model_fn_lib.EXPORT_TAG_MAP[mode]
    input_receiver_fn = input_receiver_fn_map[mode]

    with ops.Graph().as_default() as g:
      self._create_and_assert_global_step(g)
      random_seed.set_random_seed(self._config.tf_random_seed)

      input_receiver = input_receiver_fn()

      # Call the model_fn and collect the export_outputs.
      estimator_spec = self._call_model_fn(
          features=input_receiver.features,
          labels=getattr(input_receiver, 'labels', None),
          mode=mode,
          config=self.config)

      export_outputs = model_fn_lib.export_outputs_for_mode(
          mode=estimator_spec.mode,
          serving_export_outputs=estimator_spec.export_outputs,
          predictions=estimator_spec.predictions,
          loss=estimator_spec.loss,
          metrics=estimator_spec.eval_metric_ops)

      # Build the SignatureDefs from receivers and all outputs
      signature_def_map = export_helpers.build_all_signature_defs(
          input_receiver.receiver_tensors,
          export_outputs,
          getattr(input_receiver, 'receiver_tensors_alternatives', None),
          serving_only=(mode == model_fn_lib.ModeKeys.PREDICT))

      with tf_session.Session(config=self._session_config) as session:

        if estimator_spec.scaffold.local_init_op is not None:
          local_init_op = estimator_spec.scaffold.local_init_op
        else:
          local_init_op = monitored_session.Scaffold.default_local_init_op()

        # This saver will be used both for restoring variables now,
        # and in saving out the metagraph below. This ensures that any
        # Custom Savers stored with the Scaffold are passed through to the
        # SavedModel for restore later.
        graph_saver = estimator_spec.scaffold.saver or saver.Saver(sharded=True)

        if save_variables and not check_variables:
          raise ValueError('If `save_variables` is `True, `check_variables`'
                           'must not be `False`.')
        if check_variables:
          try:
            graph_saver.restore(session, checkpoint_path)
          except errors.NotFoundError as e:
            msg = ('Could not load all requested variables from checkpoint. '
                   'Please make sure your model_fn does not expect variables '
                   'that were not saved in the checkpoint.\n\n'
                   'Encountered error with mode `{}` while restoring '
                   'checkpoint from: `{}`. Full Traceback:\n\n{}').format(
                       mode, checkpoint_path, e)
            raise ValueError(msg)

        # We add the train op explicitly for now, so that we don't have to
        # change the Builder public interface. Note that this is a no-op
        # for prediction, where train_op is None.
        builder._add_train_op(estimator_spec.train_op)  # pylint: disable=protected-access

        meta_graph_kwargs = dict(
            tags=export_tags,
            signature_def_map=signature_def_map,
            assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
            main_op=local_init_op,
            saver=graph_saver,
            strip_default_attrs=True)

        self._call_add_meta_graph_and_variables(
            save_variables, builder, session, meta_graph_kwargs)

  def _call_add_meta_graph_and_variables(
      self, save_variables, builder, session, kwargs):
    """Convenience wrapper so we can swap out v1 and v2 behavior."""
    if save_variables:
      builder.add_meta_graph_and_variables(session, **kwargs)
    else:
      builder.add_meta_graph(**kwargs)

  def _get_features_from_input_fn(self, input_fn, mode):
    """Extracts the `features` from return values of `input_fn`."""
    result = self._call_input_fn(input_fn, mode)
    result, _, hooks = estimator_util.parse_input_fn_result(result)
    self._validate_features_in_predict_input(result)
    return result, hooks

  def _validate_features_in_predict_input(self, result):
    if not _has_dataset_or_queue_runner(result):
      logging.warning('Input graph does not use tf.data.Dataset or contain a '
                      'QueueRunner. That means predict yields forever. '
                      'This is probably a mistake.')

  def _get_iterator_from_input_fn(self, input_fn, mode, distribution=None):
    if distribution is not None:
      iterator = distribution.make_input_fn_iterator(
          lambda _: self._call_input_fn(input_fn, mode))
      input_hooks = [
          estimator_util.DistributedIteratorInitializerHook(iterator)]
    else:
      result = self._call_input_fn(input_fn, mode)
      iterator = result.make_initializable_iterator()
      input_hooks = [estimator_util._DatasetInitializerHook(iterator)]  # pylint: disable=protected-access
    return iterator, input_hooks

  def _get_features_and_labels_from_input_fn(self, input_fn, mode):
    """Extracts the `features` and labels from return values of `input_fn`."""
    return estimator_util.parse_input_fn_result(
        self._call_input_fn(input_fn, mode))

  def _extract_batch_length(self, preds_evaluated):
    """Extracts batch length of predictions."""
    batch_length = None
    for key, value in six.iteritems(preds_evaluated):
      batch_length = batch_length or value.shape[0]
      if value.shape[0] != batch_length:
        raise ValueError('Batch length of predictions should be same. %s has '
                         'different batch length than others.' % key)
    return batch_length

  def _extract_keys(self, predictions, predict_keys):
    """Extracts `predict_keys` from `predictions`."""
    if not predict_keys:
      return predictions
    if not isinstance(predictions, dict):
      raise ValueError(
          'predict_keys argument is not valid in case of non-dict predictions.')
    existing_keys = predictions.keys()
    predictions = {
        key: value
        for key, value in six.iteritems(predictions) if key in predict_keys
    }
    if not predictions:
      raise ValueError('Expected to run at least one output from %s, '
                       'provided %s.' % (existing_keys, predict_keys))
    return predictions

  def _create_global_step(self, graph):
    """Creates the global step tensor in graph.

    The global step tensor must be an integer type with name 'global_step' and
    be added to the collection `tf.GraphKeys.GLOBAL_STEP`.

    Args:
      graph: The graph in which to create the global step tensor.

    Returns:
      The global step `tf.Tensor`.
    """
    return training.create_global_step(graph)

  def _create_and_assert_global_step(self, graph):
    """Creates and asserts properties of the global step.

    Args:
      graph: The graph in which to create the global step tensor.

    Returns:
      The global step `tf.Tensor`.
    """
    step = self._create_global_step(graph)
    assert step == training.get_global_step()
    assert step.dtype.is_integer
    return step

  def _call_input_fn(self, input_fn, mode):
    """Calls the input function.

    Args:
      input_fn: The input function.
      mode: `tf.estimator.ModeKeys`

    Returns:
      The return value of the passed `input_fn`, which should be one of:

        * A 'tf.data.Dataset' object: Outputs of `Dataset` object must be a
            tuple `(features, labels)` with same constraints as below.
        * A tuple `(features, labels)`: Where `features` is a `Tensor` or a
          dictionary of string feature name to `Tensor` and `labels` is a
          `Tensor` or a dictionary of string label name to `Tensor`. Both
          `features` and `labels` are consumed by `model_fn`. They should
          satisfy the expectation of `model_fn` from inputs.

    Raises:
      ValueError: if `input_fn` takes invalid arguments.
    """
    input_fn_args = function_utils.fn_args(input_fn)
    kwargs = {}
    if 'mode' in input_fn_args:
      kwargs['mode'] = mode
    if 'params' in input_fn_args:
      kwargs['params'] = self.params
    if 'config' in input_fn_args:
      kwargs['config'] = self.config
    with ops.device('/cpu:0'):
      return input_fn(**kwargs)

  def _call_model_fn(self, features, labels, mode, config):
    """Calls model function.

    Args:
      features: features dict.
      labels: labels dict.
      mode: `tf.estimator.ModeKeys`
      config: `tf.estimator.RunConfig`

    Returns:
      An `tf.estimator.EstimatorSpec` object.

    Raises:
      ValueError: if `model_fn` returns invalid objects.
    """
    model_fn_args = function_utils.fn_args(self._model_fn)
    kwargs = {}
    if 'labels' in model_fn_args:
      kwargs['labels'] = labels
    else:
      if labels is not None:
        raise ValueError(
            'model_fn does not take labels, but input_fn returns labels.')
    if 'mode' in model_fn_args:
      kwargs['mode'] = mode
    if 'params' in model_fn_args:
      kwargs['params'] = self.params
    if 'config' in model_fn_args:
      kwargs['config'] = config

    logging.info('Calling model_fn.')
    model_fn_results = self._model_fn(features=features, **kwargs)
    logging.info('Done calling model_fn.')

    if not isinstance(model_fn_results, model_fn_lib.EstimatorSpec):
      raise ValueError('model_fn should return an EstimatorSpec.')

    return model_fn_results

  def _train_model(self, input_fn, hooks, saving_listeners):
    if self._train_distribution:
      return self._train_model_distributed(input_fn, hooks, saving_listeners)
    else:
      return self._train_model_default(input_fn, hooks, saving_listeners)

  def _train_model_default(self, input_fn, hooks, saving_listeners):
    """Initiate training with `input_fn`, without `DistributionStrategies`.

    Args:
      input_fn: A function that provides input data for training as minibatches.
      hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
        callbacks inside the training loop.
      saving_listeners: list of `tf.train.CheckpointSaverListener` objects. Used
        for callbacks that run immediately before or after checkpoint savings.

    Returns:
      Loss from training
    """
    worker_hooks = []
    with ops.Graph().as_default() as g, g.device(self._device_fn):
      random_seed.set_random_seed(self._config.tf_random_seed)
      global_step_tensor = self._create_and_assert_global_step(g)

      # Skip creating a read variable if _create_and_assert_global_step
      # returns None (e.g. tf.contrib.estimator.SavedModelEstimator).
      if global_step_tensor is not None:
        training_util._get_or_create_global_step_read(g)  # pylint: disable=protected-access

      features, labels, input_hooks = (
          self._get_features_and_labels_from_input_fn(
              input_fn, model_fn_lib.ModeKeys.TRAIN))
      worker_hooks.extend(input_hooks)
      estimator_spec = self._call_model_fn(
          features, labels, model_fn_lib.ModeKeys.TRAIN, self.config)
      global_step_tensor = training_util.get_global_step(g)
      return self._train_with_estimator_spec(estimator_spec, worker_hooks,
                                             hooks, global_step_tensor,
                                             saving_listeners)

  def _train_model_distributed(self, input_fn, hooks, saving_listeners):
    """Initiate training with `input_fn`, using `DistributionStrategies`.

    Args:
      input_fn: A function that provides input data for training as minibatches.
      hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
        callbacks inside the training loop.
      saving_listeners: list of `tf.train.CheckpointSaverListener` objects. Used
        for callbacks that run immediately before or after checkpoint savings.

    Returns:
      Loss from training
    """
    # pylint: disable=protected-access
    if (hasattr(self._config, '_distribute_coordinator_mode') and
        self._config._distribute_coordinator_mode):  # pylint: disable=protected-access
      distribute_coordinator_training.estimator_train(
          self,
          lambda est, s, train_hooks: est._actual_train_model_distributed(  # pylint: disable=g-long-lambda
              s, input_fn, train_hooks, saving_listeners),
          hooks)
      return self
    else:
      self._config._train_distribute.configure(self._config.session_config)
      return self._actual_train_model_distributed(
          self._config._train_distribute, input_fn, hooks, saving_listeners)
    # pylint: enable=protected-access

  def _actual_train_model_distributed(self, strategy, input_fn, hooks,
                                      saving_listeners):
    """That method that does actual training with distribution strategy."""
    # TODO(sourabhbajaj): Remove this hack once we migrate the other strategies
    # to use the new API
    is_tpu_strategy = (
        strategy.__class__.__name__ == 'TPUStrategy')

    worker_hooks = []
    with ops.Graph().as_default() as g:
      # We want to create the iterations variable outside the distribution scope
      # as that is just stored on the host and mainly used to drive the loop
      # and doesn't need to be a Mirrored/Device variable.
      if is_tpu_strategy:
        steps_per_run_variable = training.get_or_create_steps_per_run_variable()
      with strategy.scope():
        random_seed.set_random_seed(self._config.tf_random_seed)
        iterator, input_hooks = self._get_iterator_from_input_fn(
            input_fn, model_fn_lib.ModeKeys.TRAIN, strategy)
        worker_hooks.extend(input_hooks)
        global_step_tensor = self._create_and_assert_global_step(g)
        # we want to add to the global collection in the main thread not the
        # replica threads.
        ops.add_to_collection(
            training_util.GLOBAL_STEP_READ_KEY,
            strategy.extended.read_var(global_step_tensor))

        if is_tpu_strategy:
          # Create a step_fn from the train_op of grouped_estimator_spec
          def step_fn(ctx, inputs):
            """A single step that is passed to run_on_dataset."""
            if isinstance(inputs, tuple):
              features, labels = inputs
            else:
              features = inputs
              labels = None
            estimator_spec = strategy.extended.call_for_each_replica(
                self._call_model_fn,
                args=(features,
                      labels,
                      model_fn_lib.ModeKeys.TRAIN,
                      self.config))
            ctx.set_last_step_output(
                name='loss',
                output=estimator_spec.loss,
                reduce_op=reduce_util.ReduceOp.SUM)
            ctx.set_non_tensor_output(
                name='estimator_spec', output=estimator_spec)
            return estimator_spec.train_op

          # Create new train_op post graph rewrites
          initial_training_loss = constant_op.constant(1e7)
          ctx = strategy.extended.experimental_run_steps_on_iterator(
              step_fn, iterator, iterations=steps_per_run_variable,
              initial_loop_values={'loss': initial_training_loss})
          distributed_train_op = ctx.run_op
          loss = ctx.last_step_outputs['loss']
          grouped_estimator_spec = ctx.non_tensor_outputs['estimator_spec']
        else:
          features, labels = estimator_util.parse_iterator_result(
              iterator.get_next())
          grouped_estimator_spec = strategy.extended.call_for_each_replica(
              self._call_model_fn,
              args=(features,
                    labels,  # although this will be None it seems
                    model_fn_lib.ModeKeys.TRAIN,
                    self.config))
          loss = strategy.reduce(reduce_util.ReduceOp.SUM,
                                 grouped_estimator_spec.loss)
          distributed_train_op = grouped_estimator_spec.train_op

        scaffold = _combine_distributed_scaffold(
            grouped_estimator_spec.scaffold, strategy)

        # TODO(yuefengz): add a test for unwrapping per_device_hooks.
        def get_hooks_from_the_first_device(per_device_hooks):
          return [
              self._train_distribution.unwrap(per_device_hook)[0]
              for per_device_hook in per_device_hooks
          ]

        training_hooks = get_hooks_from_the_first_device(
            grouped_estimator_spec.training_hooks)
        training_chief_hooks = get_hooks_from_the_first_device(
            grouped_estimator_spec.training_chief_hooks)
        estimator_spec = model_fn_lib.EstimatorSpec(
            mode=grouped_estimator_spec.mode,
            loss=loss,
            train_op=strategy.group(distributed_train_op),
            training_hooks=training_hooks,
            training_chief_hooks=training_chief_hooks,
            scaffold=scaffold)
        return self._train_with_estimator_spec(estimator_spec, worker_hooks,
                                               hooks, global_step_tensor,
                                               saving_listeners)

  def _train_with_estimator_spec(self, estimator_spec, worker_hooks, hooks,
                                 global_step_tensor, saving_listeners):
    """Train a model with the given Estimator Spec."""
    if self._warm_start_settings:
      logging.info('Warm-starting with WarmStartSettings: %s' %
                   (self._warm_start_settings,))
      warm_starting_util.warm_start(*self._warm_start_settings)
    # Check if the user created a loss summary, and add one if they didn't.
    # We assume here that the summary is called 'loss'. If it is not, we will
    # make another one with the name 'loss' to ensure it shows up in the right
    # graph in TensorBoard.
    if not any([x.op.name == 'loss'
                for x in ops.get_collection(ops.GraphKeys.SUMMARIES)]):
      summary.scalar('loss', estimator_spec.loss)
    ops.add_to_collection(ops.GraphKeys.LOSSES, estimator_spec.loss)
    worker_hooks.extend(hooks)
    worker_hooks.append(
        training.NanTensorHook(estimator_spec.loss)
    )
    if self._config.log_step_count_steps is not None:
      worker_hooks.append(
          training.LoggingTensorHook(
              {
                  'loss': estimator_spec.loss,
                  'step': global_step_tensor
              },
              every_n_iter=self._config.log_step_count_steps)
      )
    worker_hooks.extend(estimator_spec.training_hooks)

    if not (estimator_spec.scaffold.saver or
            ops.get_collection(ops.GraphKeys.SAVERS)):
      ops.add_to_collection(
          ops.GraphKeys.SAVERS,
          training.Saver(
              sharded=True,
              max_to_keep=self._config.keep_checkpoint_max,
              keep_checkpoint_every_n_hours=(
                  self._config.keep_checkpoint_every_n_hours),
              defer_build=True,
              save_relative_paths=True))

    chief_hooks = []
    all_hooks = worker_hooks + list(estimator_spec.training_chief_hooks)
    saver_hooks = [
        h for h in all_hooks if isinstance(h, training.CheckpointSaverHook)]
    if (self._config.save_checkpoints_secs or
        self._config.save_checkpoints_steps):
      if not saver_hooks:
        chief_hooks = [
            training.CheckpointSaverHook(
                self._model_dir,
                save_secs=self._config.save_checkpoints_secs,
                save_steps=self._config.save_checkpoints_steps,
                scaffold=estimator_spec.scaffold)
        ]
        saver_hooks = [chief_hooks[0]]
    if saving_listeners:
      if not saver_hooks:
        raise ValueError(
            'There should be a CheckpointSaverHook to use saving_listeners. '
            'Please set one of the RunConfig.save_checkpoints_steps or '
            'RunConfig.save_checkpoints_secs.')
      else:
        # It is expected to have one CheckpointSaverHook. If multiple, we pick
        # up the first one to add listener.
        saver_hooks[0]._listeners.extend(saving_listeners)  # pylint: disable=protected-access

    # Add summary hooks to worker 0 if we are running with a master, to ensure
    # that summaries are written at correct intervals even with long-running
    # evaluations.
    save_summary_steps = self._config.save_summary_steps
    log_step_count_steps = self._config.log_step_count_steps

    # Check existence of appropriate cluster spec fields, as well as master and
    # worker nodes. As master also performs evaluation, summary writing must
    # occur on a different node. The presence of a worker is also checked to
    # prevent reassigning hooks for single-replica jobs with just a master node.
    if (self._config.cluster_spec and self._config.cluster_spec.jobs and
        (run_config.TaskType.WORKER in self._config.cluster_spec.jobs) and
        (run_config.TaskType.MASTER in self._config.cluster_spec.jobs)):
      # Update config values to prevent the default hooks from being created on
      # the master or other workers.
      save_summary_steps = 0
      log_step_count_steps = None

      if (self._config.task_type == run_config.TaskType.WORKER and
          self._config.task_id == 0):
        if (self._config.save_summary_steps and
            self._config.save_summary_steps > 0):
          worker_hooks.append(
              training.SummarySaverHook(
                  save_steps=self._config.save_summary_steps,
                  output_dir=self._config.model_dir,
                  scaffold=estimator_spec.scaffold))

        if (self._config.log_step_count_steps and
            self._config.log_step_count_steps > 0):
          worker_hooks.append(
              training.StepCounterHook(
                  every_n_steps=self._config.log_step_count_steps,
                  output_dir=self._config.model_dir))

    with training.MonitoredTrainingSession(
        master=self._config.master,
        is_chief=self._config.is_chief,
        checkpoint_dir=self._model_dir,
        scaffold=estimator_spec.scaffold,
        hooks=worker_hooks,
        chief_only_hooks=(
            tuple(chief_hooks) + tuple(estimator_spec.training_chief_hooks)),
        save_checkpoint_secs=0,  # Saving is handled by a hook.
        save_summaries_steps=save_summary_steps,
        config=self._session_config,
        log_step_count_steps=log_step_count_steps) as mon_sess:
      loss = None
      any_step_done = False
      while not mon_sess.should_stop():
        _, loss = mon_sess.run([estimator_spec.train_op, estimator_spec.loss])
        any_step_done = True
    if not any_step_done:
      logging.warning('Training with estimator made no steps. '
                      'Perhaps input is empty or misspecified.')
    return loss

  def _evaluate_build_graph(self, input_fn, hooks=None, checkpoint_path=None):
    """Builds the graph and related hooks to run evaluation."""
    random_seed.set_random_seed(self._config.tf_random_seed)
    self._create_and_assert_global_step(ops.get_default_graph())

    if self._eval_distribution:
      (scaffold, evaluation_hooks, input_hooks, update_op, eval_dict) = (
          self._call_model_fn_eval_distributed(input_fn, self.config))
    else:
      (scaffold, evaluation_hooks, input_hooks, update_op, eval_dict) = (
          self._call_model_fn_eval(input_fn, self.config))

    global_step_tensor = training_util.get_global_step(ops.get_default_graph())
    # Call to warm_start has to be after model_fn is called.
    self._maybe_warm_start(checkpoint_path)

    if ops.GraphKeys.GLOBAL_STEP in eval_dict:
      raise ValueError(
          'Metric with name `global_step` is not allowed, because Estimator '
          'already defines a default metric with the same name.')
    eval_dict[ops.GraphKeys.GLOBAL_STEP] = global_step_tensor

    all_hooks = list(input_hooks)
    all_hooks.extend(hooks)
    all_hooks.extend(list(evaluation_hooks or []))
    # New local variables have been added, so update the estimator spec's
    # local init op if it was defined.
    if scaffold and scaffold.local_init_op:
      # Ensure that eval step has been created before updating local init op.
      evaluation._get_or_create_eval_step()  # pylint: disable=protected-access

      scaffold = monitored_session.Scaffold(
          local_init_op=control_flow_ops.group(
              scaffold.local_init_op,
              monitored_session.Scaffold.default_local_init_op()),
          copy_from_scaffold=scaffold
      )

    return scaffold, update_op, eval_dict, all_hooks

  def _call_model_fn_eval(self, input_fn, config):
    """Call model_fn for evaluation and handle return values."""
    features, labels, input_hooks = self._get_features_and_labels_from_input_fn(
        input_fn, model_fn_lib.ModeKeys.EVAL)

    estimator_spec = self._call_model_fn(
        features, labels, model_fn_lib.ModeKeys.EVAL, config)
    eval_metric_ops = _verify_and_create_loss_metric(
        estimator_spec.eval_metric_ops, estimator_spec.loss)
    update_op, eval_dict = _extract_metric_update_ops(eval_metric_ops)
    return (estimator_spec.scaffold, estimator_spec.evaluation_hooks,
            input_hooks, update_op, eval_dict)

  def _call_model_fn_eval_distributed(self, input_fn, config):
    """Call model_fn in distribution mode and handle return values."""

    iterator, input_hooks = self._get_iterator_from_input_fn(
        input_fn, model_fn_lib.ModeKeys.EVAL, self._eval_distribution)

    is_tpu_strategy = (
        self._eval_distribution.__class__.__name__ == 'TPUStrategy')

    if is_tpu_strategy:
      steps_per_run_variable = training.get_or_create_steps_per_run_variable()
      def step_fn(ctx, inputs):
        """Runs one step of the eval computation and captures outputs."""
        if isinstance(inputs, tuple):
          features, labels = inputs
        else:
          features = inputs
          labels = None
        estimator_spec = self._eval_distribution.extended.call_for_each_replica(
            self._call_model_fn,
            args=(features, labels, model_fn_lib.ModeKeys.EVAL, config))
        eval_metric_ops = _verify_and_create_loss_metric(
            estimator_spec.eval_metric_ops, estimator_spec.loss,
            self._eval_distribution)
        update_op, eval_dict = _extract_metric_update_ops(
            eval_metric_ops, self._eval_distribution)
        ctx.set_non_tensor_output(name='estimator_spec', output=estimator_spec)
        ctx.set_non_tensor_output(name='eval_dict', output=eval_dict)
        return update_op

      # TODO(priyag): Fix eval step hook to account for steps_per_run.
      ctx = self._eval_distribution.extended.experimental_run_steps_on_iterator(
          step_fn, iterator, iterations=steps_per_run_variable)
      update_op = ctx.run_op
      eval_dict = ctx.non_tensor_outputs['eval_dict']
      grouped_estimator_spec = ctx.non_tensor_outputs['estimator_spec']
    else:
      features, labels = estimator_util.parse_iterator_result(
          iterator.get_next())
      grouped_estimator_spec = (
          self._eval_distribution.extended.call_for_each_replica(
              self._call_model_fn,
              args=(features, labels, model_fn_lib.ModeKeys.EVAL, config)))
      eval_metric_ops = _verify_and_create_loss_metric(
          grouped_estimator_spec.eval_metric_ops, grouped_estimator_spec.loss,
          self._eval_distribution)
      update_op, eval_dict = _extract_metric_update_ops(
          eval_metric_ops, self._eval_distribution)

    scaffold = _combine_distributed_scaffold(
        grouped_estimator_spec.scaffold, self._eval_distribution)
    evaluation_hooks = self._eval_distribution.unwrap(
        grouped_estimator_spec.evaluation_hooks)[0]
    return (scaffold, evaluation_hooks, input_hooks, update_op, eval_dict)

  def _evaluate_run(self, checkpoint_path, scaffold, update_op, eval_dict,
                    all_hooks, output_dir):
    """Run evaluation."""
    eval_results = evaluation._evaluate_once(  # pylint: disable=protected-access
        checkpoint_path=checkpoint_path,
        master=self._config.evaluation_master,
        scaffold=scaffold,
        eval_ops=update_op,
        final_ops=eval_dict,
        hooks=all_hooks,
        config=self._session_config)

    current_global_step = eval_results[ops.GraphKeys.GLOBAL_STEP]

    _write_dict_to_summary(
        output_dir=output_dir,
        dictionary=eval_results,
        current_global_step=current_global_step)

    if checkpoint_path:
      _write_checkpoint_path_to_summary(
          output_dir=output_dir,
          checkpoint_path=checkpoint_path,
          current_global_step=current_global_step)

    return eval_results

  def _maybe_warm_start(self, checkpoint_path):
    if not checkpoint_path and self._warm_start_settings:
      logging.info('Warm-starting with WarmStartSettings: %s' %
                   (self._warm_start_settings,))
      warm_starting_util.warm_start(*self._warm_start_settings)


@estimator_export(v1=['estimator.Estimator'])  # pylint: disable=missing-docstring
class Estimator(EstimatorV2):
  __doc__ = EstimatorV2.__doc__

  # This is settable in v1, but has been changed to True in all cases for v2.
  # The default in v1 was False, so we maintain that here.
  _strip_default_attrs = False

  def _assert_members_are_not_overridden(self):
    """Asserts members of `Estimator` are not overridden."""
    _assert_members_are_not_overridden(Estimator, self)

  def export_savedmodel(
      self, export_dir_base, serving_input_receiver_fn,
      assets_extra=None,
      as_text=False,
      checkpoint_path=None,
      strip_default_attrs=False):
    # pylint: disable=line-too-long
    """Exports inference graph as a `SavedModel` into the given dir.

    For a detailed guide, see
    [Using SavedModel with Estimators](https://tensorflow.org/guide/saved_model#using_savedmodel_with_estimators).

    This method builds a new graph by first calling the
    `serving_input_receiver_fn` to obtain feature `Tensor`s, and then calling
    this `Estimator`'s `model_fn` to generate the model graph based on those
    features. It restores the given checkpoint (or, lacking that, the most
    recent checkpoint) into this graph in a fresh session.  Finally it creates
    a timestamped export directory below the given `export_dir_base`, and writes
    a `SavedModel` into it containing a single `tf.MetaGraphDef` saved from this
    session.

    The exported `MetaGraphDef` will provide one `SignatureDef` for each
    element of the `export_outputs` dict returned from the `model_fn`, named
    using
    the same keys.  One of these keys is always
    `tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY`,
    indicating which
    signature will be served when a serving request does not specify one.
    For each signature, the outputs are provided by the corresponding
    `tf.estimator.export.ExportOutput`s, and the inputs are always the input
    receivers provided by
    the `serving_input_receiver_fn`.

    Extra assets may be written into the `SavedModel` via the `assets_extra`
    argument.  This should be a dict, where each key gives a destination path
    (including the filename) relative to the assets.extra directory.  The
    corresponding value gives the full path of the source file to be copied.
    For example, the simple case of copying a single file without renaming it
    is specified as `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.

    Args:
      export_dir_base: A string containing a directory in which to create
        timestamped subdirectories containing exported `SavedModel`s.
      serving_input_receiver_fn: A function that takes no argument and returns a
        `tf.estimator.export.ServingInputReceiver` or
        `tf.estimator.export.TensorServingInputReceiver`.
      assets_extra: A dict specifying how to populate the assets.extra directory
        within the exported `SavedModel`, or `None` if no extra assets are
        needed.
      as_text: whether to write the `SavedModel` proto in text format.
      checkpoint_path: The checkpoint path to export.  If `None` (the default),
        the most recent checkpoint found within the model directory is chosen.
      strip_default_attrs: Boolean. If `True`, default-valued attributes will be
        removed from the `NodeDef`s. For a detailed guide, see [Stripping
        Default-Valued Attributes](
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).

    Returns:
      The string path to the exported directory.

    Raises:
      ValueError: if no `serving_input_receiver_fn` is provided, no
      `export_outputs` are provided, or no checkpoint can be found.
    """
    # pylint: enable=line-too-long
    self._strip_default_attrs = strip_default_attrs
    return super(Estimator, self).export_saved_model(
        export_dir_base,
        serving_input_receiver_fn,
        assets_extra=assets_extra,
        as_text=as_text,
        checkpoint_path=checkpoint_path,
        experimental_mode=model_fn_lib.ModeKeys.PREDICT)

  def export_saved_model(
      self, export_dir_base, serving_input_receiver_fn,
      assets_extra=None,
      as_text=False,
      checkpoint_path=None,
      experimental_mode=model_fn_lib.ModeKeys.PREDICT):
    self._strip_default_attrs = True
    return super(Estimator, self).export_saved_model(
        export_dir_base,
        serving_input_receiver_fn,
        assets_extra=assets_extra,
        as_text=as_text,
        checkpoint_path=checkpoint_path,
        experimental_mode=experimental_mode)

  def _call_add_meta_graph_and_variables(
      self, save_variables, builder, session, kwargs):
    """For v1, add in the strip_default_attrs arg."""

    kwargs['strip_default_attrs'] = self._strip_default_attrs

    super(Estimator, self)._call_add_meta_graph_and_variables(
        save_variables, builder, session, kwargs)


if hasattr(Estimator.export_saved_model, '__func__'):
  # Python 2
  Estimator.export_saved_model.__func__.__doc__ = (
      EstimatorV2.export_saved_model.__doc__)
else:
  # Python 3
  Estimator.export_saved_model.__doc__ = EstimatorV2.export_saved_model.__doc__


def _assert_members_are_not_overridden(cls, obj):
  """Assert Estimator methods are not overwritten."""
  # TPUEstimator is special cased (owned by TF).
  if obj.__class__.__name__ == 'TPUEstimator':
    return

  allowed_overrides = set([
      '_create_and_assert_global_step',
      '_tf_api_names', '_tf_api_names_v1', '_estimator_api_names',
      '_estimator_api_names_v1', '_estimator_api_constants',
      '_estimator_api_constants_v1',
  ])

  estimator_members = set([m for m in cls.__dict__.keys()
                           if not m.startswith('__')])
  subclass_members = set(obj.__class__.__dict__.keys())
  common_members = estimator_members & subclass_members - allowed_overrides
  overridden_members = [
      m for m in common_members if cls.__dict__[m] != obj.__class__.__dict__[m]]
  if overridden_members:
    raise ValueError(
        'Subclasses of Estimator cannot override members of Estimator. '
        '{} does override {}'.format(obj.__class__, overridden_members))


def _verify_and_create_loss_metric(eval_metric_ops, loss, distribution=None):
  """Creates a metric for loss and throws an error if one already exists."""
  if model_fn_lib.LOSS_METRIC_KEY in eval_metric_ops:
    raise ValueError(
        'Metric with name "%s" is not allowed, because Estimator ' %
        (model_fn_lib.LOSS_METRIC_KEY) +
        'already defines a default metric with the same name.')

  if distribution is None:
    loss_metric = metrics_lib.mean(loss)
  else:
    loss_metric = distribution.extended.call_for_each_replica(
        metrics_lib.mean, args=(loss,))
  eval_metric_ops[model_fn_lib.LOSS_METRIC_KEY] = loss_metric
  return eval_metric_ops


def maybe_overwrite_model_dir_and_session_config(config, model_dir):
  """Overwrite estimator config by `model_dir` and `session_config` if needed.

  Args:
    config: Original estimator config.
    model_dir: Estimator model checkpoint directory.

  Returns:
    Overwritten estimator config.

  Raises:
    ValueError: Model directory inconsistent between `model_dir` and `config`.
  """

  if config is None:
    config = run_config.RunConfig()
    logging.info('Using default config.')
  if not isinstance(config, run_config.RunConfig):
    raise ValueError(
        'config must be an instance of `RunConfig`, but provided %s.' % config)

  if config.session_config is None:
    session_config = run_config.get_default_session_config()
    config = run_config.RunConfig.replace(config, session_config=session_config)

  model_dir = compat_internal.path_to_str(model_dir)
  if model_dir is not None:
    if (getattr(config, 'model_dir', None) is not None and
        config.model_dir != model_dir):
      raise ValueError(
          "`model_dir` are set both in constructor and `RunConfig`, but with "
          "different values. In constructor: '{}', in `RunConfig`: "
          "'{}' ".format(model_dir, config.model_dir))
  if model_dir:
    config = run_config.RunConfig.replace(config, model_dir=model_dir)
  elif getattr(config, 'model_dir', None) is None:
    model_dir = tempfile.mkdtemp()
    logging.warning('Using temporary folder as model directory: %s', model_dir)
    config = run_config.RunConfig.replace(config, model_dir=model_dir)

  return config


def create_per_replica_ready_for_local_init_op(scaffold):
  """Create a `tf.train.Scaffold.ready_for_local_init_op` inside a replica."""
  if scaffold.ready_for_local_init_op:
    return scaffold.ready_for_local_init_op

  def default_ready_for_local_init_op():
    return variables.report_uninitialized_variables(
        variables.global_variables())

  return monitored_session.Scaffold.get_or_default(
      'ready_for_local_init_op', ops.GraphKeys.READY_FOR_LOCAL_INIT_OP,
      default_ready_for_local_init_op)


def _combine_distributed_scaffold(grouped_scaffold, distribution):
  """Combines scaffold(s) returned from `call_for_each_replica`."""

  # TODO(anjalisridhar): Figure out how to resolve the following scaffold
  # parameters: init_feed_dict, init_fn.
  scaffold_list = distribution.unwrap(grouped_scaffold)
  init_feed_dict = [
      s.init_feed_dict
      for s in scaffold_list
      if s.init_feed_dict is not None
  ]
  if init_feed_dict:
    init_feed_dict = distribution.group(init_feed_dict)
  else:
    init_feed_dict = None

  init_fn = [
      s._user_init_fn for s in scaffold_list if s._user_init_fn is not None  # pylint: disable=protected-access
  ]
  if init_fn:
    init_fn = distribution.group(init_fn)
  else:
    init_fn = None

  init_op = [s.init_op for s in scaffold_list if s.init_op is not None]
  if init_op:
    init_op = distribution.group(init_op)
  else:
    init_op = None

  def _unwrap_and_concat(value):
    value = nest.flatten(distribution.unwrap(value))
    if len(value) != 1:
      return array_ops.concat(value, 0)
    return value[0]

  ready_op = distribution.extended.call_for_each_replica(
      lambda scaffold: scaffold.ready_op, args=(grouped_scaffold,))
  if ready_op is not None:
    ready_op = _unwrap_and_concat(ready_op)

  ready_for_local_init_op = distribution.extended.call_for_each_replica(
      create_per_replica_ready_for_local_init_op, args=(grouped_scaffold,))
  if ready_for_local_init_op is not None:
    ready_for_local_init_op = _unwrap_and_concat(ready_for_local_init_op)
  else:
    ready_for_local_init_op = None

  local_init_op = [
      s.local_init_op
      for s in scaffold_list
      if s.local_init_op is not None
  ]
  if local_init_op:
    local_init_op = distribution.group(local_init_op)
  else:
    local_init_op = None

  summary_op = [
      s.summary_op for s in scaffold_list if s.summary_op is not None
  ]
  if summary_op:
    summary_op = distribution.group(summary_op)
  else:
    summary_op = None

  savers = [s.saver for s in scaffold_list if s.saver is not None]
  if savers:
    saver = savers[0]
  else:
    saver = None

  scaffold = monitored_session.Scaffold(
      init_op=init_op,
      ready_op=ready_op,
      ready_for_local_init_op=ready_for_local_init_op,
      local_init_op=local_init_op,
      summary_op=summary_op,
      saver=saver,
      init_feed_dict=init_feed_dict,
      init_fn=init_fn)
  return scaffold


def _check_checkpoint_available(model_dir):
  latest_path = checkpoint_management.latest_checkpoint(model_dir)
  if not latest_path:
    raise ValueError(
        'Could not find trained model in model_dir: {}.'.format(model_dir))


def _check_hooks_type(hooks):
  """Returns hooks if all are `SessionRunHook`, raises TypeError otherwise."""
  hooks = list(hooks or [])
  for h in hooks:
    if not isinstance(h, training.SessionRunHook):
      raise TypeError('Hooks must be a SessionRunHook, given: {}'.format(h))
  return hooks


def _check_listeners_type(saving_listeners):
  """Check listeners type."""
  listeners = list(saving_listeners or [])
  for l in listeners:
    if not isinstance(l, training.CheckpointSaverListener):
      raise TypeError(
          'saving_listeners must be a list of CheckpointSaverListener, '
          'given: {}'.format(l))
  return listeners


def _get_replica_device_setter(config):
  """Creates a replica device setter if required as a default `device_fn`.

  `Estimator` uses `tf.train.ReplicaDeviceSetter` as a default device placer. It
  sets the
  distributed related arguments such as number of `ps_replicas` based on given
  `config`.

  Args:
    config: A `tf.estimator.RunConfig` instance.

  Returns:
    A replica device setter, or `None`.
  """
  if config.task_type:
    worker_device = '/job:%s/task:%d' % (config.task_type, config.task_id)
  else:
    worker_device = '/job:worker'

  if config.num_ps_replicas > 0:
    return training.replica_device_setter(
        ps_tasks=config.num_ps_replicas,
        worker_device=worker_device,
        merge_devices=True,
        ps_ops=list(device_setter.STANDARD_PS_OPS),
        cluster=config.cluster_spec)
  else:
    return None


def _verify_model_fn_args(model_fn, params):
  """Verifies `model_fn` arguments."""
  args = set(function_utils.fn_args(model_fn))
  if 'features' not in args:
    raise ValueError('model_fn (%s) must include features argument.' % model_fn)
  if params is not None and 'params' not in args:
    raise ValueError('model_fn (%s) does not include params argument, '
                     'but params (%s) is passed to Estimator.' % (model_fn,
                                                                  params))
  if params is None and 'params' in args:
    logging.warning('Estimator\'s model_fn (%s) includes params '
                    'argument, but params are not passed to Estimator.',
                    model_fn)
  non_valid_args = list(args - _VALID_MODEL_FN_ARGS)
  if non_valid_args:
    raise ValueError('model_fn (%s) has following not expected args: %s' %
                     (model_fn, non_valid_args))


def _load_global_step_from_checkpoint_dir(checkpoint_dir):
  try:
    checkpoint_reader = training.NewCheckpointReader(
        training.latest_checkpoint(checkpoint_dir))
    return checkpoint_reader.get_tensor(ops.GraphKeys.GLOBAL_STEP)
  except:  # pylint: disable=bare-except
    return 0


def _extract_metric_update_ops(eval_dict, distribution=None):
  """Separate update operations from metric value operations."""
  update_ops = []
  value_ops = {}
  # Sort metrics lexicographically so graph is identical every time.
  for name, value in sorted(six.iteritems(eval_dict)):
    value_ops[name] = value[0]
    update_ops.append(
        distribution.group(value[1]) if distribution else value[1])

  update_op = control_flow_ops.group(*update_ops) if update_ops else None
  return update_op, value_ops


def _dict_to_str(dictionary):
  """Get a `str` representation of a `dict`.

  Args:
    dictionary: The `dict` to be represented as `str`.

  Returns:
    A `str` representing the `dictionary`.
  """
  return ', '.join('%s = %s' % (k, v)
                   for k, v in sorted(six.iteritems(dictionary))
                   if not isinstance(v, six.binary_type))


def _write_dict_to_summary(output_dir,
                           dictionary,
                           current_global_step):
  """Writes a `dict` into summary file in given output directory.

  Args:
    output_dir: `str`, directory to write the summary file in.
    dictionary: the `dict` to be written to summary file.
    current_global_step: `int`, the current global step.
  """
  logging.info('Saving dict for global step %d: %s', current_global_step,
               _dict_to_str(dictionary))
  summary_writer = writer_cache.FileWriterCache.get(output_dir)
  summary_proto = summary_pb2.Summary()
  for key in dictionary:
    if dictionary[key] is None:
      continue
    if key == 'global_step':
      continue
    if (isinstance(dictionary[key], np.float32) or
        isinstance(dictionary[key], float)):
      summary_proto.value.add(tag=key, simple_value=float(dictionary[key]))
    elif (isinstance(dictionary[key], np.int64) or
          isinstance(dictionary[key], np.int32) or
          isinstance(dictionary[key], int)):
      summary_proto.value.add(tag=key, simple_value=int(dictionary[key]))
    elif isinstance(dictionary[key], six.binary_type):
      try:
        summ = summary_pb2.Summary.FromString(dictionary[key])
        for i, _ in enumerate(summ.value):
          summ.value[i].tag = '%s/%d' % (key, i)
        summary_proto.value.extend(summ.value)
      except message.DecodeError:
        logging.warn('Skipping summary for %s, cannot parse string to Summary.',
                     key)
        continue
    elif isinstance(dictionary[key], np.ndarray):
      value = summary_proto.value.add()
      value.tag = key
      value.node_name = key
      tensor_proto = tensor_util.make_tensor_proto(dictionary[key])
      value.tensor.CopyFrom(tensor_proto)
      # pylint: disable=line-too-long
      logging.info(
          'Summary for np.ndarray is not visible in Tensorboard by default. '
          'Consider using a Tensorboard plugin for visualization (see '
          'https://github.com/tensorflow/tensorboard-plugin-example/blob/master/README.md'
          ' for more information).')
      # pylint: enable=line-too-long
    else:
      logging.warn(
          'Skipping summary for %s, must be a float, np.float32, np.int64, '
          'np.int32 or int or np.ndarray or a serialized string of Summary.',
          key)
  summary_writer.add_summary(summary_proto, current_global_step)
  summary_writer.flush()


def _write_checkpoint_path_to_summary(output_dir, checkpoint_path,
                                      current_global_step):
  """Writes `checkpoint_path` into summary file in the given output directory.

  Args:
    output_dir: `str`, directory to write the summary file in.
    checkpoint_path: `str`, checkpoint file path to be written to summary file.
    current_global_step: `int`, the current global step.
  """

  checkpoint_path_tag = 'checkpoint_path'

  logging.info('Saving \'%s\' summary for global step %d: %s',
               checkpoint_path_tag, current_global_step, checkpoint_path)
  summary_proto = summary_pb2.Summary()
  summary_proto.value.add(
      tag=checkpoint_path_tag,
      tensor=tensor_util.make_tensor_proto(
          checkpoint_path, dtype=dtypes.string))
  summary_writer = writer_cache.FileWriterCache.get(output_dir)
  summary_writer.add_summary(summary_proto, current_global_step)
  summary_writer.flush()


def _has_dataset_or_queue_runner(maybe_tensor):
  """Returns `True` if `Dataset` or `QueueRunner` has been used."""
  # Check TF dataset first. Here, we use a simple algorithm to check the top
  # level Tensors only, which should be sufficient for most users.
  tensors = [x for x in nest.flatten(maybe_tensor) if isinstance(x, ops.Tensor)]
  if any([t.op.type == 'IteratorGetNext' for t in tensors]):
    return True

  # Now, check queue.
  return ops.get_default_graph().get_collection(ops.GraphKeys.QUEUE_RUNNERS)


VocabInfo = warm_starting_util.VocabInfo  # pylint: disable=invalid-name
estimator_export('estimator.VocabInfo')(VocabInfo)


@estimator_export('estimator.WarmStartSettings')
class WarmStartSettings(
    collections.namedtuple('WarmStartSettings', [
        'ckpt_to_initialize_from',
        'vars_to_warm_start',
        'var_name_to_vocab_info',
        'var_name_to_prev_var_name',
    ])):
  """Settings for warm-starting in `tf.estimator.Estimators`.

  Example Use with canned `tf.estimator.DNNEstimator`:

  ```
  emb_vocab_file = tf.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_vocabulary_file(
          "sc_vocab_file", "new_vocab.txt", vocab_size=100),
      dimension=8)
  emb_vocab_list = tf.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_vocabulary_list(
          "sc_vocab_list", vocabulary_list=["a", "b"]),
      dimension=8)
  estimator = tf.estimator.DNNClassifier(
    hidden_units=[128, 64], feature_columns=[emb_vocab_file, emb_vocab_list],
    warm_start_from=ws)
  ```

  where `ws` could be defined as:

  Warm-start all weights in the model (input layer and hidden weights).
  Either the directory or a specific checkpoint can be provided (in the case
  of the former, the latest checkpoint will be used):

  ```
  ws = WarmStartSettings(ckpt_to_initialize_from="/tmp")
  ws = WarmStartSettings(ckpt_to_initialize_from="/tmp/model-1000")
  ```

  Warm-start only the embeddings (input layer):

  ```
  ws = WarmStartSettings(ckpt_to_initialize_from="/tmp",
                         vars_to_warm_start=".*input_layer.*")
  ```

  Warm-start all weights but the embedding parameters corresponding to
  `sc_vocab_file` have a different vocab from the one used in the current
  model:

  ```
  vocab_info = tf.estimator.VocabInfo(
      new_vocab=sc_vocab_file.vocabulary_file,
      new_vocab_size=sc_vocab_file.vocabulary_size,
      num_oov_buckets=sc_vocab_file.num_oov_buckets,
      old_vocab="old_vocab.txt"
  )
  ws = WarmStartSettings(
      ckpt_to_initialize_from="/tmp",
      var_name_to_vocab_info={
          "input_layer/sc_vocab_file_embedding/embedding_weights": vocab_info
      })
  ```

  Warm-start only `sc_vocab_file` embeddings (and no other variables), which
  have a different vocab from the one used in the current model:

  ```
  vocab_info = tf.estimator.VocabInfo(
      new_vocab=sc_vocab_file.vocabulary_file,
      new_vocab_size=sc_vocab_file.vocabulary_size,
      num_oov_buckets=sc_vocab_file.num_oov_buckets,
      old_vocab="old_vocab.txt"
  )
  ws = WarmStartSettings(
      ckpt_to_initialize_from="/tmp",
      vars_to_warm_start=None,
      var_name_to_vocab_info={
          "input_layer/sc_vocab_file_embedding/embedding_weights": vocab_info
      })
  ```

  Warm-start all weights but the parameters corresponding to `sc_vocab_file`
  have a different vocab from the one used in current checkpoint, and only
  100 of those entries were used:

  ```
  vocab_info = tf.estimator.VocabInfo(
      new_vocab=sc_vocab_file.vocabulary_file,
      new_vocab_size=sc_vocab_file.vocabulary_size,
      num_oov_buckets=sc_vocab_file.num_oov_buckets,
      old_vocab="old_vocab.txt",
      old_vocab_size=100
  )
  ws = WarmStartSettings(
      ckpt_to_initialize_from="/tmp",
      var_name_to_vocab_info={
          "input_layer/sc_vocab_file_embedding/embedding_weights": vocab_info
      })
  ```

  Warm-start all weights but the parameters corresponding to `sc_vocab_file`
  have a different vocab from the one used in current checkpoint and the
  parameters corresponding to `sc_vocab_list` have a different name from the
  current checkpoint:

  ```
  vocab_info = tf.estimator.VocabInfo(
      new_vocab=sc_vocab_file.vocabulary_file,
      new_vocab_size=sc_vocab_file.vocabulary_size,
      num_oov_buckets=sc_vocab_file.num_oov_buckets,
      old_vocab="old_vocab.txt",
      old_vocab_size=100
  )
  ws = WarmStartSettings(
      ckpt_to_initialize_from="/tmp",
      var_name_to_vocab_info={
          "input_layer/sc_vocab_file_embedding/embedding_weights": vocab_info
      },
      var_name_to_prev_var_name={
          "input_layer/sc_vocab_list_embedding/embedding_weights":
              "old_tensor_name"
      })
  ```

  Warm-start all variables (including non-TRAINABLE):

  ```
  ws = WarmStartSettings(ckpt_to_initialize_from="/tmp",
                         vars_to_warm_start=[".*"])
  ```

  Warm-start non-TRAINABLE variables "v1", "v1/Momentum", and "v2" but not
  "v2/momentum":

  ```
  ws = WarmStartSettings(ckpt_to_initialize_from="/tmp",
                         vars_to_warm_start=["v1", "v2[^/]"])
  ```

  Attributes:
    ckpt_to_initialize_from: [Required] A string specifying the directory with
      checkpoint file(s) or path to checkpoint from which to warm-start the
      model parameters.
    vars_to_warm_start: [Optional] One of the following:

      - A regular expression (string) that captures which variables to
        warm-start (see tf.get_collection).  This expression will only consider
        variables in the TRAINABLE_VARIABLES collection -- if you need to
        warm-start non_TRAINABLE vars (such as optimizer accumulators or batch
        norm statistics), please use the below option.
      - A list of Variables to warm-start.  If you do not have access to the
        `Variable` objects at the call site, please use the below option.
      - A list of strings, each a regex scope provided to tf.get_collection with
        GLOBAL_VARIABLES.  For backwards compatibility reasons, this is separate
        from the single-string argument type.
      - `None`, in which case only variables specified in
        `var_name_to_vocab_info` will be warm-started.

      Defaults to `'.*'`, which warm-starts all variables in the
      TRAINABLE_VARIABLES collection.    Note that this excludes variables such
      as accumulators and moving statistics from batch norm.
    var_name_to_vocab_info: [Optional] Dict of variable names (strings) to
      `tf.estimator.VocabInfo`. The variable names should be "full" variables,
      not the names of the partitions.  If not explicitly provided, the variable
      is assumed to have no (changes to) vocabulary.
    var_name_to_prev_var_name: [Optional] Dict of variable names (strings) to
      name of the previously-trained variable in `ckpt_to_initialize_from`. If
      not explicitly provided, the name of the variable is assumed to be same
      between previous checkpoint and current model.  Note that this has no
      effect on the set of variables that is warm-started, and only controls
      name mapping (use `vars_to_warm_start` for controlling what variables to
      warm-start).
  """

  def __new__(cls,
              ckpt_to_initialize_from,
              vars_to_warm_start='.*',
              var_name_to_vocab_info=None,
              var_name_to_prev_var_name=None):
    if not ckpt_to_initialize_from:
      raise ValueError(
          '`ckpt_to_initialize_from` MUST be set in WarmStartSettings')
    return super(WarmStartSettings, cls).__new__(
        cls,
        ckpt_to_initialize_from,
        vars_to_warm_start,
        var_name_to_vocab_info or {},
        var_name_to_prev_var_name or {},
    )


def _get_saved_model_ckpt(saved_model_dir):
  """Return path to variables checkpoint in a `SavedModel` directory."""
  if not gfile.Exists(
      os.path.join(saved_model_utils.get_variables_dir(saved_model_dir),
                   compat.as_text('variables.index'))):
    raise ValueError('Directory provided has an invalid SavedModel format: %s'
                     % saved_model_dir)
  return saved_model_utils.get_variables_path(saved_model_dir)


def _get_default_warm_start_settings(warm_start_from):
  """Returns default `tf.estimator.WarmStartSettings`.

  Args:
    warm_start_from: Either a string representing the filepath of a checkpoint
      or `SavedModel` to initialize from, or an instance of
      `tf.estimator.WarmStartSettings`.

  Returns:
    Either None or an instance of `WarmStartSettings`.

  Raises:
    ValueError: If `warm_start_from` is not `None` but is neither a string nor
    an
      instance of `WarmStartSettings`.
  """
  if warm_start_from is None:
    return None
  if isinstance(warm_start_from, (six.string_types, six.binary_type)):
    # Infer that this is a SavedModel if export_path +
    # 'variables/variables.index' exists, and if so, construct the
    # WarmStartSettings pointing to the variables path
    # (export_path + 'variables/variables').
    if gfile.Exists(os.path.join(
        saved_model_utils.get_variables_dir(warm_start_from),
        compat.as_text('variables.index'))):
      logging.info('Warm-starting from a SavedModel')
      return WarmStartSettings(
          ckpt_to_initialize_from=saved_model_utils.get_variables_path(
              warm_start_from))
    return WarmStartSettings(ckpt_to_initialize_from=warm_start_from)
  elif isinstance(warm_start_from, WarmStartSettings):
    return warm_start_from
  else:
    raise ValueError('warm_start_from must be a string or a WarmStartSettings, '
                     'instead got {}'.format(type(warm_start_from)))
