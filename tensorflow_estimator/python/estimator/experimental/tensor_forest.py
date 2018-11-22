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
"""Estimator classes for TensorForest."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.estimator import estimator
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.ops.losses import losses
from tensorflow.python.summary import summary
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.ops import gen_tensor_forest_ops
from tensorflow.python.ops import tensor_forest_ops, math_ops, array_ops


_ForestHParams = collections.namedtuple('TreeHParams', [
    'logits_dimension',
    'n_trees', 'max_nodes', 'num_splits_to_consider',
    'split_node_after_samples', 'is_regression',
])

VARIANCE_LOGITS_KEY = 'logits_variance'
VARIANCE_AFTER_ACTIVATION_KEY = 'probs_variance'


class TensorForestGraphs(object):
  """Builds TF graphs for tensor forest training and inference."""

  def __init__(self,
               params,
               configs,
               tree_configs=None):
    self._params = params
    self._configs = configs
    self._variables = tensor_forest_ops.ForestVariables(
        self._params,
        tree_configs=tree_configs)

  def inference_graph(self, dense_features):
    """ Builds inference graphs"""
    # Get predictions from all the trees
    logits = [
        gen_tensor_forest_ops.tensor_forest_tree_predict(
            tree_variable,
            dense_features,
            self._params.logits_dimension)
        for tree_variable in self._variables
    ]

    # shape of all_predict should be [batch_size, n_trees, logits_dimension]
    all_predict = array_ops.stack(logits, axis=1)

    average_values = math_ops.reduce_mean(
        all_predict, name='logits', axis=1)
    expected_squares = math_ops.reduce_mean(
        all_predict * all_predict, axis=1)
    logit_variance = math_ops.maximum(
        0., expected_squares - average_values * average_values)

    return average_values, logit_variance

  def average_size(self):
    sizes = [gen_tensor_forest_ops.tensor_forest_tree_size(tree_variable)
             for tree_variable in self._variables]
    return math_ops.reduce_mean(math_ops.to_float(array_ops.stack(sizes)))


def _tensor_forest_model_fn(features,
                            labels,
                            mode,
                            head,
                            sorted_feature_columns,
                            forest_hparams,
                            config,
                            name='tensor_forest'):
  """Tensor forest model function"""
  with ops.name_scope(name):
    graph_builder = TensorForestGraphs(
        forest_hparams, config)
    # pylint: disable=protected-access
    transformed_features = feature_column_lib._transform_features(
        features, sorted_feature_columns)

    dense_features = array_ops.concat(transformed_features.values(), axis=1)

    logits, logits_variance = graph_builder.inference_graph(
        dense_features)

    summary.scalar('average_tree_size', graph_builder.average_size())

  training_graph = None

  def _train_op_fn(unused_loss):
    del unused_loss
    return training_graph

  estimator_spec = head.create_estimator_spec(
      features=features,
      mode=mode,
      labels=labels,
      train_op_fn=_train_op_fn,
      logits=logits)

  logits_after_activation = estimator_spec.predictions[
      prediction_keys.PredictionKeys.PREDICTIONS]

  average_values = math_ops.reduce_mean(
      logits_after_activation, name='avg_probs', axis=1)
  expected_squares = math_ops.reduce_mean(
      logits_after_activation * logits_after_activation, axis=1)
  var_after_activation = math_ops.maximum(
      0., expected_squares - average_values * average_values)

  estimator_spec.predictions[VARIANCE_LOGITS_KEY] = logits_variance
  estimator_spec.predictions[
      VARIANCE_AFTER_ACTIVATION_KEY] = var_after_activation

  return estimator_spec


class TensorForestClassifier(estimator.Estimator):
  """ TensorForest Classifier """

  def __init__(self,
               feature_columns,
               model_dir=None,
               n_classes=2,
               label_vocabulary=None,
               head=None,
               n_trees=100,
               max_nodes=1000,
               num_splits_to_consider=None,
               split_node_after_samples=250,
               config=None):
    """Initializes a `TensorForestClassifier` instance.

    Example:

    ```python
    feature_1 = numeric_column('feature_1')
    feature_2 = numeric_column('feature_2')

    classifier = estimator.TensorForestClassifier(feature_columns=[feature_1,
                                                feature_2],
						model_dir=None,
						n_classes=2,
						label_vocabulary=None,
						head=None,
						n_trees=100,
						max_nodes=1000,
						num_splits_to_consider=10,
						split_after_samples=250,
						config=None)


    def input_fn_train():
        ...
        return dataset

    classifier.train(input_fn=input_fn_train)

    def input_fn_predict():
        ...
        return dataset

    classifier.predict(input_fn=input_fn_predict)

    def input_fn_eval():
        ...
        return dataset

    metrics = classifier.evaluate(input_fn=input_fn_eval)
    ```

    Args:
      feature_columns: An iterable containing all the feature columns used
        by the model. All items in the set should be instances of classes
        derived from FeatureColumn.
      n_classes: Defaults to 2. The number of classes in a classification
        problem.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into an estimator
        to continue training a previously saved model.
      label_vocabulary: A list of strings representing all possible label
        values. If provided, labels must be of string type and their values
        must be present in label_vocabulary list. If label_vocabulary is
        omitted, it is assumed that the labels are already encoded as integer
        values within {0, 1} for `n_classes=2`, or encoded as integer values
        in {0, 1,..., n_classes-1} for `n_classes>2`. If vocabulary is not
        provided and labels are of string, an error will be generated.
      head: A `canned.Head` instance, the loss would be calculated for
        metrics purpose and not being used for training. If not provided,
        one will be automatically created based on `n_classes`.
      n_trees: The number of trees to create. Defaults to 100. There usually
        isn't any accuracy gain from using higher values (assuming deep
        enough trees are built).
      max_nodes: Default to 10k. No tree is allowed to grow beyond max_nodes
        nodes, and training stops when all trees in the forest are this large.
        This corresponds to a tree of approx depth 13.
      num_splits_to_consider: Defaults to sqrt(num_features). In the extremely
        randomized tree training algorithm, only this many potential splits
        are evaluated for each tree node.
      split_node_after_samples: Defaults to 250. In our online version of
        extremely randomized tree training, we pick a split for a node after
        it has accumulated this many training samples.
      config: RunConfig object to configure the runtime settings.

    Returns:
      A `TensorForestClassifier` instance.
    """

    if head is None:
      # pylint: disable=protected-access
      head = head_lib._binary_logistic_or_multi_class_head(
          n_classes=n_classes,
          weight_column=None,
          label_vocabulary=label_vocabulary,
          loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)
    else:
      if n_classes > 2:
        assert head.logits_dimension == n_classes, 'head logits_dimension: %s ' \
            % head.logits_dimension + \
            'should match n_classes: %s' % n_classes

    for fc in feature_columns:
        assert len(fc.shape) == 1, \
            'Only rank 2 Columns are supported for now, %s violates it' % fc

    num_features = sum(fc.shape[0] for fc in feature_columns)

    forest_hparams = _ForestHParams(
        head.logits_dimension,
        n_trees,
        max_nodes,
        num_splits_to_consider or int(
            math.sqrt(num_features)),
        split_node_after_samples,
        is_regression=False)

    sorted_feature_columns = sorted(feature_columns, key=lambda fc: fc.name)

    def _model_fn(features, labels, mode, config):
      return _tensor_forest_model_fn(  # pylint: disable=protected-access
          features, labels, mode, head,
          sorted_feature_columns, forest_hparams, config)

    super(TensorForestClassifier, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)
