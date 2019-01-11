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
"""Tests boosted_trees estimators and model_fn."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow_estimator.python.estimator.canned import boosted_trees_utils


class BoostedTreesDFCTest(test_util.TensorFlowTestCase):
  """Test directional feature contributions (DFC) helper functions."""

  def testDirectionalFeatureContributionsCompute(self):
    """Tests logic to compute DFCs given feature ids and logits paths."""
    # Includes one unused feature (f_1).
    feature_col_names = ('f_0', 'f_1', 'f_2')
    examples_feature_ids = ((2, 2, 0, 0), (2, 2, 0))
    e1_feature_ids, e2_feature_ids = examples_feature_ids

    # DFCs are computed by traversing the prediction path and subtracting each
    # child prediction from its parent prediction and associating the change in
    # prediction with the respective feature id used for the split.
    # For each activation function, f, (currently identity or sigmoid), DFCs are
    # calculated for the two examples as:
    # example 1:
    #   feature_0 = (f(1.114) - f(1.214)) + (f(6.114) - f(1.114))
    #   feature_1 = 0  # Feature not in ensemble, thus zero contrib.
    #   feature_2 = (f(0.114) - bias_pred) + (f(1.214) - f(0.114))
    # example 2:
    #   feature_0 = f(-5.486) - f(1.514)
    #   feature_1 = 0  # Feature not in ensemble, thus zero contrib.
    #   feature_2 = (f(0.114) - bias_pred) + (f(1.514) - f(0.114))
    # where bias_pred is = f(0) or f(0.21), with center_bias = {True, False},
    # respectively.
    # Keys are center_bias.
    expected_dfcs_identity = {
        False: ({
            'f_0': 4.9,
            'f_1': 0,
            'f_2': 1.214
        }, {
            'f_0': -7.0,
            'f_1': 0,
            'f_2': 1.514
        }),
        True: ({
            'f_0': 4.9,
            'f_1': 0,
            'f_2': 1.0039999999999998
        }, {
            'f_0': -7.0,
            'f_1': 0,
            'f_2': 1.3039999999999998
        })
    }
    expected_dfcs_sigmoid = {
        False: ({
            'f_0': 0.22678725678805578,
            'f_1': 0,
            'f_2': 0.2710059376234506
        }, {
            'f_0': -0.81552596670046507,
            'f_1': 0,
            'f_2': 0.319653250251275
        }),
        True: ({
            'f_0': 0.22678725678805578,
            'f_1': 0,
            'f_2': 0.2186980280491253
        }, {
            'f_0': -0.81552596670046507,
            'f_1': 0,
            'f_2': 0.26734534067694971
        })
    }
    # pylint: disable=protected-access
    for f, expected_dfcs in zip(
        (boosted_trees_utils._identity, boosted_trees_utils._sigmoid),
        (expected_dfcs_identity, expected_dfcs_sigmoid)):
      for center_bias in [False, True]:
        # If not center_bias, the bias after activation is 0.
        if center_bias:
          bias_logit = 0.21  # Root node of tree_0.
        else:
          bias_logit = 0  # 0 is default value when there is no original_leaf.
        f_bias = f(bias_logit)

        # Logits before and after, as is outputed from
        # boosted_trees_ops.example_debug_outputs
        examples_logits_paths = ((bias_logit, 0.114, 1.214, 1.114, 6.114),
                                 (bias_logit, 0.114, 1.514, -5.486))
        e1_logits_path, e2_logits_path = examples_logits_paths
        e1_expected_dfcs, e2_expected_dfcs = expected_dfcs[center_bias]
        # Check feature contributions are correct for both examples.
        # Example 1.
        # pylint:disable=line-too-long
        e1_bias, e1_dfc = boosted_trees_utils._compute_directional_feature_contributions(
            e1_feature_ids, e1_logits_path, f, feature_col_names)
        self.assertAllClose(e1_bias, f_bias)
        self.assertAllClose(e1_dfc, e1_expected_dfcs)
        # Example 2.
        e2_bias, e2_dfc = boosted_trees_utils._compute_directional_feature_contributions(
            e2_feature_ids, e2_logits_path, f, feature_col_names)
        # pylint:enable=line-too-long
        self.assertAllClose(e2_bias, f_bias)
        self.assertAllClose(e2_dfc, e2_expected_dfcs)
        # Check if contributions sum to final prediction.
        # For each tree, get leaf of last tree.
        expected_logits = (e1_logits_path[-1], e2_logits_path[-1])
        # Predictions should be the sum of contributions + bias.
        expected_preds = [f(logit) for logit in expected_logits]
        e1_pred = e1_bias + sum(e1_dfc.values())
        e2_pred = e2_bias + sum(e2_dfc.values())
        preds = [e1_pred, e2_pred]
        self.assertAllClose(preds, expected_preds)
    # pylint: enable=protected-access

  def testDFCComputeComparedToExternalExample(self):
    """Tests `compute_dfc` compared to external example (regression).

    Example from http://blog.datadive.net/interpreting-random-forests.
    """
    feature_col_names = ('DIS', 'LSTAT', 'NOX', 'RM')
    e1_feature_ids = (3, 1, 2)
    e2_feature_ids = (3, 3, 3)
    e3_feature_ids = (3, 3, 2)

    bias_logit = 22.60  # Root node of tree_0.
    activation = boosted_trees_utils._identity
    f_bias = activation(bias_logit)
    # Logits before and after, as is outputed from
    # boosted_trees_ops.example_debug_outputs
    e1_logits_path = (bias_logit, 19.96, 14.91, 18.11)
    e2_logits_path = (bias_logit, 37.42, 45.10, 45.90)
    e3_logits_path = (bias_logit, 37.42, 32.30, 33.58)
    e1_expected_dfcs = {'NOX': 3.20, 'LSTAT': -5.05, 'RM': -2.64, 'DIS': 0}
    e2_expected_dfcs = {'NOX': 0, 'LSTAT': 0, 'RM': 23.3, 'DIS': 0}
    e3_expected_dfcs = {'NOX': 1.28, 'LSTAT': 0, 'RM': 9.7, 'DIS': 0}
    # Check feature contributions are correct for both examples.
    # Example 1.
    # pylint: disable=protected-access
    # pylint: disable=line-too-long
    e1_bias, e1_dfc = boosted_trees_utils._compute_directional_feature_contributions(
        e1_feature_ids, e1_logits_path, activation, feature_col_names)
    self.assertAllClose(e1_bias, f_bias)
    self.assertAllClose(e1_dfc, e1_expected_dfcs)
    # Example 2.
    e2_bias, e2_dfc = boosted_trees_utils._compute_directional_feature_contributions(
        e2_feature_ids, e2_logits_path, activation, feature_col_names)
    self.assertAllClose(e2_bias, f_bias)
    self.assertAllClose(e2_dfc, e2_expected_dfcs)
    # Example 3.
    e3_bias, e3_dfc = boosted_trees_utils._compute_directional_feature_contributions(
        e3_feature_ids, e3_logits_path, activation, feature_col_names)

    # pylint: enable=line-too-long
    self.assertAllClose(e3_bias, f_bias)
    self.assertAllClose(e3_dfc, e3_expected_dfcs)
    # pylint: enable=protected-access
    # Check if contributions sum to final prediction.
    # For each tree, get leaf of last tree.
    expected_logits = (18.11, 45.90, 33.58)
    # Predictions should be the sum of contributions + bias.
    expected_preds = [activation(logit) for logit in expected_logits]
    e1_pred = e1_bias + sum(e1_dfc.values())
    e2_pred = e2_bias + sum(e2_dfc.values())
    e3_pred = e3_bias + sum(e3_dfc.values())
    preds = [e1_pred, e2_pred, e3_pred]
    self.assertAllClose(preds, expected_preds)

  def testDFCGroupByFeatureColNameAndSumUsingExternalExample(self):
    """Tests grouping by feature column name and summing contributions.

    Example modified from http://blog.datadive.net/interpreting-random-forests.
    """
    # Use features from testDFCComputeComparedToExternalExample, but merge
    # 'NOX' and 'RM' into 'NOX'.
    feature_col_names = ('DIS', 'LSTAT', 'NOX', 'NOX')
    e1_feature_ids = (2, 1, 2)
    e2_feature_ids = (2, 2, 2)
    e3_feature_ids = (2, 2, 2)

    bias_logit = 22.60  # Root node of tree_0.
    activation = boosted_trees_utils._identity
    f_bias = activation(bias_logit)
    # Logits before and after, as is outputed from
    # boosted_trees_ops.example_debug_outputs
    e1_logits_path = (bias_logit, 19.96, 14.91, 18.11)
    e2_logits_path = (bias_logit, 37.42, 45.10, 45.90)
    e3_logits_path = (bias_logit, 37.42, 32.30, 33.58)
    e1_expected_dfcs = {'NOX': 3.20 - 2.64, 'LSTAT': -5.05, 'DIS': 0}
    e2_expected_dfcs = {'NOX': 0 + 23.3, 'LSTAT': 0, 'DIS': 0}
    e3_expected_dfcs = {'NOX': 1.28 + 9.7, 'LSTAT': 0, 'DIS': 0}
    # Check feature contributions are correct for both examples.
    # Example 1.
    # pylint: disable=protected-access
    # pylint: disable=line-too-long
    e1_bias, e1_dfc = boosted_trees_utils._compute_directional_feature_contributions(
        e1_feature_ids, e1_logits_path, activation, feature_col_names)
    self.assertAllClose(e1_bias, f_bias)
    self.assertAllClose(e1_dfc, e1_expected_dfcs)
    # Example 2.
    e2_bias, e2_dfc = boosted_trees_utils._compute_directional_feature_contributions(
        e2_feature_ids, e2_logits_path, activation, feature_col_names)
    self.assertAllClose(e2_bias, f_bias)
    self.assertAllClose(e2_dfc, e2_expected_dfcs)
    # Example 3.
    e3_bias, e3_dfc = boosted_trees_utils._compute_directional_feature_contributions(
        e3_feature_ids, e3_logits_path, activation, feature_col_names)
    # pylint: enable=line-too-long
    self.assertAllClose(e3_bias, f_bias)
    self.assertAllClose(e3_dfc, e3_expected_dfcs)
    # pylint: enable=protected-access
    # Check if contributions sum to final prediction.
    # For each tree, get leaf of last tree.
    expected_logits = (18.11, 45.90, 33.58)
    # Predictions should be the sum of contributions + bias.
    expected_preds = [activation(logit) for logit in expected_logits]
    e1_pred = e1_bias + sum(e1_dfc.values())
    e2_pred = e2_bias + sum(e2_dfc.values())
    e3_pred = e3_bias + sum(e3_dfc.values())
    preds = [e1_pred, e2_pred, e3_pred]
    self.assertAllClose(preds, expected_preds)


if __name__ == '__main__':
  googletest.main()
