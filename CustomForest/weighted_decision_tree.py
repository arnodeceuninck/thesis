from __future__ import division, print_function
import numpy as np
import math

from .utils import divide_on_feature, train_test_split, standardize, mean_squared_error
from .utils import calculate_entropy, accuracy_score, calculate_variance


class WeightedDecisionNode():
    """Class that represents a decision node or leaf in the decision tree
    Parameters:
    -----------
    feature_i: int
        Feature index which we want to use as the threshold measure.
    threshold: float
        The value that we will compare feature values at feature_i against to
        determine the prediction.
    value: float
        The class prediction if classification tree, or float value if regression tree.
    true_branch: DecisionNode
        Next decision node for samples where features value met the threshold.
    false_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    """

    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i  # Index for the feature that is tested
        self.threshold = threshold  # Threshold value for feature
        self.value = value  # Value if the node is a leaf in the tree
        self.true_branch = true_branch  # 'Left' subtree
        self.false_branch = false_branch  # 'Right' subtree


def divide_on_feature_nan_to_both(X, feature_i, threshold):
    """ Divide dataset based on if sample value on feature index is larger than
        the given threshold """
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    # check if split func or nan
    X_1 = np.array([sample for sample in X if (np.isnan(sample[feature_i]) or split_func(sample))])
    X_2 = np.array([sample for sample in X if (np.isnan(sample[feature_i]) or not split_func(sample))])

    return np.array([X_1, X_2], dtype=object)


# Super class of RegressionTree and ClassificationTree
class WeightedDecisionTree(object):
    """Super class of RegressionTree and ClassificationTree.
    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    loss: function
        Loss function that is used for Gradient Boosting models to calculate impurity.
    """

    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root = None  # Root node in dec. tree
        # Minimum n of samples to justify split
        self.min_samples_split = min_samples_split
        # The minimum impurity to justify split
        self.min_impurity = min_impurity
        # The maximum depth to grow the tree to
        self.max_depth = max_depth
        # Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
        self._impurity_calculation = None
        # Function to determine prediction of y at leaf
        self._leaf_value_calculation = None
        # If y is one-hot encoded (multi-dim) or not (one-dim)
        self.one_dim = None
        # If Gradient Boost
        self.loss = loss

    def fit(self, X, y, loss=None):
        """ Build decision tree """
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)
        self.loss = None

    def _build_tree(self, X, y, weights=None, current_depth=0):
        """ Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data"""

        largest_impurity = 0
        best_criteria = None  # Feature index and threshold
        best_sets = None  # Subsets of the data

        if weights is None:
            assert current_depth == 0
            weights = np.ones(len(y))
            weights = np.expand_dims(weights, axis=1)

        # Check if expansion of y is needed
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        # Add y as last column of X
        Xyw = np.concatenate((X, y, weights), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Calculate the impurity for each feature
            for feature_i in range(n_features):
                # All values of feature_i
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                try:
                    unique_values = unique_values[~np.isnan(unique_values)]  # Filter np.nan values
                except TypeError: # as e:
                    # print(e)
                    # print(unique_values)
                    # raise e
                    unique_values = unique_values[~np.isnan(unique_values.astype(float))]  # Filter np.nan values

                # half the weight if feature_i is None for given sample
                current_weights = Xyw[:, -1:]
                try:
                    current_weights[np.isnan(feature_values.flatten())] /= 2
                except TypeError:
                    current_weights[np.isnan(feature_values.flatten().astype(float))] /= 2
                # reshape
                # current_weights = np.expand_dims(current_weights, axis=1)

                # create a copy of Xyw with the current weights
                Xyw_copy = np.concatenate((Xyw[:, :-1], current_weights), axis=1)

                # Iterate through all unique values of feature column i and
                # calculate the impurity
                for threshold in unique_values:
                    # Divide X and y depending on if the feature value of X at index feature_i
                    # meets the threshold
                    Xyw1, Xyw2 = divide_on_feature_nan_to_both(Xyw_copy, feature_i, threshold)

                    if len(Xyw1) > 0 and len(Xyw2) > 0:
                        # Select the y-values of the two sets
                        y1 = Xyw1[:, n_features:-1]
                        y2 = Xyw2[:, n_features:-1]

                        y1_weights = Xyw1[:, -1:]
                        y2_weights = Xyw2[:, -1:]

                        # Calculate impurity
                        impurity = self._impurity_calculation(y, y1, y2, current_weights, y1_weights, y2_weights)

                        # If this threshold resulted in a higher information gain than previously
                        # recorded save the threshold value and the feature
                        # index
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            # print("best_criteria:", best_criteria)
                            x1 = Xyw1[:, :n_features]
                            x2 = Xyw2[:, :n_features]
                            best_sets = {
                                "leftX": x1,  # X of left subtree
                                "lefty": y1,  # y of left subtree
                                "left_weights": y1_weights,
                                "rightX": x2,  # X of right subtree
                                "righty": y2,  # y of right subtree
                                "right_weights": y2_weights
                            }

        if largest_impurity > self.min_impurity:
            assert best_criteria["feature_i"] is not None
            # Build subtrees for the right and left branches
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], weights=best_sets["left_weights"],
                                           current_depth=current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"],
                                            weights=best_sets["right_weights"], current_depth=current_depth + 1)
            return WeightedDecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                "threshold"], true_branch=true_branch, false_branch=false_branch)

        # We're at leaf => determine value
        leaf_value = self._leaf_value_calculation(y)

        return WeightedDecisionNode(value=leaf_value)

    def predict_value(self, x, tree=None):
        """ Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at """

        if tree is None:
            tree = self.root

        # If we have a value (i.e we're at a leaf) => return value as the prediction
        if tree.value is not None:
            return tree.value

        # Choose the feature that we will test
        feature_value = x[tree.feature_i]

        # Determine if we will follow left or right branch
        branch = tree.false_branch
        try:
            if isinstance(feature_value, int) or isinstance(feature_value, float):
                if feature_value >= tree.threshold:
                    branch = tree.true_branch
            elif feature_value == tree.threshold:
                branch = tree.true_branch
        except ValueError as e:
            print("feature_value:", feature_value)
            print("tree.threshold:", tree.threshold)
            raise e

        # Test subtree
        return self.predict_value(x, branch)

    def predict(self, X):
        """ Classify samples one by one and return the set of labels """
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        """ Recursively print the decision tree """
        if not tree:
            tree = self.root

        # If we're at leaf => print the label
        if tree.value is not None:
            print(tree.value)
        # Go deeper down the tree
        else:
            # Print test
            print("%s:%s? " % (tree.feature_i, tree.threshold))
            # Print the true scenario
            print("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            # Print the false scenario
            print("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)


class XGBoostRegressionTree(WeightedDecisionTree):
    """
    Regression tree for XGBoost
    - Reference -
    http://xgboost.readthedocs.io/en/latest/model.html
    """

    def _split(self, y):
        """ y contains y_true in left half of the middle column and
        y_pred in the right half. Split and return the two matrices """
        col = int(np.shape(y)[1] / 2)
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred

    def _gain(self, y, y_pred):
        nominator = np.power((y * self.loss.gradient(y, y_pred)).sum(), 2)
        denominator = self.loss.hess(y, y_pred).sum()
        return 0.5 * (nominator / denominator)

    def _gain_by_taylor(self, y, y1, y2):
        # Split
        y, y_pred = self._split(y)
        y1, y1_pred = self._split(y1)
        y2, y2_pred = self._split(y2)

        true_gain = self._gain(y1, y1_pred)
        false_gain = self._gain(y2, y2_pred)
        gain = self._gain(y, y_pred)
        return true_gain + false_gain - gain

    def _approximate_update(self, y):
        # y split into y, y_pred
        y, y_pred = self._split(y)
        # Newton's Method
        gradient = np.sum(y * self.loss.gradient(y, y_pred), axis=0)
        hessian = np.sum(self.loss.hess(y, y_pred), axis=0)
        update_approximation = gradient / hessian

        return update_approximation

    def fit(self, X, y):
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = self._approximate_update
        super(XGBoostRegressionTree, self).fit(X, y)


class RegressionTree(WeightedDecisionTree):
    def _calculate_variance_reduction(self, y, y1, y2):
        var_tot = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)

        # Calculate the variance reduction
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)

        return sum(variance_reduction)

    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree, self).fit(X, y)


def calculate_weighted_entropy(y, weights):
    """ Calculate the weighted entropy of label array y """
    assert len(y) == len(weights)
    # Counts diff from normal frequency in most occuring class
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    normalized_weights = weights / np.sum(weights)
    entropy = 0
    for label in unique_labels:
        p = np.sum(normalized_weights[y == label])
        entropy += -p * log2(p)
    return entropy


class WeightedClassificationTree(WeightedDecisionTree):

    def _calculate_information_gain_weighted(self, y, y1, y2, y_weights, y1_weights, y2_weights):
        # Calculate information gain
        p = len(y1) / len(y)
        entropy = calculate_weighted_entropy(y, y_weights)
        info_gain = entropy - p * \
                    calculate_weighted_entropy(y1, y1_weights) - (1 - p) * \
                    calculate_weighted_entropy(y2, y2_weights)

        return info_gain

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # Count number of occurences of samples with label
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_information_gain_weighted
        self._leaf_value_calculation = self._majority_vote
        super(WeightedClassificationTree, self).fit(X, y)
