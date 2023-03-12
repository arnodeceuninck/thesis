from __future__ import division, print_function
import numpy as np
import math
import progressbar
from tqdm import tqdm

# Import helper functions
from .utils import divide_on_feature, train_test_split, get_random_subsets, normalize, bar_widgets
from .decision_tree import ClassificationTree
from .weighted_decision_tree import WeightedClassificationTree


class RandomForest():
    """Random Forest classifier. Uses a collection of classification trees that
    trains on random subsets of the data using a random subsets of the features.
    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    max_features: int
        The maximum number of features that the classification trees are allowed to
        use.
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_gain: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    """

    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2,
                 min_gain=0, max_depth=float("inf"), weighted=False):
        self.n_estimators = n_estimators  # Number of trees
        self.max_features = max_features  # Maxmimum number of features per tree
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain  # Minimum information gain req. to continue
        self.max_depth = max_depth  # Maximum depth for tree
        self.progressbar = tqdm #progressbar.ProgressBar(widgets=bar_widgets)

        # Initialize decision trees
        self.trees = []
        for _ in range(n_estimators):
            if not weighted:
                self.trees.append(
                    ClassificationTree(
                        min_samples_split=self.min_samples_split,
                        min_impurity=min_gain,
                        max_depth=self.max_depth))
            else:
                self.trees.append(
                    WeightedClassificationTree(
                        min_samples_split=self.min_samples_split,
                        min_impurity=min_gain,
                        max_depth=self.max_depth))

    def fit(self, X, y):
        # if X or y is a DataFrame or Series, convert to numpy array
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        if not isinstance(y, np.ndarray):
            y = y.to_numpy()

        n_features = np.shape(X)[1]
        # If max_features have not been defined => select it as
        # sqrt(n_features)
        if not self.max_features:
            self.max_features = int(math.sqrt(n_features))

        # Choose one random subset of the data for each tree
        subsets = get_random_subsets(X, y, self.n_estimators)

        for i in self.progressbar(range(self.n_estimators)):
            X_subset, y_subset = subsets[i]
            # Feature bagging (select random subsets of the features)
            idx = np.random.choice(range(n_features), size=self.max_features, replace=True)
            # Save the indices of the features for prediction
            self.trees[i].feature_indices = idx
            # Choose the features corresponding to the indices
            X_subset = X_subset[:, idx]
            # Fit the tree to the data
            self.trees[i].fit(X_subset, y_subset)

    def predict(self, X):
        # if X or y is a DataFrame or Series, convert to numpy array
        X = np.array(X)

        # Make predictions based on probability of each class
        y_pred =self.predict_proba(X).argmax(axis=1)
        return y_pred

    def predict_proba(self, X):
        # if X or y is a DataFrame or Series, convert to numpy array
        X = np.array(X)

        # same as predict, but returns the probability of each class
        y_preds = np.empty((X.shape[0], len(self.trees)))
        # Let each tree make a prediction on the data
        for i, tree in enumerate(self.trees):
            # Indices of the features that the tree has trained on
            idx = tree.feature_indices
            # Make a prediction based on those features
            prediction = tree.predict(X[:, idx])
            y_preds[:, i] = prediction

        y_pred = []
        # For each sample
        for sample_predictions in y_preds:
            # Get the relative frequency of each class
            if len(np.unique(np.array(sample_predictions))) == 1:
                unique_value = sample_predictions[0]
                if unique_value == 0:
                    y_pred.append([1,0])
                else:
                    y_pred.append([0,1])
            else:
                pred_count = np.bincount(sample_predictions.astype('int'))
                pred_prob = pred_count/len(sample_predictions)
                y_pred.append(pred_prob)
        return np.stack(y_pred, axis=0)