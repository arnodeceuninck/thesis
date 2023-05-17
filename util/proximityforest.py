import numpy as np


class Branch:
    """
    A branch in a tree, contains an exemplar and a subtree
    An exemplar is an input data point, if a data point is closest to this exemplar, it goes to the subtree
    A class label is for the leaf nodes
    """
    def __init__(self, exemplar, class_label, subtree):
        self.exemplar = exemplar  # An object, if from a node it is closest to this exemplar, it goes to given tree
        self.subtree = subtree  # Should be an internal node or a leaf node

        self.class_label = class_label  # The class label of the leaf node, not required for the algorithm, but might be useful for debuggin


class TreeNode:
    """
    Node in the proximity tree. This can be either an internal node or a leaf node.
    """
    def predict(self, data):
        raise NotImplementedError()

    def print(self, depth):
        raise NotImplementedError()


class InternalNode(TreeNode):
    """
    Internal node in the proximity tree.
    """
    def __init__(self, data_x, data_y, groups, depth, max_depth, splits_to_sample, distance_measure, distance_kwargs):
        """
        Create & train an internal node

        :param data_x: x values
        :param data_y: target labels
        :param groups: if you subbranches to be from the same group
        :param depth: current depth
        :param max_depth: maximum depth
        :param splits_to_sample: number of splits to sample (the split with the lowest gini impurity is chosen as actual split)
        :param distance_measure: distance measure to use
        :param distance_kwargs: kwargs for the distance measure
        """
        assert len(data_x) == len(data_y) == len(groups)
        assert len(data_x) > 0

        depth += 1

        self.measure = lambda x, y: distance_measure(x, y, **distance_kwargs)
        self.branches = []  # Contains internal nodes or leaf nodes

        exemplars, closest_exemplars = get_split(data_x, data_y, groups, splits_to_sample, self.measure)

        for i, exemplar in enumerate(exemplars):
            closest_data_x = data_x[closest_exemplars == i]
            closest_data_y = data_y[closest_exemplars == i]
            closest_data_groups = groups[closest_exemplars == i]

            if len(closest_data_x) == 0:
                # If everything is closer to the other one, you don't need to create a node with 0 data points
                continue

            subtree = get_node(closest_data_x, closest_data_y, closest_data_groups, depth, max_depth, splits_to_sample, distance_measure, distance_kwargs)
            self.branches.append(Branch(exemplar, i, subtree))  # TODO: Fix label somewhere here instead of i

    def predict(self, data):
        """
        Returns the prediction result of the subbranch that is closest to the given data point
        """
        if len(self.branches) == 1:
            # If there is only one branch, it means that all data points are closer to the exemplar of that branch
            return self.branches[0].subtree.predict(data)

        # Return the label of the closest subtree to the data point
        exemplar_distance = [self.measure(data, branch.exemplar) for branch in self.branches]
        # get the index of the closest exemplar (lowest number)
        closest_exemplar = np.argmin(exemplar_distance)
        # closest_exemplars = get_closest_exemplars(exemplar_distance, [data], self.measure)
        branch = self.branches[closest_exemplar]
        # print(f"Taking branch {branch.class_label} after comparing {data} with {self.branches[0].exemplar} (d={exemplar_distance[0]}) and {self.branches[1].exemplar} (d={exemplar_distance[1]})")
        return branch.subtree.predict(data)

    def print(self, depth):
        """
        Prints the tree in a nice format
        :param depth: current depth
        """
        for branch in self.branches:
            print(f"{'-' * depth}Exemplar ({branch.class_label}): " + str(branch.exemplar))
            branch.subtree.print(depth + 1)


# if all data reaching a node has the same class (node is pure), create_leaf function creates a new leaf node and assigns this class lbel to its field class
class LeafNode(TreeNode):
    def __init__(self, class_label):
        self.class_label = class_label  # This label is assigned to all data reaching this node

    def predict(self, data):
        return self.class_label

    def print(self, depth):
        print(f"{'-' * depth}Leaf ({self.class_label})")


# Splitting criteriaa
class ProximityTreeClassifier:
    """
    Proximity Tree Classifier
    """
    def __init__(self, max_depth=5, num_features_to_keep=None, splits_to_sample=10, distance_measure=None, distance_kwargs=None):
        """

        :param max_depth:
        :param num_features_to_keep: Number of features to keep, if None, all features are used. DO NOT USE THIS WITH TCR DISTANCE. The distance function depends on the position of the features.
        :param splits_to_sample: Number of splits to sample in each node, the split with the lowest gini impurity is chosen as actual split
        :param distance_measure: Distance measure to use, if None, linalg.norm(x-y) is used
        :param distance_kwargs: kwargs for the distance measure
        """
        self.root = None
        self.max_depth = max_depth
        self.num_features_to_keep = num_features_to_keep
        self.features_to_use_indices = None
        self.splits_to_sample = splits_to_sample
        self.distance_measure = distance_measure if distance_measure is not None else lambda x, y: np.linalg.norm(x - y)
        self.distance_kwargs = distance_kwargs if distance_kwargs is not None else {}

    def fit(self, data_x, data_y, groups=None):
        groups = np.zeros(len(data_x)) if groups is None else groups

        self.num_features = data_x.shape[1]  # Total number of features, not the number used!!!

        self.features_to_use_indices = np.random.choice(len(data_x[0]), self.num_features_to_keep,
                                                        replace=False) if self.num_features_to_keep is not None else np.arange(
            len(data_x[0]))
        data_x_reduced = subsample_features(data_x, self.features_to_use_indices)

        # TODO: Remove this line to enable subsampling and bootstrapping
        data_x_reduced = data_x
        # assert self.num_features_to_keep is None, 'Be sure the distance measure supports this!'

        self.root = get_node(data_x_reduced, data_y, groups, depth=0, max_depth=self.max_depth,
                             splits_to_sample=self.splits_to_sample, distance_measure=self.distance_measure,
                             distance_kwargs=self.distance_kwargs)

        return self

    def predict(self, data):
        # Convert data to numpy array if not yet the case
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        # If data is 1d, add a dimension to make it 2d
        if data.ndim == 1:
            data = data[np.newaxis, :]
        assert self.root is not None, "You must fit the model before predicting"
        assert data.shape[
                   1] == self.num_features, "The number of features in the data must match the number of features in the training data"
        data_reduced = subsample_features(data, self.features_to_use_indices)
        predictions = np.apply_along_axis(self.root.predict, 1, data_reduced)
        assert predictions.shape[0] == data.shape[0], "The number of predictions must match the number of data points"
        return predictions

    def print(self):
        self.root.print(0)


def is_pure(data_y):
    # Check if all data has the same class label
    unique_y = np.unique(data_y)
    return len(unique_y) == 1


def get_most_common_element(array):
    return np.argmax(np.bincount(array))


def subsample_features(data_x, feature_indices):
    return data_x[:, feature_indices]


def get_node(data_x, data_y, groups, depth, max_depth, splits_to_sample, distance_measure, distance_kwargs):
    assert len(data_x) == len(data_y) == len(groups)
    assert len(data_x) > 0

    if is_pure(data_y) or depth >= max_depth:
        # class_label = data_y.iloc[0] # must use iloc here to get element of first row and not element at index 0
        class_label = get_most_common_element(data_y)
        return LeafNode(class_label)
    else:
        return InternalNode(data_x, data_y, groups, depth, max_depth, splits_to_sample, distance_measure, distance_kwargs)


# Some util functions
def random_pick_row(data):
    return data[np.random.randint(0, data.shape[0])]
    # return np.random.choice(group_x_with_label) # Can't use this for 2d arrays


def get_from_group_if_exists_else_random(group_x, group_y, data_x, data_y, label):
    # Get all elements of group_x where group_y == label
    group_x_with_label = group_x[group_y == label]
    if group_x_with_label.size != 0:
        # sample random with same label
        return random_pick_row(group_x_with_label)
    else:
        # If none exist, return a random element from data_x where data_y == label
        data_x_with_label = data_x[data_y == label]
        return random_pick_row(data_x_with_label)


def get_group_data(data_x, data_y, groups, group_id):
    # random_group_indices = np.where(groups == group_id)[0]
    # group_x = np.take(data_x, random_group_indices, axis=0)
    # group_y = np.take(data_y, random_group_indices)
    # return group_x, group_y
    return data_x[groups == group_id], data_y[groups == group_id]


def get_single_split(data_x, data_y, groups):
    assert len(data_x) == len(data_y) == len(groups)
    assert len(data_x) != 0

    # # Sample uniformly any of the groups
    # # Get the unique values in groups
    # unique_groups = np.unique(groups)
    # random_group = np.random.choice(unique_groups)
    #
    # random_group_x, random_group_y = get_group_data(data_x, data_y, groups, random_group)

    random_group_x = data_x
    random_group_y = data_y

    assert len(random_group_x) == len(random_group_y) != 0

    exemplar_winner = get_from_group_if_exists_else_random(random_group_x, random_group_y, data_x, data_y, 1)
    exemplar_loser = get_from_group_if_exists_else_random(random_group_x, random_group_y, data_x, data_y, 0)
    exemplars = [exemplar_loser, exemplar_winner]

    return exemplars


def gini(y, classes):
    # Code from https://stackoverflow.com/questions/64741099/how-to-calculate-gini-index-using-two-numpy-arrays
    if not y.shape[0]:
        return 0

    probs = []
    for cls in classes:
        probs.append((y == cls).sum() / y.shape[0])  # For each class c in classes compute class probabilities

    p = np.array(probs)
    return 1 - ((p * p).sum())


def tree_gini_index(Y_left, Y_right):
    classes = (0, 1)

    # Code from https://stackoverflow.com/questions/64741099/how-to-calculate-gini-index-using-two-numpy-arrays
    N = Y_left.shape[0] + Y_right.shape[0]
    p_L = Y_left.shape[0] / N
    p_R = Y_right.shape[0] / N

    return p_L * gini(Y_left, classes) + p_R * gini(Y_right, classes)


def get_split(data_x, data_y, groups, splits_to_sample, distance_measure):
    assert len(data_x) == len(data_y) == len(groups)
    assert len(data_x) != 0

    min_gini_index = np.inf
    for split in range(splits_to_sample):
        exemplars = get_single_split(data_x, data_y, groups)

        closest_exemplars = get_closest_exemplars(exemplars, data_x, distance_measure)
        data_splits = []
        # empty_split = False
        for i, exemplar in enumerate(exemplars):
            closest_data_y = data_y[closest_exemplars == i]
            data_splits.append(closest_data_y)

            # if len(closest_data_y) == 0:
            #     empty_split = True
            #     break

        # if empty_split:
        #     continue

        gini_index = tree_gini_index(data_splits[0], data_splits[1])

        if gini_index < min_gini_index:
            min_gini_index = gini_index
            best_exemplars = exemplars
            best_closest_exemplars = closest_exemplars

    # if min_gini_index == np.inf:
    #     # If no split was found (all containing an empty split), return None
    #     return None, None

    return best_exemplars, best_closest_exemplars


def get_distance_to_exemplars(data_x_row, exemplars, measure):
    return [measure(data_x_row, exemplar) for exemplar in exemplars]


def get_closest_exemplars(exemplars, data, measure):
    # Return the index of the exemplar that is closest to the data point
    distances = [get_distance_to_exemplars(data_row, exemplars, measure) for data_row in data]
    closest_exemplar_indices = np.argmin(distances, axis=1)
    assert len(closest_exemplar_indices) == len(data)
    return closest_exemplar_indices


from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm


class ProximityForestClassifier:
    def __init__(self, n_trees=100, show_progress=True, use_bootstrapping=True, reduce_features=True,
                 sample_multiple_splits=10, max_depth=5, distance_measure=None, distance_kwargs=None, multithreaded=True):
        self.n_trees = n_trees
        self.trees = []
        self.classes_ = None
        self.show_progress = show_progress
        self.use_bootstrapping = use_bootstrapping
        self.reduce_features = reduce_features
        self.sample_multiple_splits = sample_multiple_splits
        self.max_depth = max_depth
        self.distance_measure = distance_measure
        self.distance_kwargs = distance_kwargs
        self.multithreaded = multithreaded

    def fit(self, data_x, data_y, groups=None):
        data_x = self.preprocess_data(data_x)
        data_y = self.preprocess_data(data_y.astype(int))
        groups = self.preprocess_data(groups) if groups is not None else np.zeros(len(data_x))

        if self.multithreaded:
            iterator = tqdm(range(self.n_trees), disable=not self.show_progress, desc='Fitting')
            self.trees = Parallel(n_jobs=3)(delayed(self.fit_tree)(data_x, data_y, groups) for i in iterator)
        else:
            for tree in tqdm(range(self.n_trees), disable=not self.show_progress, desc='Fitting'):
                self.trees.append(self.fit_tree(data_x, data_y, groups))

        self.classes_ = np.unique(data_y)

        return self

    def fit_tree(self, data_x, data_y, groups):
        if self.use_bootstrapping:
            data_x_bootstrap, data_y_bootstrap, groups_bootstrap = self.bootstrap(data_x, data_y, groups)
        else:
            data_x_bootstrap, data_y_bootstrap, groups_bootstrap = data_x, data_y, groups

        if self.reduce_features:
            num_features_to_keep = int(np.sqrt(data_x_bootstrap.shape[1]))
        else:
            num_features_to_keep = None
            # assert num_features_to_keep <= len(data_x[0])

        return ProximityTreeClassifier(num_features_to_keep=num_features_to_keep,
                                       splits_to_sample=self.sample_multiple_splits, max_depth=self.max_depth,
                                       distance_measure=self.distance_measure, distance_kwargs=self.distance_kwargs) \
            .fit(data_x_bootstrap, data_y_bootstrap, groups_bootstrap)

    def bootstrap(self, data_x, data_y, groups):
        # Bootstrap some rows
        indices = np.random.choice(len(data_x), size=int(len(data_x) * 0.1), replace=True)
        data_x_bootstrap = data_x[indices]
        data_y_bootstrap = data_y[indices]
        groups_bootstrap = groups[indices]
        return data_x_bootstrap, data_y_bootstrap, groups_bootstrap

    def get_predictions(self, data):
        data = self.preprocess_data(data)

        if self.multithreaded:
            iterator = tqdm(range(self.n_trees), disable=not self.show_progress, desc='Predicting')
            predictions = Parallel(n_jobs=3)(delayed(self.trees[i].predict)(data) for i in iterator)
        else:
            predictions = []
            for tree in tqdm(self.trees, disable=not self.show_progress, desc='Predicting'):
                predictions.append(tree.predict(data))

        return predictions

    def preprocess_data(self, data):
        # if data is a pandas dataframe, convert to numpy array
        if isinstance(data, pd.DataFrame):
            data = data.replace({True: 1, False: 0})
            data = data.to_numpy()
        if isinstance(data, pd.Series):
            data = data.replace({True: 1, False: 0})
            data = data.to_numpy()
        return data

    def predict_proba(self, data):
        tree_predictions = self.get_predictions(data)
        means = np.mean(tree_predictions, axis=0)
        # expand to 2d array
        return np.stack((1 - means, means), axis=1)

    def predict(self, data):
        tree_predictions = self.get_predictions(data)
        most_occuring = np.apply_along_axis(get_most_common_element, 0, tree_predictions)
        return most_occuring
        # Get most occuring element for each column


def get_most_common_element(array):
    return np.argmax(np.bincount(array))
