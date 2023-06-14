from functools import lru_cache
import numpy as np
from collections import Counter
from sklearn.metrics import pairwise_distances


class KNNString:
    def __init__(self, k, metric, metric_kwargs=None):
        self.k = k
        self.metric = metric
        self.metric_kwargs = metric_kwargs if metric_kwargs is not None else {}

    def fit(self, X, y):
        self.X_train = X.reset_index(drop=True)
        self.y_train = y.reset_index(drop=True)
        self.classes_ = np.unique(y)

    def predict(self, X):
        y_pred = []
        for i, row_to_predict in X.iterrows():
            distances = []
            for j, row_train in self.X_train.iterrows():
                dist = self.metric(row_train, row_to_predict, **self.metric_kwargs)
                distances.append(dist)
            indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            y_pred.append(most_common[0][0])
        result = np.array(y_pred)
        assert len(result) == len(X)
        return result

    def predict_proba(self, X):
        y_pred_proba = []
        for _, row_to_predict in X.iterrows():
            distances = []
            for j, row_train in self.X_train.iterrows():
                dist = self.metric(row_train, row_to_predict, **self.metric_kwargs)
                distances.append(dist)
            indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in indices]
            label_counts = Counter(k_nearest_labels)
            probabilities = [label_counts[label] / self.k for label in self.classes_]
            len_y = len(y_pred_proba)
            y_pred_proba.append(probabilities)
            assert len(y_pred_proba) == len_y + 1
        result = np.array(y_pred_proba)
        assert result.shape == (len(X), len(self.classes_))
        return result

def levenstein_distance(a, b, nan_distance=0):
    if not isinstance(a, str) or not isinstance(b, str):
        return nan_distance

    @lru_cache(None)  # for memorization
    def min_dist(s1, s2):

        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        # no change required
        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(
            min_dist(s1, s2 + 1),  # insert character
            min_dist(s1 + 1, s2),  # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )

    dist = min_dist(0, 0)
    return dist


def multi_lev(a, b, nan_distance=0):
    return np.mean([levenstein_distance(a_, b_, nan_distance) for a_, b_ in zip(a, b)])


def multi_dist(a, b, metric, metric_kwargs=None):
    if metric_kwargs:
        return np.mean([metric(a_, b_, **metric_kwargs) for a_, b_ in zip(a, b)])
    else:
        return np.mean([metric(a_, b_) for a_, b_ in zip(a, b)])
