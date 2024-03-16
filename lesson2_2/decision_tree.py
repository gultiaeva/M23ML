# coding:utf-8
import random

import numpy as np
from scipy import stats


def f_entropy(p):
    # Convert values to probability
    p = np.bincount(p) / float(p.shape[0])

    ep = stats.entropy(p)
    if ep == -float("inf"):
        return 0.0
    return ep


def information_gain(y, splits):
    splits_entropy = sum([f_entropy(split) * (float(split.shape[0]) / y.shape[0]) for split in splits])
    return f_entropy(y) - splits_entropy


def mse_criterion(y, splits):
    y_mean = np.mean(y)
    return -sum([np.sum((split - y_mean) ** 2) * (float(split.shape[0]) / y.shape[0]) for split in splits])


def get_split_mask(X, column, value):
    left_mask = X[:, column] < value
    right_mask = X[:, column] >= value
    return left_mask, right_mask


def split(X, y, value):
    left_mask = X < value
    right_mask = X >= value
    return y[left_mask], y[right_mask]


def split_dataset(X, target, column, value, return_X=True):
    left_mask, right_mask = get_split_mask(X, column, value)

    left, right = {}, {}
    for key in target.keys():
        left[key] = target[key][left_mask]
        right[key] = target[key][right_mask]

    if return_X:
        left_X, right_X = X[left_mask], X[right_mask]
        return left_X, right_X, left, right
    else:
        return left, right


class Tree:
    """Recursive implementation of decision tree."""

    def __init__(self, regression=False, criterion=None, n_classes=None):
        self.regression = regression
        self.impurity = None
        self.threshold = None
        self.column_index = None
        self.outcome = None
        self.criterion = criterion
        self.n_classes = n_classes  # Only for classification

        self.left_child = None
        self.right_child = None

    @property
    def is_terminal(self):
        return not bool(self.left_child and self.right_child)

    def _find_splits(self, X):
        """Find all possible split values."""
        split_values = set()

        # Get unique values in a sorted order
        x_unique = list(np.unique(X))
        for i in range(1, len(x_unique)):
            # Find a point between two values
            average = (x_unique[i - 1] + x_unique[i]) / 2.0
            split_values.add(average)

        return list(split_values)

    def _find_best_split(self, X, target, n_features):
        """Find best feature and value for a split. Greedy algorithm."""

        # Sample random subset of features
        subset = random.sample(list(range(0, X.shape[1])), n_features)
        max_gain, max_col, max_val = None, None, None

        for column in subset:
            split_values = self._find_splits(X[:, column])
            for value in split_values:
                splits = split(X[:, column], target["y"], value)
                gain = self.criterion(target["y"], splits)

                if (max_gain is None) or (gain > max_gain):
                    max_col, max_val, max_gain = column, value, gain
        return max_col, max_val, max_gain

    def _train(self, X, target, max_features=None, min_samples_split=10, max_depth=None, minimum_gain=0.01):
        try:
            assert X.shape[0] > min_samples_split
            assert max_depth > 0

            if max_features is None:
                max_features = X.shape[1]

            column, value, gain = self._find_best_split(X, target, max_features)
            assert gain is not None
            if self.regression:
                assert gain != 0
            else:
                assert gain > minimum_gain

            self.column_index = column
            self.threshold = value
            self.impurity = gain

            # Split dataset
            left_X, right_X, left_target, right_target = split_dataset(X, target, column, value)

            # Grow left and right child
            self.left_child = Tree(self.regression, self.criterion, self.n_classes)
            self.left_child._train(
                left_X, left_target, max_features, min_samples_split, max_depth - 1, minimum_gain
            )

            self.right_child = Tree(self.regression, self.criterion, self.n_classes)
            self.right_child._train(
                right_X, right_target, max_features, min_samples_split, max_depth - 1, minimum_gain
            )
        except AssertionError:
            self._calculate_leaf_value(target)

    def train(self, X, target, max_features=None, min_samples_split=10, max_depth=None, minimum_gain=0.01):
        """Build a decision tree from training set.

        Parameters
        ----------

        X : array-like
            Feature dataset.
        target : dictionary or array-like
            Target values.
        max_features : int or None
            The number of features to consider when looking for the best split.
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        max_depth : int
            Maximum depth of the tree.
        minimum_gain : float, default 0.01
            Minimum gain required for splitting.
        """

        if not isinstance(target, dict):
            target = {"y": target}

        if not self.regression:
            self.n_classes = len(np.unique(target['y']))

        self._train(X, target, max_features=max_features, min_samples_split=min_samples_split,
                    max_depth=max_depth, minimum_gain=minimum_gain)


    def _calculate_leaf_value(self, targets):
        """Find optimal value for leaf."""
        if self.regression:
            # Mean value for regression task
            self.outcome = np.mean(targets["y"])
        else:
            # Probability for classification task
            self.outcome = np.bincount(targets["y"], minlength=self.n_classes) / targets["y"].shape[0]

    def predict_row(self, row):
        """Predict single row."""
        if not self.is_terminal:
            if row[self.column_index] < self.threshold:
                return self.left_child.predict_row(row)
            else:
                return self.right_child.predict_row(row)
        return self.outcome

    def predict(self, X):
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            result[i] = self.predict_row(X[i, :])
        return result