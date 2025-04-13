import copy
import numpy as np

class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature_idx = None
        self.feature_value = None
        self.node_prediction = None

    def gini(self, y):
        if len(y) == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def calculate_gain(self, y_left, y_right):
        weight_left = len(y_left) / (len(y_left) + len(y_right))
        weight_right = 1 - weight_left
        gain = 1 - (weight_left * self.gini(y_left) + weight_right * self.gini(y_right))
        return gain

    def split_data(self, X, y, idx, val):
        left_mask = X[:, idx] < val
        return (X[left_mask], y[left_mask]), (X[~left_mask], y[~left_mask])

    def find_possible_splits(self, data):
        possible_split_points = []
        for idx in range(data.shape[0] - 1):
            if data[idx] != data[idx + 1]:
                possible_split_points.append((data[idx] + data[idx + 1]) / 2)
        return possible_split_points

    def find_best_split(self, X, y, feature_subset=None):
        best_gain = -np.inf
        best_split = None

        if feature_subset is not None:
            features = np.random.choice(X.shape[1], feature_subset, replace=False)
        else:
            features = range(X.shape[1])

        for d in features:
            order = np.argsort(X[:, d])
            X_sorted, y_sorted = X[order], y[order]
            possible_splits = self.find_possible_splits(X_sorted[:, d])
            for split_value in possible_splits:
                (X_left, y_left), (X_right, y_right) = self.split_data(X_sorted, y_sorted, d, split_value)
                gain = self.calculate_gain(y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (d, split_value)

        if best_split is None:
            return None, None

        return best_split[0], best_split[1]

    def predict(self, x):
        if self.feature_idx is None:
            return self.node_prediction
        if x[self.feature_idx] < self.feature_value:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)

    def train(self, X, y, params):
        self.node_prediction = np.mean(y)
        if X.shape[0] == 1 or len(np.unique(y)) == 1:
            return True

        self.feature_idx, self.feature_value = self.find_best_split(X, y, params.get("feature_subset", None))
        if self.feature_idx is None:
            return True

        (X_left, y_left), (X_right, y_right) = self.split_data(X, y, self.feature_idx, self.feature_value)

        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            self.feature_idx = None
            return True

        # Maksymalna głębokość drzewa
        if params.get("depth", None) is not None:
            params["depth"] -= 1
        if params.get("depth", None) == 0:
            self.feature_idx = None
            return True

        # Tworzenie nowych węzłów
        self.left_child, self.right_child = Node(), Node()
        self.left_child.train(X_left, y_left, copy.deepcopy(params))
        self.right_child.train(X_right, y_right, copy.deepcopy(params))
