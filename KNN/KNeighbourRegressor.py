import numpy as np
from sklearn.metrics import mean_squared_error

from base.estimator import BaseEstimator


class KNNRegressor(BaseEstimator):
    def __init__(self, k=5, weights='uniform', metric='euclidean'):
        """
        :param k: number of neighbours
        :param weights: 'uniform' or 'distance'
        :param metric: 'Euclidean' or 'manhattan'
        """
        self.k = k
        self.weights = weights
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        pred = []
        for x in X:
            # Calculate distances
            if self.metric == 'euclidean':
                distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            elif self.metric == 'manhattan':
                distances = np.sum(np.abs(self.X_train - x), axis=1)

            # Get k nearest neighbors
            k_indices = np.argpartition(distances, self.k)[:self.k]
            k_distances = distances[k_indices]
            k_labels = self.y_train[k_indices]

            # Calculate weighted average if needed
            if self.weights == 'distance':
                weights = 1 / (k_distances + 1e-8)  # Avoid division by zero
                pred.append(np.average(k_labels, weights=weights))
            else:
                pred.append(np.mean(k_labels))

        return np.array(pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)  # Negative MSE for consistency with sklearn