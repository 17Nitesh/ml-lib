import numpy as np
from base.estimator import BaseEstimator


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression(BaseEstimator):
    def __init__(self, lr=0.01, epochs=1000):
        """
        :param lr: learning rate
        :param epochs: no. of iterations
        """
        self.weights = None
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.weights = np.random.randn(X.shape[1])

        for _ in range(self.epochs):
            y_hat = _sigmoid(np.dot(X, self.weights))
            self.weights += self.lr * (np.dot((y - y_hat), X)/X.shape[0])
        return self

    def predict_proba(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return _sigmoid(X @ self.weights)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
