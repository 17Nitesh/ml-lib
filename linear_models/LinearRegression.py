import numpy as np
from base.estimator import BaseEstimator

class LinearRegression(BaseEstimator):
    """Linear Regression"""
    def __init__(self, lr=0.001, epochs=5000, alpha=0.0, l1=0.0, l2=0.0):
        """
        :param lr: learning rate
        :param epochs: no. of iterations
        :param alpha: regularization strength
        :param l1: lasso regularization strength
        :param l2: ridge regularization strength
        """
        self.weights = None
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        self.l1 = l1
        self.l2 = l2

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Add bias term
        self.weights = np.random.randn(X.shape[1]) * 0.01
        n = X.shape[0]

        for _ in range(self.epochs):
            y_pred = X @ self.weights
            error = y_pred - y
            gradient = (2/n) * (X.T @ error)

            # Regularization terms
            l1_term = self.l1 * np.sign(self.weights)
            l2_term = self.l2 * self.weights
            l1_term[0] = 0
            l2_term[0] = 0

            reg_term = self.alpha * (l1_term + 2 * l2_term)
            gradient += reg_term

            self.weights -= self.lr * gradient

        return self

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return X @ self.weights