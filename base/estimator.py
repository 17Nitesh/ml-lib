from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Train model on data"""
        pass
    @abstractmethod
    def predict(self, X):
        """Predict"""
        pass
