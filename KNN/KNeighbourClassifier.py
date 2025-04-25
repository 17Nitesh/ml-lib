import numpy as np
from collections import Counter
from base.estimator import BaseEstimator

class KNNClassifier(BaseEstimator):

    def __init__(self, k):
        """
        :param k: number of neighbours
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        pred = []
        for i in X:
            distances = []
            for j in self.X_train:
                distances.append(np.linalg.norm(i - j))
            n_neighbours = sorted(list(enumerate(distances)), key=lambda x: x[1])[0:self.k]
            label = self._majority_count(n_neighbours)
            pred.append(label)
        return np.array(pred)



    def _majority_count(self, n_neighbours):
        votes = []

        for i in n_neighbours:
            votes.append(self.y_train[i[0]])

        votes = Counter(votes)
        return votes.most_common()[0][0]