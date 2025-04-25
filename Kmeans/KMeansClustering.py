import numpy as np
import random

def _move_centroids(X, cluster_group):
    new_centroids = []
    cluster_group = np.array(cluster_group)

    for cluster_id in np.unique(cluster_group):
        points = X[cluster_group == cluster_id]
        centroid = points.mean(axis=0)
        new_centroids.append(centroid)

    return np.array(new_centroids)


class KMeansClustering:
    def __init__(self, n_clusters=2, max_iter=100):
        """
        :param n_clusters: number of clusters
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X):
        random_index = random.sample(range(len(X)), self.n_clusters)
        self.centroids = X[random_index]

        for i in range(self.max_iter):
            cluster_group = self._assign_clusters(X)
            old_centroids = self.centroids
            self.centroids = _move_centroids(X, cluster_group)
            if np.allclose(self.centroids, old_centroids):
                break

        return cluster_group

    def _assign_clusters(self, X):
        cluster_group = []

        for row in X:
            distances = [np.linalg.norm(row - centroid) for centroid in self.centroids]
            closest_cluster = np.argmin(distances)
            cluster_group.append(closest_cluster)

        return cluster_group
