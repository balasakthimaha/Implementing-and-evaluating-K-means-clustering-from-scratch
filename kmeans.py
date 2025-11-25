import numpy as np

class KMeans:
    def __init__(self, k, max_iters=300):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        np.random.seed(42)
        random_indices = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            clusters = self._create_clusters(X)
            new_centroids = self._calculate_centroids(X, clusters)
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids

        self.clusters = clusters

    def _create_clusters(self, X):
        clusters = [[] for _ in range(self.k)]
        for idx, point in enumerate(X):
            centroid_idx = self._closest_centroid(point)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, point):
        distances = np.linalg.norm(point - self.centroids, axis=1)
        return np.argmin(distances)

    def _calculate_centroids(self, X, clusters):
        centroids = np.zeros((self.k, X.shape[1]))
        for i, cluster in enumerate(clusters):
            centroids[i] = np.mean(X[cluster], axis=0)
        return centroids

    def wcss(self, X):
        wcss = 0
        for idx, cluster in enumerate(self.clusters):
            distances = np.linalg.norm(X[cluster] - self.centroids[idx], axis=1)
            wcss += np.sum(distances**2)
        return wcss
