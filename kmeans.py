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
            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids
        self.clusters = clusters

    def _create_clusters(self, X):
        clusters = [[] for _ in range(self.k)]
        for idx, point in enumerate(X):
            centroid_idx = np.argmin(np.linalg.norm(point - self.centroids, axis=1))
            clusters[centroid_idx].append(idx)
        return clusters

    def _calculate_centroids(self, X, clusters):
        return np.array([np.mean(X[cluster], axis=0) for cluster in clusters])

    def wcss(self, X):
        total = 0
        for i, cluster in enumerate(self.clusters):
            d = np.linalg.norm(X[cluster] - self.centroids[i], axis=1)
            total += (d**2).sum()
        return total
