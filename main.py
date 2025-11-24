import numpy as np

class KMeansScratch:
    def __init__(self, k, max_iters=300):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        np.random.seed(42)
        random_idxs = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_idxs]

        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, None] - self.centroids[None, :], axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array(
                [X[labels == i].mean(axis=0) if np.any(labels == i) else self.centroids[i]
                 for i in range(self.k)]
            )

            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids

        self.labels = labels
        return labels

    def inertia(self, X):
        return np.sum((X - self.centroids[self.labels])**2)
