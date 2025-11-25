import numpy as np

def initialize_centroids(X, k, random_state=None):
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), k, replace=False)
    return X[idx].astype(float)

def assign_clusters(X, centroids):
    # distances: (n_samples, k)
    distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(distances, axis=1), distances

def update_centroids(X, labels, k):
    centroids = np.zeros((k, X.shape[1]), dtype=float)
    for i in range(k):
        members = X[labels == i]
        if len(members) == 0:
            # empty cluster: will be handled by caller (reinitialize)
            centroids[i] = np.nan
        else:
            centroids[i] = members.mean(axis=0)
    return centroids

def inertia(X, labels, centroids):
    # sum of squared distances to assigned centroid
    ssd = 0.0
    for i in range(centroids.shape[0]):
        members = X[labels == i]
        if len(members) == 0:
            continue
        ssd += np.sum((members - centroids[i])**2)
    return ssd

def kmeans(X, k, max_iters=300, tol=1e-4, random_state=None):
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    centroids = initialize_centroids(X, k, random_state=random_state)
    for it in range(max_iters):
        labels, distances = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        # Handle empty clusters by reinitializing to a random point
        for i in range(k):
            if np.isnan(new_centroids[i]).any():
                # pick a random sample as centroid
                new_centroids[i] = X[np.random.randint(0, n_samples)]
        shift = np.linalg.norm(centroids - new_centroids)
        centroids = new_centroids
        if shift <= tol:
            break
    final_labels, _ = assign_clusters(X, centroids)
    return centroids, final_labels, inertia(X, final_labels, centroids)
