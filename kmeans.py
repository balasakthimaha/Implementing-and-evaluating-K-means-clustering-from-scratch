"""
K-Means clustering implemented from scratch using NumPy only.
No scikit-learn clustering functions are used.

This module exposes a single convenience function `kmeans` plus a few
smaller helpers that are tested in `synthetic_kmeans_analysis.py`.
"""

from __future__ import annotations
import numpy as np


def initialize_centroids(X: np.ndarray, k: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Choose k unique points from X as the initial centroids.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_samples = X.shape[0]
    if k <= 0 or k > n_samples:
        raise ValueError("k must be in [1, n_samples]")
    indices = rng.choice(n_samples, size=k, replace=False)
    return X[indices]


def compute_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean distance from every point in X to every centroid.

    Returns
    -------
    distances : array, shape (n_samples, k)
        distances[i, j] is the distance between X[i] and centroids[j]
    """
    # Broadcasting: (n_samples, 1, n_features) - (1, k, n_features)
    diff = X[:, None, :] - centroids[None, :, :]
    return np.linalg.norm(diff, axis=2)


def assign_clusters(distances: np.ndarray) -> np.ndarray:
    """
    Assign each sample to the closest centroid based on the distance matrix.
    """
    return np.argmin(distances, axis=1)


def update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """
    Re-compute centroids as the mean of the assigned points.
    """
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features), dtype=float)
    for cluster_id in range(k):
        mask = labels == cluster_id
        if not np.any(mask):
            # If a cluster lost all its points, re‑initialize it at a random data point
            centroids[cluster_id] = X[np.random.randint(0, X.shape[0])]
        else:
            centroids[cluster_id] = X[mask].mean(axis=0)
    return centroids


def kmeans(
    X: np.ndarray,
    k: int,
    max_iters: int = 100,
    tol: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run the K-Means algorithm.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Data to cluster.
    k : int
        Number of clusters.
    max_iters : int, optional
        Maximum number of iterations.
    tol : float, optional
        Convergence threshold on the change in centroids.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    centroids : array, shape (k, n_features)
    labels : array, shape (n_samples,)
    inertia : float
        Sum of squared distances of samples to their closest centroid.
    """
    if rng is None:
        rng = np.random.default_rng()

    centroids = initialize_centroids(X, k, rng)
    for _ in range(max_iters):
        distances = compute_distances(X, centroids)
        labels = assign_clusters(distances)
        new_centroids = update_centroids(X, labels, k)
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    # Compute inertia: within-cluster sum of squares
    distances = compute_distances(X, centroids)
    closest_distances = distances[np.arange(X.shape[0]), labels]
    inertia = float((closest_distances ** 2).sum())
    return centroids, labels, inertia


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the mean Silhouette Score for the given clustering completely
    from scratch using NumPy.

    For each point i:
        a(i) = average distance to points in the same cluster
        b(i) = minimum over other clusters of the average distance
               to points in that cluster
        s(i) = (b(i) - a(i)) / max(a(i), b(i))

    The function returns the mean s(i) over all samples.
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    k = unique_labels.size
    if k < 2:
        raise ValueError("Silhouette score is undefined for a single cluster.")

    # Pre-compute full distance matrix (n_samples, n_samples)
    diff = X[:, None, :] - X[None, :, :]
    all_distances = np.linalg.norm(diff, axis=2)

    silhouettes = np.zeros(n_samples, dtype=float)
    for idx in range(n_samples):
        own_cluster = labels[idx]
        mask_same = labels == own_cluster
        mask_same[idx] = False  # exclude self

        if mask_same.any():
            a_i = all_distances[idx, mask_same].mean()
        else:
            # Degenerate case: point is the only member of its cluster
            a_i = 0.0

        b_i = np.inf
        for other_cluster in unique_labels:
            if other_cluster == own_cluster:
                continue
            mask_other = labels == other_cluster
            if mask_other.any():
                avg_dist = all_distances[idx, mask_other].mean()
                if avg_dist < b_i:
                    b_i = avg_dist

        denom = max(a_i, b_i)
        s_i = 0.0 if denom == 0 else (b_i - a_i) / denom
        silhouettes[idx] = s_i

    return float(silhouettes.mean())
