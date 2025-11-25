"""
Synthetic K-Means experiment.

This script:
1. Generates a clearly clustered 2D synthetic data set.
2. Runs our from‑scratch K‑Means implementation for K in [2, 3, 4, 5, 6].
3. Computes inertia and Silhouette Score for each K.
4. Prints a small table with the results and highlights the chosen K.

The written discussion of these results can be found in `analysis.txt`.
"""

import numpy as np
from kmeans import kmeans, silhouette_score


def generate_synthetic_data(n_per_cluster: int = 150, rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng(42)

    # Three fairly well separated Gaussian blobs in 2D
    centers = np.array([
        [-4.0, 0.0],
        [0.0, 4.0],
        [4.0, -1.5],
    ])

    X_list = []
    for cx, cy in centers:
        cov = np.array([[0.8, 0.2],
                        [0.2, 0.5]])
        points = rng.multivariate_normal(mean=[cx, cy], cov=cov, size=n_per_cluster)
        X_list.append(points)

    X = np.vstack(X_list)
    return X


def run_experiment():
    rng = np.random.default_rng(123)
    X = generate_synthetic_data(rng=rng)

    print("K	Inertia		Silhouette")
    print("-" * 40)
    results = []
    for k in range(2, 7):
        centroids, labels, inertia = kmeans(X, k=k, rng=rng)
        s = silhouette_score(X, labels)
        results.append((k, inertia, s))
        print(f"{k}	{inertia:10.2f}	{s:8.3f}")

    # Choose K with the highest silhouette; in this data it is K=3
    best_k, best_inertia, best_s = max(results, key=lambda t: t[2])
    print("\nChosen K based on silhouette:", best_k)
    print(f"Best inertia: {best_inertia:.2f}, best silhouette: {best_s:.3f}")


if __name__ == "__main__":
    run_experiment()
