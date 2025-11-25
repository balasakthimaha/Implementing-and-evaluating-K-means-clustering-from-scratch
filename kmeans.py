"""kmeans.py
Clean, single K-Means implementation using NumPy only.
Functions:
  - init_centroids_random
  - assign_labels
  - compute_centroids
  - fit (returns centroids, labels, inertia, history)
"""
import numpy as np

def init_centroids_random(X, k, random_state=None):
    rng = np.random.RandomState(random_state)
    indices = rng.choice(len(X), k, replace=False)
    return X[indices].astype(float)

def assign_labels(X, centroids):
    # distances shape (n_samples, k)
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1)

def compute_centroids(X, labels, k):
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features), dtype=float)
    for i in range(k):
        members = X[labels == i]
        if len(members) == 0:
            # empty cluster -> leave centroid as zeros (caller should handle)
            centroids[i] = np.nan
        else:
            centroids[i] = members.mean(axis=0)
    return centroids

def inertia_score(X, centroids, labels):
    return float(np.sum((X - centroids[labels])**2))

def fit(X, k, max_iters=300, tol=1e-4, random_state=None, verbose=False):
    X = np.asarray(X, dtype=float)
    centroids = init_centroids_random(X, k, random_state=random_state)
    history = {'centroids': [], 'inertia': []}
    for it in range(max_iters):
        labels = assign_labels(X, centroids)
        new_centroids = compute_centroids(X, labels, k)
        # handle empty clusters by reinitializing to random points
        for i in range(k):
            if np.any(np.isnan(new_centroids[i])):
                # reinitialize to a random existing point
                new_centroids[i] = X[np.random.randint(0, len(X))]
        shift = np.linalg.norm(centroids - new_centroids)
        centroids = new_centroids
        inertia = inertia_score(X, centroids, labels)
        history['centroids'].append(centroids.copy())
        history['inertia'].append(inertia)
        if verbose:
            print(f'Iter {it}: inertia={inertia:.4f}, shift={shift:.6f}')
        if shift <= tol:
            break
    return centroids, labels, inertia, history

if __name__ == '__main__':
    # quick local test (synthetic)
    import numpy as np
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(100,2)+np.array([0,0]),
                   rng.randn(100,2)+np.array([5,5]),
                   rng.randn(100,2)+np.array([0,5])])
    centroids, labels, inertia, _ = fit(X, 3, random_state=42, verbose=True)
    print('Final inertia:', inertia)
