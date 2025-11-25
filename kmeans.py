
import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(X, k):
    idx = np.random.choice(len(X), k, replace=False)
    return X[idx]

def assign_clusters(X, centroids):
    dists = np.linalg.norm(X[:,None] - centroids[None,:], axis=2)
    return np.argmin(dists, axis=1)

def update_centroids(X, labels, k):
    return np.array([X[labels==i].mean(axis=0) for i in range(k)])

def kmeans(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    inertia = np.sum((np.linalg.norm(X - centroids[labels], axis=1)**2))
    return centroids, labels, inertia

if __name__ == "__main__":
    np.random.seed(42)
    c1 = np.random.randn(100,2) + [0,0]
    c2 = np.random.randn(100,2) + [5,5]
    c3 = np.random.randn(100,2) + [0,5]
    X = np.vstack([c1,c2,c3])

    inertias = {}
    for k in range(2,7):
        _,_, inertia = kmeans(X,k)
        inertias[k] = inertia
        print(f"K={k}, Inertia={inertia}")

    print("\nElbow Method:")
    for k,v in inertias.items():
        print(f"{k}: {'*'*int(v/100)}")
