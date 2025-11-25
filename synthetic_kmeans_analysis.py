import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans
from silhouette import silhouette_score
import os

def make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42):
    rng = np.random.default_rng(random_state)
    centers_coords = rng.normal(loc=0.0, scale=5.0, size=(centers, 2))
    X = []
    y = []
    per_cluster = n_samples // centers
    for i, c in enumerate(centers_coords):
        pts = rng.normal(loc=c, scale=cluster_std, size=(per_cluster, 2))
        X.append(pts)
        y += [i] * per_cluster
    X = np.vstack(X)
    return X, np.array(y)

def run_experiments():
    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.8)
    results = []
    ks = range(2,7)
    for k in ks:
        centroids, labels, inert = kmeans(X, k, random_state=42)
        sil = silhouette_score(X, labels)
        results.append((k, inert, sil))
        print(f"K={k}: inertia={inert:.2f}, silhouette={sil:.4f}")
        # save a simple plot
        fig, ax = plt.subplots()
        ax.scatter(X[:,0], X[:,1], c=labels, s=10)
        ax.scatter(centroids[:,0], centroids[:,1], marker='X', s=100, edgecolors='k')
        ax.set_title(f'K={k}  Silhouette={sil:.3f}')
        os.makedirs('figures', exist_ok=True)
        fig.savefig(f'figures/k_{k}.png')
        plt.close(fig)
    # save results
    with open('results.txt', 'w') as f:
        for k, inert, sil in results:
            f.write(f"K={k}, inertia={inert:.4f}, silhouette={sil:.6f}\n")
    print('Saved results.txt and figures/')

if __name__ == '__main__':
    run_experiments()
