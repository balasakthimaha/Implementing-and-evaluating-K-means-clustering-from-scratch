import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from kmeans import KMeans

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.2, random_state=42)

wcss_results = []
silhouette_results = []

for k in range(2, 8):
    model = KMeans(k)
    model.fit(X)

    wcss = model.wcss(X)
    wcss_results.append((k, wcss))

    labels = np.zeros(len(X), dtype=int)
    for cluster_id, points in enumerate(model.clusters):
        labels[points] = cluster_id

    sil_score = silhouette_score(X, labels)
    silhouette_results.append((k, sil_score))

with open("elbow_results.txt", "w") as f:
    for k, w in wcss_results:
        f.write(f"K={k}, WCSS={w}\n")

with open("silhouette_results.txt", "w") as f:
    for k, s in silhouette_results:
        f.write(f"K={k}, Silhouette={s}\n")

print("DONE")
