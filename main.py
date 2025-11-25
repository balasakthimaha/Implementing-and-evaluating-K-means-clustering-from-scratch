import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from kmeans import KMeans

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.2, random_state=42)

wcss_results = []
sil_results = []
ascii_map = []

best_k = 0
best_sil = -1
best_labels = None

for k in range(2, 8):
    model = KMeans(k)
    model.fit(X)

    labels = np.zeros(len(X), dtype=int)
    for c_id, pts in enumerate(model.clusters):
        labels[pts] = c_id

    wcss = model.wcss(X)
    sil = silhouette_score(X, labels)

    wcss_results.append(f"K={k}, WCSS={wcss}")
    sil_results.append(f"K={k}, Silhouette={sil}")

    if sil > best_sil:
        best_sil = sil
        best_k = k
        best_labels = labels

# ASCII visualization
for i in range(len(X)):
    ascii_map.append(f"Point {i}: Cluster {best_labels[i]}")

# Save outputs
with open("wcss_results.txt","w") as f: f.write("\n".join(wcss_results))
with open("silhouette_results.txt","w") as f: f.write("\n".join(sil_results))
with open("ascii_clusters.txt","w") as f: f.write("\n".join(ascii_map))
with open("cluster_assignments.txt","w") as f: f.write(",".join(map(str,best_labels)))

print("Optimal K =", best_k)
