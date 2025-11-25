import numpy as np

def silhouette_score(X, labels):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        return 0.0
    # Precompute distance matrix
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    sil_samples = np.zeros(n)
    for i in range(n):
        own = labels[i]
        mask_own = labels == own
        a = 0.0
        if np.sum(mask_own) > 1:
            a = np.sum(dists[i, mask_own]) / (np.sum(mask_own) - 1)
        else:
            a = 0.0
        b = np.inf
        for other in unique_labels:
            if other == own:
                continue
            mask_other = labels == other
            if np.sum(mask_other) == 0:
                continue
            dist_other = np.mean(dists[i, mask_other])
            if dist_other < b:
                b = dist_other
        sil = 0.0
        denom = max(a, b)
        if denom > 0:
            sil = (b - a) / denom
        sil_samples[i] = sil
    return np.mean(sil_samples)
