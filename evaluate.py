"""evaluate.py
- Loads Iris dataset
- Runs custom KMeans (from kmeans.py)
- Runs scikit-learn KMeans
- Computes inertia and silhouette_score for both (silhouette requires >1 cluster)
- Saves comparison results to a CSV
"""
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans as SKKMeans

from kmeans import fit as custom_kmeans_fit

def load_iris():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    return X, y, feature_names

def run_comparison(k=3, random_state=42):
    X, y, feat = load_iris()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # custom kmeans
    centroids_c, labels_c, inertia_c, _ = custom_kmeans_fit(Xs, k, random_state=random_state)

    # sklearn kmeans
    sk = SKKMeans(n_clusters=k, random_state=random_state, n_init=10)
    sk.fit(Xs)
    labels_sk = sk.labels_
    inertia_sk = sk.inertia_

    # silhouette (only if k>1)
    sil_c = silhouette_score(Xs, labels_c) if k > 1 else float('nan')
    sil_sk = silhouette_score(Xs, labels_sk) if k > 1 else float('nan')

    results = {
        'method': ['custom'] + ['sklearn'],
        'inertia': [inertia_c, inertia_sk],
        'silhouette': [sil_c, sil_sk]
    }
    df = pd.DataFrame(results)
    df.to_csv('comparison_results_k{}.csv'.format(k), index=False)
    print(df)
    return df

if __name__ == '__main__':
    run_comparison(k=3)
