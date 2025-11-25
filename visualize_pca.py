"""visualize_pca.py
- PCA projection of Iris to first two components
- Scatter plots of true labels, custom kmeans labels, sklearn labels
- Saves plots to PNG
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from kmeans import fit as custom_kmeans_fit
from sklearn.cluster import KMeans as SKKMeans

def plot_pca_comparisons(k=3, random_state=42):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)

    cent_c, labels_c, _, _ = custom_kmeans_fit(Xs, k, random_state=random_state)
    sk = SKKMeans(n_clusters=k, random_state=random_state, n_init=10)
    sk.fit(Xs)
    labels_sk = sk.labels_

    fig, axes = plt.subplots(1,3, figsize=(15,4))
    axes[0].scatter(Xp[:,0], Xp[:,1], c=y, cmap='viridis', s=30)
    axes[0].set_title('True labels (Iris)')
    axes[1].scatter(Xp[:,0], Xp[:,1], c=labels_c, cmap='viridis', s=30)
    axes[1].set_title('Custom KMeans labels')
    axes[2].scatter(Xp[:,0], Xp[:,1], c=labels_sk, cmap='viridis', s=30)
    axes[2].set_title('scikit-learn KMeans labels')
    for ax in axes:
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    plt.tight_layout()
    plt.savefig('pca_comparison_k{}.png'.format(k), dpi=150)
    print('Saved pca_comparison_k{}.png'.format(k))

if __name__ == '__main__':
    plot_pca_comparisons(k=3)
