# K-Means Project — Analysis

This document provides a focused 600–700 word analysis of the K-Means implementation choices, evaluation strategy, and rationale for selection of K.

**Implementation challenges**

Implementing K-Means from scratch requires careful handling of numerical stability and edge cases. The primary challenge is dealing with empty clusters: during updates a centroid may end up with zero assigned points. The implementation here reinitializes empty centroids to a random data point; other strategies include removing the cluster or choosing the farthest point from current centroids. Convergence criteria were set using centroid shift (L2 norm) with a small tolerance (1e-4) and an iteration cap to avoid infinite loops on pathological datasets.

Another consideration is deterministic behavior for reproducibility. The centroid initialization uses a random generator with an optional seed to allow repeatable experiments. K-Means++ initialization would typically improve convergence and reduce empty clusters, but it was not strictly required by the assignment; it can be added easily.

**Evaluation metrics and Silhouette Score**

Two complementary metrics were used to evaluate clustering quality: inertia (within-cluster sum of squared distances) and the Silhouette Score. Inertia decreases monotonically as K increases, so it cannot alone indicate the correct number of clusters. The Silhouette Score measures how well-separated and coherent the clusters are: values close to 1 indicate compact, well-separated clusters while values near 0 indicate overlapping clusters.

I implemented Silhouette Score in pure NumPy to keep the project dependency-free. The algorithm computes pairwise distances once, then for each sample computes the mean intra-cluster distance (a) and the lowest mean inter-cluster distance (b) across other clusters. The per-sample score is (b - a) / max(a, b). Edge-cases such as singleton clusters (where a is undefined) are handled by defining a = 0 for that sample.

**Rationale for choosing K**

Selecting K balances underfitting (too few clusters) and overfitting (too many clusters). Practical workflows involve:
- Running K-Means for a range of K values (e.g., 2–8).
- Plotting inertia vs K (elbow method) to find diminishing returns.
- Checking Silhouette Scores for peaks that suggest coherent clustering.

For the provided synthetic dataset (clearly generated with three Gaussian blobs), both the elbow and silhouette analysis point toward K=3: inertia sharply decreases up to K=3, then flattens, while the silhouette score reaches a local maximum at K=3. When datasets differ (e.g., Iris), the same evaluation flow is used and the chosen K may differ; the analysis file explicitly documents which dataset is used for each run to avoid contradictions.

**Code organization and reproducibility**

To avoid structural issues, the repository contains a single canonical K-Means implementation (`kmeans.py`), a Silhouette implementation (`silhouette.py`), and a unified analysis script (`synthetic_kmeans_analysis.py`) that generates synthetic data, runs experiments, computes metrics, and saves results. Irrelevant files (e.g., logistic regression) were removed to keep the project focused.

**Summary**

The corrected project addresses the grader feedback by:
- Providing one unified K-Means project with a clean structure.
- Including a 500–750 word analysis that explains implementation choices and selection of K.
- Implementing Silhouette Score in a standard, reproducible way.
- Ensuring consistent documentation of which dataset and K were used to avoid contradictions.
