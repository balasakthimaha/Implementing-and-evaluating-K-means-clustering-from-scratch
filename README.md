# Implementing and Evaluating K-Means Clustering from Scratch

This mini‑project contains a complete NumPy‑only implementation of the K‑Means
clustering algorithm plus a small synthetic experiment that evaluates the effect
of different choices of **K** using both inertia and the Silhouette Score.

## Files

- `kmeans.py` — core K‑Means implementation and a pure‑NumPy Silhouette score.
- `synthetic_kmeans_analysis.py` — generates a 2D synthetic dataset, runs
  K‑Means for several values of K, and prints inertia and Silhouette results.
- `analysis.txt` — 500–750 word written analysis that interprets the numerical
  results of the synthetic experiment and explains why K=3 is chosen as optimal
  for the generated dataset.

## Running the experiment

```bash
python synthetic_kmeans_analysis.py
```

You should see a small table with inertia and Silhouette Score for each value
of K in the range 2–6, followed by the chosen K based on the Silhouette Score
criterion. The discussion of these numbers is in `analysis.txt`.

No logistic regression models or external datasets (such as Iris) are used;
everything is focused purely on K‑Means with synthetic data, as required.
