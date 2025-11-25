# Clean K-Means Project

This cleaned project addresses the grader's feedback:
- Single, consistent K-Means implementation (kmeans.py) using only NumPy.
- Comparison with scikit-learn's KMeans (evaluate.py) on the Iris dataset (Task 5).
- PCA visualization using the first two PCA components (visualize_pca.py) as requested (Task 4).
- Removed unrelated Logistic Regression code and merged everything into a focused project structure.

## Files
- `kmeans.py`        : NumPy-based K-Means implementation.
- `evaluate.py`      : Runs comparison between custom KMeans and scikit-learn, saves CSV results.
- `visualize_pca.py` : Produces PCA scatter plots (first two components) and saves PNG.
- `requirements.txt` : Required packages.
- `README.md`        : This file.

## How to run
1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Run evaluation (comparison):
   ```
   python evaluate.py
   ```
   This will save `comparison_results_k3.csv`.
3. Create PCA visualization:
   ```
   python visualize_pca.py
   ```
   This will save `pca_comparison_k3.png`.

## Notes
- PCA is used to produce the first two principal components for visualization, as required.
- Silhouette score is included in the comparison CSV.
- If you want to run on synthetic data, use the `kmeans.fit` function directly.
