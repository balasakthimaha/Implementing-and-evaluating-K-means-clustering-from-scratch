
# K-Means Clustering Report

This project implements the K-Means algorithm **from scratch** using NumPy, demonstrating understanding of initialization, assignment, and update steps.

## Dataset
A synthetic 2D dataset of ~300 points was generated, containing three visually separable clusters created using Gaussian distributions.

## Experiment
The custom K-Means implementation was applied using values of K ranging from **2 to 6**. For each K, the Within-Cluster Sum of Squares (WCSS / Inertia) was computed to evaluate clustering performance.

## Results
The inertia consistently decreased with increasing K, which is expected because more clusters reduce within-cluster variance. However, the rate of decrease slowed significantly around **K = 3**, forming a clear “elbow.”

## Conclusion
Based on the elbow method, **K = 3** is the optimal choice. It balances model complexity and clustering performance, accurately reflecting the three underlying structures in the dataset.
