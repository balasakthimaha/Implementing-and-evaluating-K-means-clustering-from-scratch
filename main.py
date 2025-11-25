import numpy as np
from sklearn.linear_model import LogisticRegression
from logistic_regression import LogisticRegressionScratch

# Example synthetic data
np.random.seed(0)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Scratch implementation
model_scratch = LogisticRegressionScratch(lr=0.1, iterations=2000)
model_scratch.fit(X, y)
preds_scratch = model_scratch.predict(X)

# Sklearn implementation
model_sklearn = LogisticRegression()
model_sklearn.fit(X, y)
preds_sklearn = model_sklearn.predict(X)

print("Scratch accuracy:", (preds_scratch == y).mean())
print("Sklearn accuracy:", (preds_sklearn == y).mean())
