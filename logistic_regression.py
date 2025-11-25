import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.1, iterations=1000):
        self.lr = lr
        self.iterations = iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        for _ in range(self.iterations):
            z = X.dot(self.theta)
            h = self.sigmoid(z)
            gradient = X.T.dot(h - y) / y.size
            self.theta -= self.lr * gradient

    def predict(self, X):
        return (self.sigmoid(X.dot(self.theta)) >= 0.5).astype(int)
