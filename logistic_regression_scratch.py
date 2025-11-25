import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.1, iterations=5000):
        self.lr = lr
        self.iterations = iterations

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.theta=np.zeros(X.shape[1])
        for _ in range(self.iterations):
            h=self.sigmoid(X.dot(self.theta))
            grad=X.T.dot(h-y)/y.size
            self.theta-=self.lr*grad

    def predict_proba(self, X):
        X=np.c_[np.ones(X.shape[0]), X]
        return self.sigmoid(X.dot(self.theta))

    def predict(self, X):
        return (self.predict_proba(X)>=0.5).astype(int)

    def odds_ratios(self):
        return np.exp(self.theta)
