import numpy as np
from sklearn.linear_model import LogisticRegression
from logistic_regression_scratch import LogisticRegressionScratch
from metrics import classification_metrics

np.random.seed(0)
X=np.random.randn(200,3)
y=(X[:,0]*0.8 + X[:,1]*-0.3 + 0.5 >0).astype(int)

model=LogisticRegressionScratch()
model.fit(X,y)
pred=model.predict(X)

print("Scratch metrics:", classification_metrics(y,pred))
print("Odds ratios:", model.odds_ratios())

model2=LogisticRegression().fit(X,y)
print("Sklearn score:", model2.score(X,y))
