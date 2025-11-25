"""
Logistic Regression from Scratch
Files:
- main.py : run to reproduce dataset generation, training, evaluation, and outputs.
This project meets the requirements:
- Synthetic dataset: 5 features, 500 samples
- NumPy implementation of logistic regression components (sigmoid, loss, gradient descent)
- Hyperparameter tuning (learning rate, regularization)
- Performance metrics: accuracy, precision, recall, F1-score
- Report and README included in zip bundle

Run:
python main.py

Outputs:
- metrics.json
- model.npy
- results.txt
- report.txt
"""

import numpy as np
import json
from datetime import datetime
np.random.seed(42)

def generate_synthetic(n_samples=500, n_features=5, separation=2.0):
    # Create two Gaussian blobs with controllable separation for good separability
    n1 = n_samples // 2
    n2 = n_samples - n1
    mean1 = np.zeros(n_features) - separation/2.0
    mean2 = np.ones(n_features) * separation/2.0
    cov = np.eye(n_features) * 1.0
    X1 = np.random.multivariate_normal(mean1, cov, size=n1)
    X2 = np.random.multivariate_normal(mean2, cov, size=n2)
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n1), np.ones(n2)])
    # shuffle
    perm = np.random.permutation(n_samples)
    return X[perm], y[perm]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def compute_loss_and_grad(X, y, w, b, reg_lambda):
    n = X.shape[0]
    z = X.dot(w) + b
    p = sigmoid(z)
    # Avoid log(0)
    eps = 1e-12
    loss = - (y * np.log(p + eps) + (1-y) * np.log(1-p + eps)).mean()
    # L2 regularization
    loss += 0.5 * reg_lambda * np.sum(w*w)
    # gradients
    error = p - y
    grad_w = (X.T.dot(error) / n) + reg_lambda * w
    grad_b = error.mean()
    return loss, grad_w, grad_b

def train_logistic_regression(X, y, lr=0.1, reg_lambda=0.0, epochs=500, tol=1e-7, verbose=False):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    history = {"loss": []}
    for epoch in range(epochs):
        loss, grad_w, grad_b = compute_loss_and_grad(X, y, w, b, reg_lambda)
        w -= lr * grad_w
        b -= lr * grad_b
        history["loss"].append(loss)
        if epoch % 50 == 0 and verbose:
            print(f"Epoch {epoch}, loss={loss:.6f}")
        if epoch>0 and abs(history["loss"][-2] - history["loss"][-1]) < tol:
            break
    return w, b, history

def predict(X, w, b, threshold=0.5):
    probs = sigmoid(X.dot(w) + b)
    return (probs >= threshold).astype(int), probs

def metrics(y_true, y_pred):
    tp = np.sum((y_true==1)&(y_pred==1))
    tn = np.sum((y_true==0)&(y_pred==0))
    fp = np.sum((y_true==0)&(y_pred==1))
    fn = np.sum((y_true==1)&(y_pred==0))
    accuracy = (tp+tn) / len(y_true)
    precision = tp / (tp+fp) if (tp+fp)>0 else 0.0
    recall = tp / (tp+fn) if (tp+fn)>0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "tp":int(tp),"tn":int(tn),"fp":int(fp),"fn":int(fn)}

def train_test_split(X, y, test_size=0.2):
    n = X.shape[0]
    idx = np.random.permutation(n)
    cut = int(n*(1-test_size))
    return X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]

def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std==0]=1.0
    Xtr = (X_train - mean)/std
    Xte = (X_test - mean)/std
    return Xtr, Xte, mean, std

def hyperparameter_search(X_train, y_train, X_val, y_val, lrs=[0.5,0.1,0.05,0.01], regs=[0.0, 0.01, 0.1]):
    best = None
    results = []
    for lr in lrs:
        for reg in regs:
            w,b,_ = train_logistic_regression(X_train, y_train, lr=lr, reg_lambda=reg, epochs=2000, tol=1e-9)
            y_pred, _ = predict(X_val, w, b)
            m = metrics(y_val, y_pred)
            results.append({"lr":lr, "reg":reg, "metrics":m})
            if best is None or m["f1"] > best["metrics"]["f1"]:
                best = {"lr":lr, "reg":reg, "w":w, "b":b, "metrics":m}
    return best, results

def main():
    X, y = generate_synthetic(n_samples=500, n_features=5, separation=2.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # further split train into train/val
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2)
    X_trs, X_vals, mean, std = standardize(X_tr, X_val)
    X_tests = (X_test - mean)/std
    # hyperparameter search
    best, results = hyperparameter_search(X_trs, y_tr, X_vals, y_val, lrs=[0.5,0.1,0.05,0.01], regs=[0.0, 0.01, 0.1])
    # evaluate on test
    w_best = best["w"]
    b_best = best["b"]
    y_pred, probs = predict(X_tests, w_best, b_best)
    m_test = metrics(y_test, y_pred)
    # save model and metrics
    np.save("model.npy", np.concatenate([w_best, np.array([b_best])]))
    out = {"timestamp": datetime.now().isoformat(), "best_hyperparams": {"lr":best["lr"], "reg":best["reg"]}, "validation_best_metrics": best["metrics"], "test_metrics": m_test}
    with open("metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    # write a short results file
    with open("results.txt", "w") as f:
        f.write(json.dumps(out, indent=2))
    # print summary
    print("Best hyperparameters:", best["lr"], best["reg"])
    print("Test metrics:", m_test)
    # Save a report
    with open("report.txt", "w") as f:
        f.write("Logistic Regression from scratch - Report\\n")
        f.write("="*40 + "\\n")
        f.write("This project generated a synthetic binary classification dataset with 5 features and 500 samples.\\n")
        f.write("Separation parameter was chosen to ensure a reasonably learnable task, and the implementation below uses NumPy only.\\n\\n")
        f.write("Model details:\\n")
        f.write("- Sigmoid activation\\n- Binary cross-entropy loss with L2 regularization\\n- Batch Gradient Descent (full-batch)\\n\\n")
        f.write("Hyperparameter search explored learning rates and L2 regularization strengths.\\n\\n")
        f.write("Best hyperparameters (validation):\\n")
        f.write(json.dumps(best['lr'] if 'lr' in best else best, indent=2) + "\\n")
        f.write(\"\\nValidation metrics:\\n\") 
        f.write(json.dumps(best['metrics'], indent=2) + \"\\n\\n\")
        f.write(\"Test metrics:\\n\")
        f.write(json.dumps(m_test, indent=2) + \"\\n\\n\")
        f.write(\"Model coefficients (w and b):\\n\")
        f.write(np.array2string(w_best, precision=6) + \"\\n\")
        f.write(\"Bias (b):\\n\" + str(b_best) + \"\\n\\n\")
        f.write(\"Interpretation:\\n\")
        f.write(\"Positive coefficients increase log-odds of class=1; magnitude indicates relative influence.\\n\")
        f.write(\"Metrics used: accuracy, precision, recall, F1.\\n\")
    print('Report written to report.txt and metrics.json.')

if __name__ == '__main__':
    main()
