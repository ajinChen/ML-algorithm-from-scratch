import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def normalize(X):
    """
    Creating standard variables here (u-x)/sigma
    :param X:
    :return:
    """
    if isinstance(X, pd.DataFrame):
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                u = np.mean(X[c])
                s = np.std(X[c])
                X[c] = (X[c] - u) / s
        return
    for j in range(X.shape[1]):
        u = np.mean(X[:, j])
        s = np.std(X[:, j])
        X[:, j] = (X[:, j] - u) / s


def loss_gradient(X, y, B, lmbda):
    grad = -1 * np.dot(np.transpose(X), (y - np.dot(X, B)))
    return grad


def loss_ridge(X, y, B, lmbda):
    func_r = np.dot((y - np.dot(X, B)), (y - np.dot(X, B))) + \
        lmbda * np.dot(B, B)
    return func_r


def loss_gradient_ridge(X, y, B, lmbda):
    grad = -1 * np.dot(np.transpose(X), (y - np.dot(X, B))) + lmbda * B
    return grad


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_likelihood(X, y, B, lmbda):
    z = np.dot(X, B)
    func_l = -1 * np.sum(np.dot(y, B) - np.log(1 + np.exp(z)))
    return func_l


def log_likelihood_gradient(X, y, B, lmbda):
    z = np.dot(X, B)
    grad = -1 * np.dot(np.transpose(X), (y - sigmoid(z)))
    return grad


def L1_log_likelihood(X, y, B, lmbda):
    z = np.dot(X, B)
    func_l = -1 * np.sum(np.dot(y, B) - np.log(1 + np.exp(z))) + lmbda * np.sum(np.abs(B))
    return func_l


def L1_log_likelihood_gradient(X, y, B, lmbda):
    n, p = X.shape
    z = np.dot(X, B)
    err = y - sigmoid(z)
    r = lmbda * np.sign(B)
    r[0] = 0
    grad = (np.dot(np.transpose(X), err) - r) / n
    return B - grad


def minimize(X, y, loss_gradient,
             eta=0.00001, lmbda=0.0,
             max_iter=1000, addB0=True,
             precision=1e-9):
    if X.ndim != 2:
        raise ValueError("X must be n x p for p features")
    n, p = X.shape
    if y.shape != (n, 1):
        raise ValueError(f"y must be n={n} x 1 not {y.shape}")

    if addB0:
        X = np.hstack([np.ones(shape=(n, 1)), X])
        n, p = X.shape

    B = np.random.random_sample(size=(p, 1)) * 2 - 1  # make between [-1,1)
    H = 0
    eps = 1e-5  # prevent division by 0
    while max_iter > 0:
        max_iter -= 1
        grad = loss_gradient(X, y, B, lmbda)
        if np.linalg.norm(grad) < precision:
            return B
        H += np.multiply(grad, grad)
        B = B - eta * grad / (np.sqrt(H) + eps)
    return B


class LinearRegression621:
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                          loss_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)


class LogisticRegression621:
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict_proba(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        z = np.dot(X, self.B)
        return sigmoid(z)

    def predict(self, X):
        prob = self.predict_proba(X)
        pred = []
        for i in prob.flatten():
            if i > 0.5:
                pred.append(1)
            else:
                pred.append(0)
        pred = np.array(pred).reshape(-1, 1)
        return pred

    def fit(self, X, y):
        self.B = minimize(X, y,
                          log_likelihood_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)


class RidgeRegression621:
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000, y_mean=0):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.y_mean = y_mean

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                          loss_gradient_ridge,
                          self.eta,
                          self.lmbda,
                          self.max_iter,
                          addB0=False)
        self.y_mean = np.array(np.mean(y))
        self.B = np.vstack([self.y_mean, self.B]).reshape(-1, 1)


class LassoLogistic621:
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict_proba(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        z = np.dot(X, self.B)
        return sigmoid(z)

    def predict(self, X):
        prob = self.predict_proba(X)
        pred = []
        for i in prob.flatten():
            if i > 0.5:
                pred.append(1)
            else:
                pred.append(0)
        pred = np.array(pred).reshape(-1, 1)
        return pred

    def fit(self, X, y):
        self.B = minimize(X, y,
                          L1_log_likelihood_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter,
                          addB0=False)
        z = np.dot(X, self.B)
        err = y - sigmoid(z)
        beta_b0 = np.array(np.mean(err))
        self.B = np.vstack([beta_b0, self.B]).reshape(-1, 1)

# python -m pytest -v --count=20 test_regr.py
# python -m pytest -v --count=20 test_class.py
