import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


def accuracy(y, pred):
    return np.sum(y == pred) / float(y.shape[0]) 


def parse_spambase_data(filename):
    """
    Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)
    """
    dataset = np.loadtxt(filename, delimiter=",")
    K = len(dataset[0])
    Y = dataset[:, K - 1]
    X = dataset[:, 0 : K - 1]
    Y = np.array([-1. if y == 0. else 1. for y in Y])
    return X, Y


def normalize(X, X_val):
    """
    Given X, X_val compute X_scaled, X_val_scaled
    return X_scaled, X_val_scaled
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_val_scaled = scaler.transform(X_val)
    return X_scaled, X_val_scaled


class NN(nn.Module):
    def __init__(self, D, seed, hidden=10):
        super(NN, self).__init__()
        torch.manual_seed(seed)
        self.linear1 = nn.Linear(D, hidden)
        self.linear2 = nn.Linear(hidden, 1)
        self.bn1 = nn.BatchNorm1d(num_features=hidden)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(F.relu(x))
        return self.linear2(x)


def fitNN(model, X, r, epochs=20, lr=0.1):
    """
    Fit a regression model to the pseudo-residuals
    returns the fitted values on training data as a numpy array
    Shape of the resturn should be (N,) not (N,1).
    """
    X = torch.FloatTensor(X)
    r = torch.FloatTensor(r).unsqueeze(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for i in range(epochs):
        model.train()
        out = model(X)
        loss = torch.log(1 + torch.exp(-out * r)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return out.squeeze(1).detach().numpy()


def gradient_boosting_predict(X, f0, models, nu):
    """
    Given X, models, f0 and nu predict y_hat in {-1, 1}
    y_hat should be a numpy array with shape (N,)
    """
    X = torch.FloatTensor(X)
    for model in models:
        model.eval()
        f0 += nu * model(X).squeeze(1).detach().numpy()
    y_hat = np.sign(f0)
    return y_hat


def compute_pseudo_residual(y, fm):
    """
    vectorized computation of the pseudoresidual
    """
    res = y / (1 + np.exp(y * fm))
    return res


def boostingNN(X, Y, num_iter, nu):
    """
    Given an numpy matrix X, a array y and num_iter return trees and weights 
    Input: X, y, num_iter
    Outputs: array of Regression models
    Assumes y is {-1, 1}
    """
    models = []
    N, D = X.shape
    seeds = [s+1 for s in range(num_iter)]
    f0 = np.log(np.sum(Y == 1) / np.sum(Y == -1))
    res = Y - f0
    for seed in seeds:
        model = NN(D, seed)
        y_hat = fitNN(model, X, res) * nu
        res = compute_pseudo_residual(Y, y_hat)
        models.append(model)
    return f0, models


