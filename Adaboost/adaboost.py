import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
import pandas as pd


def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))


def parse_spambase_data(filename):
    """
    Given a filename return X and Y numpy arrays
    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)
    """
    data = pd.read_csv(filename, header=None)
    X = data.iloc[:, :-1].to_numpy()
    Y = data.iloc[:, -1].apply(lambda x: -1. if x == 0 else x).to_numpy()
    return X, Y


def adaboost(X, y, num_iter, max_depth=1):
    """
    Given an numpy matrix X, a array y and num_iter return trees and weights
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is {-1, 1}
    """
    trees = []
    trees_weights = []
    N, _ = X.shape
    w = np.ones(N) / N
    delta = 0.0001
    for i in range(num_iter):
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        tree.fit(X, y, sample_weight=w)
        y_hat = tree.predict(X)
        mask = [int(x) if int(x) == 1 else 0 for x in (y_hat != y)]
        error = np.dot(w, (y_hat != y)) / np.sum(w)
        alpha = np.log((1-error) / (error+delta))
        w = np.multiply(w, np.exp([x * alpha for x in mask]))
        trees.append(tree)
        trees_weights.append(alpha)
    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """
    Given X, trees and weights predict Y
    """
    y = np.sign(np.sum([alpha * tree.predict(X) for alpha, tree in zip(trees_weights, trees)], axis=0))
    return y
