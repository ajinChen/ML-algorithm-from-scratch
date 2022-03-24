import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def load_dataset(path="data/rent-ideal.csv"):
    dataset = np.loadtxt(path, delimiter=",", skiprows=1)
    y = dataset[:, -1]
    X = dataset[:, 0:-1]
    return X, y


def gradient_boosting_mse(X, y, num_iter, max_depth=1, nu=0.1):
    """
    Given X, a array y and num_iter return y_mean and trees
    Input: X, y, num_iter
           max_depth
           nu (is the shinkage)
    Outputs:y_mean, array of trees from DecisionTreeRegression
    """
    trees = []
    N, _ = X.shape
    y_mean = np.mean(y)
    fm = y_mean * np.ones(N)
    for i in range(num_iter):
        rm = y - fm
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
        tree.fit(X, rm)
        fm += nu * tree.predict(X)
        trees.append(tree)
    return y_mean, trees  


def gradient_boosting_predict(X, trees, y_mean,  nu=0.1):
    """
    Given X, trees, y_mean predict y_hat
    """
    N, _ = X.shape
    fm = y_mean * np.ones(N)
    for tree in trees:
        fm += nu * tree.predict(X)
    y_hat = fm
    return y_hat

