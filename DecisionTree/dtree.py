import numpy as np
from sklearn.metrics import r2_score, accuracy_score


class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        if x_test[self.col] <= self.split:
            return self.lchild.predict(x_test)
        else:
            return self.rchild.predict(x_test)


class LeafNode:
    def __init__(self, y=None, prediction=None):
        """
        Create leaf node from y values and prediction
        """
        self.prediction = prediction

    def predict(self, x_test):
        return self.prediction


class DecisionTree621:
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss

    def fit(self, X, y):
        """
        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)

    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classifier or regressor.
        """
        if len(X) <= self.min_samples_leaf:
            return self.create_leaf(y)
        col, split = self.best_split_func(X, y, self.loss)
        if col == -1:
            return self.create_leaf(y)
        lchild = self.fit_(X[X[:,col] <= split], y[X[:,col] <= split])
        rchild = self.fit_(X[X[:,col] > split], y[X[:,col] > split])
        return DecisionNode(col, split, lchild, rchild)

    def best_split_func(self, X, y, loss):
        col, split = -1, -1
        loss_value = loss(y)
        best = (col, split, loss_value)
        k = 11
        for col_idx in range(len(X[0])):
            candidates = np.random.choice(X[:,col_idx], k)
            for can_split in candidates:
                yl = y[X[:,col_idx] <= can_split]
                yr = y[X[:,col_idx] > can_split]
                if len(yl) < self.min_samples_leaf or len(yr) < self.min_samples_leaf:
                    continue
                l = (len(yl) * loss(yl) + len(yr) * loss(yr)) / len(y)
                if l == 0:
                    return col_idx, can_split
                if l < best[2]:
                    best = (col_idx, can_split, l)
        return best[0], best[1]

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621
        """
        result = []
        for record in X_test:
                result.append(self.root.predict(record))
        return np.array(result)


class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.std)

    def score(self, X_test, y_test):
        """
        Return the R^2 of y_test
        """
        return r2_score(y_test, self.predict(X_test))

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))


class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)

    def score(self, X_test, y_test):
        """
        Return the accuracy_score() of y_test vs predictions for each record in X_test
        """
        return accuracy_score(y_test, self.predict(X_test))

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor.
        """
        label, count = np.unique(y, return_counts=True)
        idx = np.argmax(count)
        return LeafNode(y, label[idx])


def gini(y):
    "Return the gini impurity score for values in y"
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(np.square(p))


# python -m pytest -v test_dtree.py
