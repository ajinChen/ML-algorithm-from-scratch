from dtree import *
from sklearn.utils import resample
from collections import defaultdict


class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.
        """
        rf_list = []
        for i in range(self.n_estimators):
            X_boot, y_boot, oob_idxs = bootstrap(X, y, len(X))
            T = self.tree(oob_idxs)
            rf_list.append(T.fit(X_boot, y_boot))
        self.trees = rf_list

        if self.oob_score:
            self.oob_score_ = self.compute_oob_score(X, y)


class RandomForestRegressor621(RandomForest621):
    def __init__(
            self,
            n_estimators=10,
            min_samples_leaf=3,
            max_features=0.3,
            oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest.
        """
        w_avg = []
        for x_test in X_test:
            leaves = []
            for tree in self.trees:
                leaves.append(tree.leaf(x_test))
            nops = 0
            ysum = 0
            for leaf in leaves:
                nops += leaf.n_leaf()
                ysum += np.sum(leaf.y_value())
            w_avg.append(ysum / nops)
        return np.array(w_avg)

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        return r2_score(y_test, self.predict(X_test))

    def tree(self, oob_idxs):
        return RegressionTree621(
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            oob_idxs=oob_idxs)

    def compute_oob_score(self, X, y):
        n = len(X)
        oob_counts = np.zeros((n,))
        oob_preds = np.zeros((n,))
        for tree in self.trees:
            for idx in tree.oob_idxs:
                leafsizes = len(tree.leaf(X[idx]).y_value())
                oob_preds[idx] += leafsizes * tree.predict(X[idx])
                oob_counts[idx] += leafsizes
        oob_avg_preds = oob_preds[oob_preds != 0] / oob_counts[oob_counts != 0]
        return r2_score(y[oob_counts != 0], oob_avg_preds)


class RandomForestClassifier621(RandomForest621):
    def __init__(
            self,
            n_estimators=10,
            min_samples_leaf=3,
            max_features=0.3,
            oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.trees = None

    def predict(self, X_test) -> np.ndarray:
        output = []
        for x_test in X_test:
            count = defaultdict(int)
            for tree in self.trees:
                leaf = tree.leaf(x_test)
                for y_class in leaf.y_value():
                    count[y_class] += 1
            idx = np.argmax(list(count.values()))
            output.append(list(count.keys())[idx])
        return np.array(output)

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        return accuracy_score(y_test, self.predict(X_test))

    def tree(self, oob_idxs):
        return ClassifierTree621(
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            oob_idxs=oob_idxs)

    def compute_oob_score(self, X, y):
        n = len(X)
        labels, _, k = nunique(y)
        oob_counts = np.zeros((n,))
        oob_preds = np.zeros((n, k))
        for tree in self.trees:
            for idx in tree.oob_idxs:
                leafsizes = len(tree.leaf(X[idx]).y_value())
                tpred = np.where(labels == tree.predict(X[idx]))
                oob_preds[idx, tpred] += leafsizes
                oob_counts[idx] += 1
        oob_votes = labels[np.argmax(oob_preds[oob_counts != 0], axis=1)]
        return accuracy_score(y[oob_counts != 0], oob_votes)


# function region
def bootstrap(X, y, size):
    idx = [i for i in range(size)]
    n = size
    X_boot, idx_sample, y_boot = resample(
        X, idx, y, replace=True, n_samples=int(n))
    idx_oob = list(set(idx) - set(idx_sample))
    return X_boot, y_boot, idx_oob


def nunique(y):
    labels, counts = np.unique(y, return_counts=True)
    return np.array(labels), counts, len(labels)

# pytest -v -n 8 test_rf.py
