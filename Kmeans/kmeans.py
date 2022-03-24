import numpy as np
from sklearn.cluster import SpectralClustering


def kmeans(X:np.ndarray, k:int, centroids=None, max_iter=30, tolerance=1e-2):
    """
    :param X: training data
    :param k: the num of k nearest neighbor
    :param centroids: None for ramdon pick, 'kmeans++' for kmeans++ mechanism
    :param max_iter: max num of iteration
    :param tolerance: stop condition for early stop
    :return: centroids are the mk for center of group
    """
    if not centroids:
        centroids = kmeans_select_centroids(X, k)
    elif centroids == 'kmeans++':
        centroids = kmeans_plus_select_centroids(X, k)
    elif centroids == 'spectral':
        cluster = SpectralClustering(n_clusters=k, affinity="nearest_neighbors")
        labels = cluster.fit_predict(X)
        return labels

    labels = np.zeros((len(X),), dtype=int)
    for _ in range(max_iter):
        # assign points to clusters
        for i, x in enumerate(X):
            dis = [np.linalg.norm(x - c) for c in centroids]
            labels[i] = np.argmin(dis)

        # recompute k centroids
        new_centroids = []
        for i in range(k):
            new_centroids.append(np.sum(np.array(X[(labels == i), :]), axis=0) / len(X[(labels == i), :]))

        # stop condition
        if np.linalg.norm(np.array(new_centroids) - centroids) < tolerance:
            break
        centroids = new_centroids
    return np.array(centroids), labels


def kmeans_select_centroids(X, k):
    """
    kmeans initial points of centroid randomly
    """
    idx_cen = np.random.choice(len(X), k, replace=False)
    centroids = X[idx_cen, :]
    return centroids


def kmeans_plus_select_centroids(X, k):
    """
    kmeans++ algorithm to select initial points:
    Return centroids as k x p array of points from X.
    """
    # pick first point randomly
    centroids, dis = [], []
    idx_first = np.random.choice(len(X), 1, replace=False)
    centroids.append(X[int(idx_first), :])

    while len(centroids) < k:
        for x in X:
            min_dis = float('inf')
            for i in range(len(centroids)):
                distance = np.linalg.norm(x - centroids[i])
                if distance < min_dis:
                    min_dis = distance
            dis.append(min_dis)
        centroids.append(X[np.argmax(dis), :])
        dis = []
    return np.array(centroids)


def leaf_samples(rf, X:np.ndarray):
    """
    Return a list of arrays where each array is the set of X sample indexes
    residing in a single leaf of some tree in rf forest. For example, if there
    are 4 leaves (in one or multiple trees), we might return:

        array([array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
               array([10, 11, 12, 13, 14, 15]), array([16, 17, 18, 19, 20]),
               array([21, 22, 23, 24, 25, 26, 27, 28, 29]))
    """
    n_trees = len(rf.estimators_)
    leaf_samples = []
    # which leaf does each X_i go to for sole tree
    leaf_ids = rf.apply(X)
    for t in range(n_trees):
        # Group by id and return sample indexes
        uniq_ids = np.unique(leaf_ids[:, t])
        sample_idxs_in_leaves = [np.where(leaf_ids[:, t] == id)[0] for id in uniq_ids]
        leaf_samples.extend(sample_idxs_in_leaves)
    return leaf_samples
