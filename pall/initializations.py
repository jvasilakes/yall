import numpy as np
from sklearn.neighbors import NearestNeighbors


class LDS(object):
    """
    Initialization using the Local Density Score (LDS) as described in
    V. Vu, N. Labroche and B. Bouchon-Meunier, "Active Learning for
    Semi-Supervised K-Means Clustering," 2010 22nd IEEE International
    Conference on Tools with Artificial Intelligence, Arras, 2010, pp. 12-15.
    """
    def __init__(self, k=30, threshold="auto"):
        """
        LDS(x_i) = \frac{sum_{x_j \in NN(x_i)} weight(x_i, x_j)} / k
            where weight(x_i, x_j) = |NN(x_i) \cap NN(x_j)|

        :param int k: Number of k nearest neighbors to use for LDS.
        :param float threshold: The minimum LDS score required for
                                a point to be chosen. The authors
                                of the original paper recommend the
                                interval [k/2 - 2, k/2 + 2].
        """
        self.k = k
        if threshold == "auto":
            self.threshold = (self.k / 2) + 1
        else:
            self.threshold = float(threshold)
        self.scores = []

    def find_centers(self, X, y):
        neighbors = self._KNN(X)
        self.scores = self._compute_scores(neighbors, X)
        idxs = self._init_points(self.scores)
        return idxs

    def _KNN(self, X):
        NN = NearestNeighbors(n_neighbors=self.k, algorithm="brute",
                              metric="euclidean")
        NN.fit(X)
        idxs = NN.kneighbors(return_distance=False)
        neighbors = dict(zip(range(idxs.shape[0]), idxs))
        return neighbors

    def _compute_scores(self, neighbors, X):
        def weight(x_i, x_j):
            return len(set(neighbors[x_i]).intersection(neighbors[x_j]))
        scores = np.zeros(X.shape[0], dtype=float)
        for x_i in range(X.shape[0]):
            nns = neighbors[x_i]
            score = np.sum([weight(x_i, x_j) for x_j in nns]) / len(nns)
            scores[x_i] = score
        return scores

    def _init_points(self, scores):
        keep_idxs = [i for (i, score) in enumerate(scores)
                     if score >= self.threshold]
        return keep_idxs


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.manifold import TSNE

    iris = load_iris()
    X = TSNE(n_components=2).fit_transform(iris.data)
    y = iris.target

    lds = LDS(k=10, threshold=7)
    idxs = lds(X, y)
    print(f"N: {len(idxs)}")
    print(f"T: {lds.threshold}")
    print(f"Scores: {lds.scores.max()} {lds.scores.min()}")
    cmap = ['y', 'b', 'g', 'm']
    colors = np.copy(y)
    colors = np.array([cmap[y_i] for y_i in y])
    colors[idxs] = 'r'

    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7)
    plt.show()
