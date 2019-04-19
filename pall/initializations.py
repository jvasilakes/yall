import warnings
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


class CentralityMeasure(object):
    """
    :math:`score(x)= \\frac{1}{k-1} \\sum_{x_j \\in NN(x_i)} \\omega(x_i, x_j)`

    :math:`NN(x)`: The k nearest neighbors of :math:`x`.

    :math:`\\omega`: A weight method.
    """

    def __init__(self, X, k):
        self.X = X
        self.k = int(k)
        # Adjacency and distance matrices
        self.A, self.D = self._make_graphs()

    def _make_graphs(self):
        neighbors, distances = self._KNN()
        # Convert these to sparse matrix form.
        A, D = self._to_sparse_adjacency(neighbors, distances)
        return A, D

    def _KNN(self):
        NN = NearestNeighbors(n_neighbors=self.k, algorithm="brute",
                              metric="minkowski")
        NN.fit(self.X)
        dists, neighbors = NN.kneighbors()
        return neighbors, dists

    def _to_sparse_adjacency(self, neighbors, distances):
        n = neighbors.shape[0]
        A = sparse.lil_matrix((n, n), dtype=int)  # Adjacency matrix
        D = sparse.lil_matrix((n, n), dtype=float)  # Distance matrix
        for (i, js) in enumerate(neighbors):
            A[i, js] = 1
            D[i, js] = distances[i]
        return A, D

    def weight(self, i, j):
        """
        Computes the weight between nodes i and j
        according to the graph matrix.
        """
        raise NotImplementedError

    def centrality(self):
        scores = np.zeros(self.X.shape[0], dtype=float)
        for x_i in range(self.X.shape[0]):
            neighbor_idxs = np.nonzero(self.A[x_i])[1]
            score = np.sum([self.weight(x_i, x_j) for x_j in neighbor_idxs])
            scores[x_i] = score / (self.k - 1)
        return scores


class ClosenessCentrality(CentralityMeasure):
    """
    :math:`\\omega(x_i, x_j) = \\frac{1}{dist(x_i, x_j)}`
    """

    def __init__(self, X, k=30):
        super().__init__(X, k)
        # Use the distance matrix

    def weight(self, i, j):
        dist = self.D[i, j]
        # If the distance is 0 that means that we have two identical points.
        # Hence their similarity is infinite.
        if dist == 0:
            msg = "inf generated. Check that data does not include duplicates"
            warnings.warn(msg, RuntimeWarning)
            return np.inf
        # Return the similarity of the two points
        # N.B. This is actually the harmonic closeness centrality.
        return 1 / dist


class DegreeCentrality(CentralityMeasure):
    """
    :math:`\\omega(x_i, x_j) = \\delta_{ij}`
    """

    def __init__(self, X, k=30):
        super().__init__(X, k)
        # Use the adjacency matrix
        self.graph = self.A

    def weight(self, i, j):
        # 1 if i is connected to j, else 0.
        return self.A[j, i]


class EigenvectorCentrality(CentralityMeasure):
    """
    We solve for the eigenvalues :math:`\\lambda` of the
    adjecency matrix :math:`A`

    :math:`Ax = \\lambda x`

    The nodes with the highest eigenvalues :math:`\\lambda`
    are the most central.
    """

    def __init__(self, X, k=30, n="auto"):
        super().__init__(X, k)  # Computes the adjacency graph
        if n == "auto":
            msg = "Setting n to 'auto' can result in long computation times."
            warnings.warn(msg, RuntimeWarning)
            self.n = X.shape[0] - 2
        else:
            self.n = int(n)

    def centrality(self):
        vals, vecs = sparse.linalg.eigs(self.A.asfptype(), k=self.n)
        return vals


class LDSCentrality(CentralityMeasure):
    """
    :math:`\\omega(x_i, x_j) = ~\\mid NN(x_i) \\cap NN(x_j) \\mid`
    """

    def __init__(self, X, k=30):
        super().__init__(X, k)

    def weight(self, i, j):
        # Because graph is the adjacency matrix,
        # this is the same as the intersection.
        # Elementwise multiplication
        return np.sum(self.A[i, ].multiply(self.A[j, ]))


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris, load_digits, load_breast_cancer
    from pall.datasets import load_dexter
    from sklearn.manifold import TSNE

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="iris",
                        choices=["breast_cancer", "dexter", "digits", "iris"])
    parser.add_argument("--metric", type=str, default="degree",
                        choices=["closeness", "degree", "eigen", "lds"])
    args = parser.parse_args()

    np.random.seed(0)

    if args.dataset == "breast_cancer":
        dataset = load_breast_cancer()
    elif args.dataset == "dexter":
        dataset = load_dexter()
    elif args.dataset == "digits":
        dataset = load_digits()
    elif args.dataset == "iris":
        dataset = load_iris()
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'.")

    X = dataset.data
    y = dataset.target

    K = 20
    N = 50
    if args.metric == "closeness":
        cm = ClosenessCentrality(X, K)
    elif args.metric == "degree":
        cm = DegreeCentrality(X, K)
    elif args.metric == "eigen":
        cm = EigenvectorCentrality(X, K, n=N)
    elif args.metric == "lds":
        cm = LDSCentrality(X, K)

    X = TSNE(n_components=2).fit_transform(X)
    scores = cm.centrality()
    idxs = np.argsort(scores)[::-1][:N]
    print(f"K: {K}")
    print(f"N: {N}")
    print(f"Scores: {scores.max():.3f} {scores.min():.3f}")
    mask = np.zeros(X.shape[0], dtype=bool)
    mask[idxs] = True
    LX = X[mask, ]
    UX = X[np.logical_not(mask), ]
    Ly = y[mask]
    Uy = y[np.logical_not(mask)]

    Ucol = plt.cm.Set2(Uy)
    plt.scatter(UX[:, 0], UX[:, 1], c=Ucol, alpha=0.7)
    plt.scatter(LX[:, 0], LX[:, 1], c='r', alpha=0.7)
    plt.show()
