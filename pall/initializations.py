import warnings
import numpy as np
import cvxpy as cp
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from collections import Counter


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

    def find_centers(self, n=50):
        self.scores = self.centrality()
        # Return indices of the N data points with the highest scores.
        return np.argsort(self.scores)[::-1][:n]


class ClosenessCentrality(CentralityMeasure):
    """
    :math:`\\omega(x_i, x_j) = \\frac{1}{dist(x_i, x_j)}`
    """

    def __init__(self, X, k=30):
        super().__init__(X, k)  # Computes the distance graph

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
        super().__init__(X, k)  # Computes the adjacency graph

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

    # Overload base centrality.
    def centrality(self):
        vals, vecs = sparse.linalg.eigs(self.A.asfptype(), k=self.n)
        return vals


class LDSCentrality(CentralityMeasure):
    """
    :math:`\\omega(x_i, x_j) = ~\\mid NN(x_i) \\cap NN(x_j) \\mid`
    """

    def __init__(self, X, k=30):
        super().__init__(X, k)  # Compute the adjacency graph

    def weight(self, i, j):
        # Because this is the adjacency matrix, elementwise multiplication
        # is the same as the intersection.
        return np.sum(self.A[i, ].multiply(self.A[j, ]))


class SetCover(object):

    def __init__(self, X, k=30):
        self.X = X
        self.k = int(k)
        self.neighbors, self.distances = self._KNN()
        self.cover = None

    def _KNN(self):
        NN = NearestNeighbors(n_neighbors=self.k, algorithm="brute",
                              metric="minkowski")
        NN.fit(self.X)
        dists, neighbors = NN.kneighbors()
        return neighbors, dists

    def find_centers(self, n=50):
        raise NotImplementedError


class GreedySetCover(SetCover):
    """
    Given a set of partial covers :math:`S` of :math:`X`,
    greedily search for a subset of them, indexed by :math:`I`
    such that :math:`\\bigcup_{i \\in I} S_i~ = X`
    """

    def __init__(self, X, k=30):
        super().__init__(X, k)

    def _set_cover(self):
        nns_set = [set(n) for n in self.neighbors]
        elements = set.union(*nns_set)
        covered = set()
        cover = []
        while covered != elements:
            subset = max(nns_set, key=lambda s: len(s - covered))
            cover.append(list(subset))
            covered |= subset
        return np.array(cover)

    def find_centers(self, n=50):
        self.cover = self._set_cover()
        counts = Counter(self.cover.ravel())
        counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        # The indices of the first n sorted counts.
        return np.array(counts)[:n, 0]


class FacilityLocation(SetCover):
    """
    This is a simplified version of the uncapacitated facility
    location problem in which there is no cost to open a facility.
    Customers are data points and facilities are centers. The cost
    to ship from a facility to a customer is computed as the distance
    between them in a k nearest neighbor graph.

    :math:`I` : Set of candidate center locations.

    :math:`J` : Set of data points.

    N.B. In this case :math:`I ~=~ J` as each data point is a potential center.

    :math:`M` : Maximum number of centers.

    .. math::

        y_{ij} =
        \\begin{cases}
          1 & \\text{if center} ~i~ \\text{covers data point} ~j

          0 & \\text{otherwise}
        \\end{cases}

    :math:`D_{ij} =` distance between center :math:`i` and data point :math:`j`

    :math:`\\epsilon =` number of permissable outliers

    minimize :math:`\\sum_{i \\in I} \\sum_{j \\in J} D_{ij} y_{ij}`

    subject to

                :math:`\\sum_i max_j ~y_{ij} \leq M`

                :math:`\\sum_{ij} y_{ij} = ~|J| - ~\\epsilon`

                :math:`y_{ij} \\in \\{0,1\\} ~~\\forall i \\in I, j \\in J`

    """

    def __init__(self, X, k=30):
        super().__init__(X, k)

    def _distance_matrix(self):
        n = self.neighbors.shape[0]
        D = np.zeros((n, n), dtype=float)
        for (i, js) in enumerate(self.neighbors):
            D[i, js] = self.distances[i]
        return D

    def find_centers(self, n=50, epsilon="auto"):
        if epsilon == "auto":
            epsilon = int(np.ceil(1e-4 * X.shape[0]))
        else:
            epsilon = int(epsilon)
        D = self._distance_matrix()
        y = cp.Variable(D.shape, boolean=True)
        cost = cp.sum(cp.multiply(D, y))
        constraints = [cp.sum(y) == D.shape[0],
                       cp.sum(cp.max(y, axis=1)) <= n - epsilon]
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver="GUROBI")

        y_int = np.around(y.value)
        centers = np.argwhere(np.max(y_int, axis=1) == 1).ravel()
        return centers


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris, load_digits, load_breast_cancer
    from pall.datasets import load_dexter
    from sklearn.manifold import TSNE

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="iris",
                        choices=["breast_cancer", "dexter", "digits", "iris"],
                        help="Which data set to use.")
    parser.add_argument("--method", type=str, default="degree",
                        choices=["closeness", "degree", "eigen", "lds",
                                 "greedy", "facloc"],
                        help="Which method to use.")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of initial data points to find.")
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
    if args.method == "closeness":
        cm = ClosenessCentrality(X, K)
    elif args.method == "degree":
        cm = DegreeCentrality(X, K)
    elif args.method == "eigen":
        cm = EigenvectorCentrality(X, K, n=args.n)
    elif args.method == "lds":
        cm = LDSCentrality(X, K)
    elif args.method == "greedy":
        cm = GreedySetCover(X, K)
    elif args.method == "facloc":
        cm = FacilityLocation(X, K)
    else:
        raise ValueError(f"Unknown method '{args.method}'.")

    print(f"K: {K}")
    print(f"N: {args.n}")
    idxs = cm.find_centers(n=args.n)
    print(f"Found {len(idxs)} centers.")

    mask = np.zeros(X.shape[0], dtype=bool)
    mask[idxs] = True
    X = TSNE(n_components=2).fit_transform(X)
    LX = X[mask, ]
    UX = X[np.logical_not(mask), ]
    Ly = y[mask]
    Uy = y[np.logical_not(mask)]

    Ucol = plt.cm.Set2(Uy)
    plt.scatter(UX[:, 0], UX[:, 1], c=Ucol, alpha=0.7)
    plt.scatter(LX[:, 0], LX[:, 1], c='r', alpha=0.7)
    plt.show()
