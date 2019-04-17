import warnings
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


class CentralityMeasure(object):

    def __init__(self, X, k):
        self.X = X
        self.k = int(k)
        self.A, self.D = self._make_graph(X)

    def _make_graph(self, X):
        neighbors, distances = self._KNN()
        # Convert these to the adjacency matrix and distance matrix
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

    def centrality(self):
        raise NotImplementedError


class EigenvectorCentrality(CentralityMeasure):

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


class LDS(CentralityMeasure):

    def __init__(self, X, k=30, weight_func="intersection"):
        super().__init__(X, k)  # Computes the adjacency graph
        self.weight_func = weight_func

    def _weight_intersection(self, graph, i, j):
        # Because graph is the adjacency matrix,
        # this is the same as the intersection
        # Elementwise multiplication
        if graph.dtype != int and graph.dtype != bool:
            msg = "Input does not seem to be an adjacency matrix."
            warnings.warn(msg, RuntimeWarning)
        return np.sum(graph[i, ].multiply(graph[j, ]))

    def _weight_closeness(self, graph, i, j):
        # N.B. graph must be a distance matrix.
        if graph.dtype != float:
            msg = "Input does not seem to be a distance matrix."
            warnings.warn(msg, RuntimeWarning)
        dist = graph[i, j]
        # If the distance is 0 that means that we have two identical points.
        # Hence their similarity is infinite.
        if dist == 0:
            msg = "inf generated. Check that data does not include duplicates"
            warnings.warn(msg, RuntimeWarning)
            return np.inf
        # Return the similarity of the two points
        return 1 / dist

    def _weight_popularity(self, graph, i, j):
        if graph.dtype != int and graph.dtype != bool:
            msg = "Input does not seem to be an adjacency matrix."
            warnings.warn(msg, RuntimeWarning)
        # 1 if i is connected to j, else 0.
        return graph[j, i]

    def centrality(self):
        weight_func_str = f"_weight_{self.weight_func}"
        wf = getattr(self, weight_func_str)

        # Closeness uses the distance matrix.
        if self.weight_func == "closeness":
            graph = self.D
        else:
            graph = self.A

        scores = np.zeros(self.X.shape[0], dtype=float)
        for x_i in range(self.X.shape[0]):
            neighbor_idxs = np.nonzero(self.A[x_i])[1]
            score = np.sum([wf(graph, x_i, x_j) for x_j in neighbor_idxs])
            scores[x_i] = score / self.k
        return scores


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris, load_digits, load_breast_cancer
    from sklearn.manifold import TSNE

    np.random.seed(0)

    iris = load_digits()
    X = TSNE(n_components=2).fit_transform(iris.data)
    y = iris.target

    N = 50
    #cent = LDS(X, k=20, weight_func="intersection")
    cent = EigenvectorCentrality(X, k=20, n='auto')
    scores = cent.centrality()
    idxs = np.argsort(scores)[::-1][:N]
    print(f"N: {len(idxs)}")
    print(f"Scores: {scores.max()} {scores.min()}")
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
