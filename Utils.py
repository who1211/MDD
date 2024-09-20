import numpy as np
from scipy.sparse import coo_matrix


def compute_KNN_graph(matrix, k_degree=10):
    """ Calculate the adjacency matrix from the connectivity matrix."""

    matrix = np.abs(matrix)
    idx = np.argsort(-matrix)[:, 0:k_degree]
    matrix.sort()
    matrix = matrix[:, ::-1]
    matrix = matrix[:, 0:k_degree]

    matrix = np.nan_to_num(matrix)

    A = adjacency(matrix, idx).astype(np.float32)

    return A


def adjacency(dist, idx):
    m, k = dist.shape
    assert m, k == idx.shape
    assert dist.min() >= 0 and not np.isnan(dist).any(), "Dist matrix contains negative or NaN values"

    # Weight matrix.
    I = np.arange(0, m).repeat(k)
    J = idx.reshape(m * k)
    V = dist.reshape(m * k)
    W = coo_matrix((V, (I, J)), shape=(m, m))

    W.setdiag(0)

    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)
    return W.todense()














