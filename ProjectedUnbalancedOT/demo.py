import numpy as np


def gen_gaussian(n, m, d, k_star, seed=None):
    if seed is not None:
        np.random.seed(seed)

    mean_1 = np.zeros(d, dtype=np.float64)
    mean_2 = np.zeros(d, dtype=np.float64)

    cov_1 = np.random.randn(d, k_star).astype(np.float64)
    cov_1 = cov_1.dot(cov_1.T)
    cov_2 = np.random.randn(d, k_star).astype(np.float64)
    cov_2 = cov_2.dot(cov_2.T)

    X = np.random.multivariate_normal(mean_1, cov_1, size=n).astype(np.float64)
    Y = np.random.multivariate_normal(mean_2, cov_2, size=m).astype(np.float64)

    return X, Y


def T(x, d, dim):
    assert dim <= d
    assert dim >= 1

    return x + 2*np.sign(x)*np.array(dim*[1]+(d-dim)*[0])


def gen_fragmented_hypercube(n, m, d, dim, seed=None):
    assert dim <= d
    assert dim >= 1

    if seed is not None:
        np.random.seed(seed)

    X = np.random.uniform(-1, 1, size=(n, d))
    Y = T(np.random.uniform(-1, 1, size=(m, d)), d, dim)

    return X, Y
