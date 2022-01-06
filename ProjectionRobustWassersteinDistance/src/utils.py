import numpy as np


def pairwise_l2(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return ((X[..., None] - Y[..., None].T) ** 2).sum(1)


def norm_inf(X: np.ndarray) -> np.float64:
    return np.max(np.abs(X))
