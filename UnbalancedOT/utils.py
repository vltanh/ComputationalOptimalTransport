import numpy as np
from scipy.special import xlogy


def norm_inf(x: np.ndarray) -> float:
    return np.amax(np.abs(x))


def calc_entropy(P: np.ndarray) -> float:
    return -np.sum(xlogy(P, P) - P)


def calc_KL(x: np.ndarray, y: np.ndarray) -> float:
    print(x.dtype, y.dtype)
    return np.sum(xlogy(x, x / y) - x + y)
