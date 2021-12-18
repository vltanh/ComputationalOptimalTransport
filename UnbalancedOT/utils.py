import numpy as np


def norm_inf(x: np.ndarray) -> float:
    return np.amax(np.abs(x))


def calc_entropy(P: np.ndarray) -> float:
    return -np.sum(P * np.log(P + 1e-9) - P)


def calc_KL(x: np.ndarray, y: np.ndarray) -> float:
    return np.sum(x * np.log(x / y) - x + y)
