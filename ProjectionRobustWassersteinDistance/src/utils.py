import numpy as np


def norm_inf(X: np.ndarray) -> np.float64:
    return np.max(np.abs(X))
