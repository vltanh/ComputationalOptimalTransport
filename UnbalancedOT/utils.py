import numpy as np
from scipy.spatial.distance import cdist


def norm_inf(x: np.ndarray) -> float:
    return np.amax(np.abs(x))


def calc_entropy(P: np.ndarray) -> float:
    return -np.sum(P * np.log(P) - P)


def calc_KL(x: np.ndarray, y: np.ndarray) -> float:
    return np.sum(x * np.log(x / y) - x + y)


def get_distance_matrix(n: int, d: str = 'cityblock'):
    row_idx = np.repeat(np.arange(n), n)
    col_idx = np.tile(np.arange(n), n)
    pixel_coords = np.vstack([row_idx, col_idx]).T

    C = cdist(pixel_coords, pixel_coords, d)

    return C
