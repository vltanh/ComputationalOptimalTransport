from typing import Union

import numpy as np
from src.uot import UOT

from src.utils import pairwise_l2, calc_KL


class PUOT:
    def __init__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 a: np.ndarray,
                 b: np.ndarray,
                 tau: Union[np.ndarray, np.float64],
                 k: int) -> None:
        # Sets
        self.X = X.copy()
        self.Y = Y.copy()

        # Marginals
        self.a = a.copy()
        self.b = b.copy()

        # Regularization parameters
        if isinstance(tau, np.ndarray):
            assert len(tau) == 2
            self.tau = tau.copy()
        elif isinstance(tau, np.float64):
            self.tau = np.array([tau, tau], dtype=np.float64)

        # Dimensions
        self.n, self.d = self.X.shape
        self.m = self.Y.shape[0]

        self.k = k

        # Original cost
        self.C = pairwise_l2(X, Y)

    def calc_proj_cost(self, U):
        '''
        Calculate the cost matrix on the projected space.
        '''
        projX = self.X @ U
        projY = self.Y @ U
        return pairwise_l2(projX, projY)

    def calc_f(self,
               pi: np.ndarray) -> float:
        return (self.C * pi).sum() \
            + self.tau[0] * calc_KL(pi.sum(-1), self.a) \
            + self.tau[1] * calc_KL(pi.sum(0), self.b)

    def entropic_regularize(self, eta: float):
        return EntropicPUOT(self.X, self.Y, self.a, self.b, self.tau, self.k, eta)

    def original(self):
        return UOT(self.C, self.a, self.b, self.tau)


class EntropicPUOT(PUOT):
    def __init__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 a: np.ndarray,
                 b: np.ndarray,
                 tau: Union[np.ndarray, np.float64],
                 k: int,
                 eta: np.float64) -> None:
        super().__init__(X, Y, a, b, tau, k)
        self.eta = eta

    def calc_logpi(self,
                   u: np.ndarray,
                   v: np.ndarray,
                   C: np.ndarray) -> np.ndarray:
        return (u[:, np.newaxis] + v[np.newaxis, :] - C) / self.eta

    def calc_pi(self,
                u: np.ndarray,
                v: np.ndarray,
                C: np.ndarray) -> np.ndarray:
        return np.exp(self.calc_logpi(u, v, C))

    def original(self):
        return super().original().entropic_regularize(self.eta)
