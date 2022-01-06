import numpy as np

from src.utils import pairwise_l2


class WassersteinDistance:
    def __init__(self, X, Y, a, b) -> None:
        assert X.shape[0] == a.shape[0]
        assert Y.shape[0] == b.shape[0]

        self.X = X.copy()
        self.Y = Y.copy()
        self.a = a.copy()
        self.b = b.copy()

        self.n, self.d = self.X.shape
        self.m, _ = self.Y.shape


class PRW(WassersteinDistance):
    def __init__(self, X, Y, a, b) -> None:
        super().__init__(X, Y, a, b)

    def calc_proj_cost(self, U):
        projX = self.X @ U
        projY = self.Y @ U
        return pairwise_l2(projX, projY)

    def entropic_regularize(self, eta):
        return EntropicPRW(self.X, self.Y, self.a, self.b, eta)


class EntropicPRW(PRW):
    def __init__(self, X, Y, a, b, eta) -> None:
        super().__init__(X, Y, a, b)
        self.eta = eta

    def calc_logpi(self, u, v, C):
        return (u[:, None] + v[None, :] - C) / self.eta

    def calc_pi(self, u, v, C):
        return np.exp(self.calc_logpi(u, v, C))
