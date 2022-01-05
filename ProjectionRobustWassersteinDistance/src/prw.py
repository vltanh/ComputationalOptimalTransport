import numpy as np


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

    def entropic_regularize(self, eta):
        return EntropicPRW(self.X, self.Y, self.a, self.b, eta)


class EntropicPRW(PRW):
    def __init__(self, X, Y, a, b, eta) -> None:
        super().__init__(X, Y, a, b)
        self.eta = eta

    def calc_logpi(self, u, v, U):
        projX = self.X @ U
        projY = self.Y @ U
        C = ((projX[..., None] - projY[..., None].T) ** 2).sum(1)
        return (u[:, None] + v[None, :] - C) / self.eta

    def calc_pi(self, u, v, U):
        return np.exp(self.calc_logpi(u, v, U))
