from typing import Tuple
import numpy as np
import cvxpy as cp
from scipy.special import logsumexp

from src.utils import calc_KL, calc_entropy


class RSBP:
    def __init__(self,
                 C: np.ndarray,
                 p: np.ndarray,
                 w: np.ndarray,
                 tau: float) -> None:
        self.C = C.copy()
        self.p = p.copy()
        self.w = w.copy()
        self.tau = tau

        self.n = C.shape[1]
        self.m = w.shape[0]

    def calc_f(self,
               X: np.ndarray) -> float:
        f = 0.0
        for i in range(self.m):
            f += self.w[i] * (
                (self.C[i] * X[i]).sum()
                + self.tau * calc_KL(X[i].sum(-1), self.p[i])
            )
        return f

    def optimize_f(self,
                   solver: str = 'ECOS',
                   verbose: bool = False) -> np.ndarray:
        X = [
            cp.Variable((self.n, self.n), nonneg=True)
            for _ in range(self.m)
        ]

        f = 0
        for i in range(self.m):
            f += self.w[i] * (
                cp.sum(cp.multiply(self.C[i], X[i]))
                + self.tau * cp.sum(cp.kl_div(cp.sum(X[i], 1), self.p[i]))
            )
        objective = cp.Minimize(f)

        constraints = [
            cp.sum(X[i]) == 1
            for i in range(self.m)
        ]
        constraints += [
            cp.sum(X[i], 0) == cp.sum(X[i+1], 0)
            for i in range(self.m - 1)
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver, verbose=verbose)

        return np.array([XX.value for XX in X])

    def entropic_regularize(self, eta: float):
        return EntropicRSBP(self.C, self.p, self.w, self.tau, eta)


class EntropicRSBP(RSBP):
    def __init__(self,
                 C: np.ndarray,
                 p: np.ndarray,
                 w: np.ndarray,
                 tau: float,
                 eta: float) -> None:
        super().__init__(C, p, w, tau)
        self.eta = eta

    def calc_B(self,
               u: np.ndarray,
               v: np.ndarray) -> np.ndarray:
        return np.exp(self.calc_logB(u, v))

    def calc_logB(self,
                  u: np.ndarray,
                  v: np.ndarray) -> np.ndarray:
        return np.array([
            (u[i][:, np.newaxis] + v[i][np.newaxis, :] - self.C[i]) / self.eta
            for i in range(self.m)
        ])

    def calc_g(self,
               X: np.ndarray) -> float:
        g = 0.0
        for i in range(self.m):
            g += self.w[i] * (
                (self.C[i] * X[i]).sum()
                + self.tau * calc_KL(X[i].sum(-1), self.p[i])
                - self.eta * calc_entropy(X[i])
            )
        return g

    def calc_h(self,
               u: np.ndarray,
               v: np.ndarray) -> float:
        h = 0.0
        logX = self.calc_logB(u, v)
        for i in range(self.m):
            h += self.w[i] * (
                self.eta * logsumexp(logX[i])
                + self.tau * (np.exp(- u[i] / self.tau) @ self.p[i])
            )
        return h

    def optimize_g(self,
                   with_norm_constraint: bool = False,
                   solver: str = 'ECOS',
                   verbose: bool = False) -> np.ndarray:
        X = [
            cp.Variable((self.n, self.n), nonneg=True)
            for _ in range(self.m)
        ]

        g = 0.0
        for i in range(self.m):
            g += self.w[i] * (
                cp.sum(cp.multiply(self.C[i], X[i]))
                + self.tau * cp.sum(cp.kl_div(cp.sum(X[i], 1), self.p[i]))
                - self.eta * cp.sum(cp.entr(X[i]))
            )
        objective = cp.Minimize(g)

        constraints = [
            cp.sum(X[i], 0) == cp.sum(X[i + 1], 0)
            for i in range(self.m - 1)
        ]
        if with_norm_constraint:
            constraints += [
                cp.sum(X[i]) == 1
                for i in range(self.m)
            ]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver, verbose=verbose)

        return np.array([XX.value for XX in X])

    def optimize_h(self,
                   solver: str = 'ECOS',
                   verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        u = cp.Variable(shape=(self.m, self.n))
        v = cp.Variable(shape=(self.m, self.n))

        h = 0
        for i in range(self.m):
            h += self.w[i] * (
                self.eta * cp.sum(
                    cp.exp((u[i][:, None] + v[i][None, :] - self.C[i]) / self.eta)
                ) + self.tau * cp.sum(
                    cp.multiply(cp.exp(-u[i] / self.tau), self.p[i])
                )
            )
        objective = cp.Minimize(h)

        constraints = [
            v.T @ self.w == 0
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver, verbose=verbose)

        return u.value, v.value
