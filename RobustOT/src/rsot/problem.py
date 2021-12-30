from typing import Tuple

import numpy as np
import cvxpy as cp

from src.utils import calc_KL, calc_entropy


class RSOT:
    def __init__(self,
                 C: np.ndarray,
                 a: np.ndarray,
                 b: np.ndarray,
                 tau: float) -> None:
        self.C = C.copy()
        self.a = a.copy()
        self.b = b.copy()
        self.tau = tau

        self.n = self.C.shape[0]

    def calc_f(self,
               X: np.ndarray) -> float:
        return (self.C * X).sum() \
            + self.tau * calc_KL(X.sum(-1), self.a)

    def optimize_f(self,
                   solver: str = 'ECOS',
                   verbose: bool = False) -> np.ndarray:
        # Variables
        X = cp.Variable((self.n, self.n), nonneg=True)

        # Objective
        f = cp.sum(cp.multiply(self.C, X)) \
            + self.tau * cp.sum(cp.kl_div(cp.sum(X, 1), self.a))
        objective = cp.Minimize(f)

        # Constraints
        constraints = [
            cp.sum(X, 0) == self.b,
        ]

        # Optimize
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver, verbose=verbose)

        return X.value

    def entropic_regularize(self, eta: float):
        return EntropicRSOT(self.C, self.a, self.b, self.tau, eta)


class EntropicRSOT(RSOT):
    def __init__(self,
                 C: np.ndarray,
                 a: np.ndarray,
                 b: np.ndarray,
                 tau: float,
                 eta: float) -> None:
        super().__init__(C, a, b, tau)
        self.eta = eta

    def calc_B(self,
               u: np.ndarray,
               v: np.ndarray) -> np.ndarray:
        return np.exp(self.calc_logB(u, v))

    def calc_logB(self,
                  u: np.ndarray,
                  v: np.ndarray) -> np.ndarray:
        return (u[:, np.newaxis] + v[np.newaxis, :] - self.C) / self.eta

    def calc_g(self,
               X: np.ndarray) -> float:
        return self.calc_f(X) - self.eta * calc_entropy(X)

    def calc_h(self,
               u: np.ndarray,
               v: np.ndarray) -> float:
        B = self.calc_B(u, v)
        return self.eta * np.sum(B) \
            + self.tau * (np.exp(- u / self.tau) @ self.a) \
            - (v @ self.b)

    def optimize_g_primal(self,
                          solver: str = 'ECOS',
                          verbose: bool = False) -> np.ndarray:
        X = cp.Variable((self.n, self.n), nonneg=True)

        g = cp.sum(cp.multiply(self.C, X)) \
            + self.tau * cp.sum(cp.kl_div(cp.sum(X, 1), self.a)) \
            - self.eta * cp.sum(cp.entr(X))
        objective = cp.Minimize(g)

        constraints = [
            cp.sum(X, 0) == self.b,
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver, verbose=verbose)

        return X.value

    def optimize_g_dual(self,
                        solver: str = 'ECOS',
                        verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        u = cp.Variable(shape=self.n)
        v = cp.Variable(shape=self.n)

        h = self.eta * cp.sum(cp.exp((u[:, None] + v[None, :] - self.C) / self.eta)) \
            + self.tau * cp.sum(cp.multiply(cp.exp(-u / self.tau), self.a)) \
            - cp.sum(cp.multiply(v, self.b))
        objective = cp.Minimize(h)

        prob = cp.Problem(objective)
        prob.solve(solver=solver, verbose=verbose)

        return u.value, v.value
