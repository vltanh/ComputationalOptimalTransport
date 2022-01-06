from typing import Tuple, Union

import numpy as np
import cvxpy as cp

from src.utils import calc_KL, calc_entropy


class UOT:
    def __init__(self,
                 C: np.ndarray,
                 a: np.ndarray,
                 b: np.ndarray,
                 tau: Union[np.ndarray, np.float64]) -> None:
        self.C = C.copy()

        self.a = a.copy()
        self.b = b.copy()

        if isinstance(tau, np.ndarray):
            assert len(tau) == 2
            self.tau = tau.copy()
        elif isinstance(tau, np.float64):
            self.tau = np.array([tau, tau], dtype=np.float64)

        self.n, self.m = self.C.shape

    def calc_f(self,
               pi: np.ndarray) -> float:
        return (self.C * pi).sum() \
            + self.tau[0] * calc_KL(pi.sum(-1), self.a) \
            + self.tau[1] * calc_KL(pi.sum(0), self.b)

    def optimize_f(self,
                   solver: str = 'ECOS',
                   verbose: bool = False) -> np.ndarray:
        pi = cp.Variable((self.n, self.m), nonneg=True)

        f = cp.sum(cp.multiply(self.C, pi)) \
            + self.tau[0] * cp.sum(cp.kl_div(cp.sum(pi, 1), self.a)) \
            + self.tau[0] * cp.sum(cp.kl_div(cp.sum(pi, 0), self.b))
        objective = cp.Minimize(f)

        prob = cp.Problem(objective)
        prob.solve(solver=solver, verbose=verbose)

        return pi.value

    def entropic_regularize(self, eta: float):
        return EntropicUOT(self.C, self.a, self.b, self.tau, eta)


class EntropicUOT(UOT):
    def __init__(self,
                 C: np.ndarray,
                 a: np.ndarray,
                 b: np.ndarray,
                 tau: np.float64,
                 eta: np.float64) -> None:
        super().__init__(C, a, b, tau)
        self.eta = eta

    def calc_logpi(self,
                   u: np.ndarray,
                   v: np.ndarray) -> np.ndarray:
        return (u[:, np.newaxis] + v[np.newaxis, :] - self.C) / self.eta

    def calc_pi(self,
                u: np.ndarray,
                v: np.ndarray) -> np.ndarray:
        return np.exp(self.calc_logpi(u, v))

    def calc_g(self,
               pi: np.ndarray) -> float:
        return self.calc_f(pi) - self.eta * calc_entropy(pi)

    def calc_h(self,
               u: np.ndarray,
               v: np.ndarray) -> float:
        pi = self.calc_pi(u, v)
        return self.eta * np.sum(pi) \
            + self.tau * (np.exp(-u / self.tau) @ self.a) \
            + self.tau * (np.exp(-v / self.tau) @ self.b)

    def optimize_g(self,
                   solver: str = 'ECOS',
                   verbose: bool = False) -> np.ndarray:
        pi = cp.Variable((self.n, self.m), nonneg=True)

        g = cp.sum(cp.multiply(self.C, pi)) \
            + self.tau[0] * cp.sum(cp.kl_div(cp.sum(pi, 1), self.a)) \
            + self.tau[1] * cp.sum(cp.kl_div(cp.sum(pi, 0), self.b)) \
            - self.eta * cp.sum(cp.entr(pi))
        objective = cp.Minimize(g)

        prob = cp.Problem(objective)
        prob.solve(solver=solver, verbose=verbose)

        return pi.value

    def optimize_h(self,
                   solver: str = 'ECOS',
                   verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        u = cp.Variable(shape=self.n)
        v = cp.Variable(shape=self.m)

        h = self.eta * cp.sum((u[:, None] + v[None, :] - self.C) / self.eta) \
            + self.tau[0] * \
            cp.sum(cp.multiply(cp.exp(-u / self.tau[0]), self.a)) \
            + self.tau[1] * \
            cp.sum(cp.multiply(cp.exp(-v / self.tau[1]), self.b))
        objective = cp.Minimize(h)

        prob = cp.Problem(objective)
        prob.solve(solver=solver, verbose=verbose)

        return u.value, v.value
