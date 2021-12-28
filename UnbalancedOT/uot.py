from dataclasses import dataclass

import numpy as np
import cvxpy as cp

from utils import calc_KL, calc_entropy

# ======================================= #
# ======================================= #


@dataclass()
class UOT:
    C: np.ndarray
    a: np.ndarray
    b: np.ndarray
    tau: float


def calc_f(p: UOT,
           X: np.ndarray) -> float:
    return (p.C * X).sum() \
        + p.tau * calc_KL(X.sum(-1), p.a) \
        + p.tau * calc_KL(X.sum(0), p.b)


def exact_uot(p: UOT, verbose: bool = False):
    n = p.C.shape[0]
    X = cp.Variable((n, n), nonneg=True)

    obj = cp.sum(cp.multiply(p.C, X)) \
        + p.tau * cp.sum(cp.kl_div(cp.sum(X, 1), p.a)) \
        + p.tau * cp.sum(cp.kl_div(cp.sum(X, 0), p.b))

    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(verbose=verbose)

    return prob.value, X.value


# ======================================= #
# ======================================= #


@dataclass()
class EntRegUOT(UOT):
    eta: float


def calc_B(p: EntRegUOT,
           u: np.ndarray,
           v: np.ndarray) -> np.ndarray:
    return np.exp((u[:, np.newaxis] + v[np.newaxis, :] - p.C) / p.eta)


def calc_logB(p: EntRegUOT,
              u: np.ndarray,
              v: np.ndarray) -> np.ndarray:
    return (u[:, np.newaxis] + v[np.newaxis, :] - p.C) / p.eta


def calc_g(p: EntRegUOT,
           X: np.ndarray) -> float:
    return calc_f(p, X) - p.eta * calc_entropy(X)


def calc_g_dual(p: EntRegUOT,
                u: np.ndarray,
                v: np.ndarray) -> float:
    B = calc_B(p, u, v)
    return p.eta * np.sum(B) + p.tau * (np.exp(- u / p.tau) @ p.a) + p.tau * (np.exp(- v / p.tau) @ p.b)


def exact_entreg_uot(p: EntRegUOT):
    n = p.C.shape[0]

    u = cp.Variable(shape=n)
    v = cp.Variable(shape=n)

    obj = p.eta * cp.sum(cp.exp((u[:, None] + v[None, :] - p.C) / p.eta))
    obj += p.tau * cp.sum(cp.multiply(cp.exp(-u / p.tau), p.a))
    obj += p.tau * cp.sum(cp.multiply(cp.exp(-v / p.tau), p.b))

    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()

    return prob.value, u.value, v.value
