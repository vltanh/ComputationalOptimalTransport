from dataclasses import dataclass

import numpy as np
import cvxpy as cp

from utils import calc_KL, calc_entropy

# ======================================= #
# ======================================= #


@dataclass()
class RSOT:
    C: np.ndarray
    a: np.ndarray
    b: np.ndarray
    tau: float


def calc_f_rsot(p: RSOT,
                X: np.ndarray) -> float:
    return (p.C * X).sum() \
        + p.tau * calc_KL(X.sum(-1), p.a)


def exact_rsot(p: RSOT, verbose: bool = False):
    n = p.C.shape[0]
    X = cp.Variable((n, n), nonneg=True)

    f = cp.sum(cp.multiply(p.C, X)) \
        + p.tau * cp.sum(cp.kl_div(cp.sum(X, 1), p.a))
    objective = cp.Minimize(f)

    constraints = [
        cp.sum(X, 0) == p.b,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)

    return prob.value, X.value


# ======================================= #
# ======================================= #


@dataclass()
class EntropicRSOT(RSOT):
    eta: float


def calc_B(p: EntropicRSOT,
           u: np.ndarray,
           v: np.ndarray) -> np.ndarray:
    return np.exp((u[:, np.newaxis] + v[np.newaxis, :] - p.C) / p.eta)


def calc_logB(p: EntropicRSOT,
              u: np.ndarray,
              v: np.ndarray) -> np.ndarray:
    return (u[:, np.newaxis] + v[np.newaxis, :] - p.C) / p.eta


def calc_g_rsot(p: EntropicRSOT,
                X: np.ndarray) -> float:
    return calc_f_rsot(p, X) - p.eta * calc_entropy(X)


def calc_h_rsot(p: EntropicRSOT,
                u: np.ndarray,
                v: np.ndarray) -> float:
    B = calc_B(p, u, v)
    return p.eta * np.sum(B) \
        + p.tau * (np.exp(- u / p.tau) @ p.a) \
        - (v @ p.b)


def exact_entreg_rsot(p: EntropicRSOT, verbose: bool = False):
    n = p.C.shape[0]

    u = cp.Variable(shape=n)
    v = cp.Variable(shape=n)

    h = p.eta * cp.sum(cp.exp((u[:, None] + v[None, :] - p.C) / p.eta)) \
        + p.tau * cp.sum(cp.multiply(cp.exp(-u / p.tau), p.a)) \
        - cp.sum(cp.multiply(v, p.b))
    obj = cp.Minimize(h)

    prob = cp.Problem(obj)
    prob.solve(verbose=verbose)

    return prob.value, u.value, v.value


def exact_entreg_rsot_primal(p: EntropicRSOT, verbose: bool = False):
    n = p.C.shape[0]
    X = cp.Variable((n, n), nonneg=True)

    f = cp.sum(cp.multiply(p.C, X)) \
        + p.tau * cp.sum(cp.kl_div(cp.sum(X, 1), p.a)) \
        - p.eta * cp.sum(cp.entr(X))
    objective = cp.Minimize(f)

    constraints = [
        cp.sum(X, 0) == p.b,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)

    return prob.value, X.value
