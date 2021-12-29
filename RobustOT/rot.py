from dataclasses import dataclass

import numpy as np
import cvxpy as cp
from scipy.special import logsumexp

from utils import calc_KL, calc_entropy

# ======================================= #
# ======================================= #


@dataclass()
class ROT:
    C: np.ndarray
    a: np.ndarray
    b: np.ndarray
    tau: float


def calc_f_rot(p: ROT,
               X: np.ndarray) -> float:
    return (p.C * X).sum() \
        + p.tau * calc_KL(X.sum(-1), p.a) \
        + p.tau * calc_KL(X.sum(0), p.b)


def exact_rot(p: ROT, verbose: bool = False):
    n = p.C.shape[0]
    X = cp.Variable((n, n), nonneg=True)

    f = cp.sum(cp.multiply(p.C, X)) \
        + p.tau * cp.sum(cp.kl_div(cp.sum(X, 1), p.a)) \
        + p.tau * cp.sum(cp.kl_div(cp.sum(X, 0), p.b))
    objective = cp.Minimize(f)

    constraints = [
        cp.sum(X) == 1.0,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)

    return prob.value, X.value


# ======================================= #
# ======================================= #


@dataclass()
class EntropicROT(ROT):
    eta: float


def calc_B(p: EntropicROT,
           u: np.ndarray,
           v: np.ndarray) -> np.ndarray:
    return np.exp((u[:, np.newaxis] + v[np.newaxis, :] - p.C) / p.eta)


def calc_logB(p: EntropicROT,
              u: np.ndarray,
              v: np.ndarray) -> np.ndarray:
    return (u[:, np.newaxis] + v[np.newaxis, :] - p.C) / p.eta


def calc_g_rot(p: EntropicROT,
               X: np.ndarray) -> float:
    return calc_f_rot(p, X) - p.eta * calc_entropy(X)


def calc_h_rot(p: EntropicROT,
               u: np.ndarray,
               v: np.ndarray) -> float:
    B = calc_B(p, u, v)
    return p.eta * np.sum(B) \
        + p.tau * (np.exp(- u / p.tau) @ p.a) \
        + p.tau * (np.exp(- v / p.tau) @ p.b)


def exact_entreg_rot(p: EntropicROT, verbose: bool = False):
    n = p.C.shape[0]

    u = cp.Variable(shape=n)
    v = cp.Variable(shape=n)

    h = p.eta * cp.sum(cp.exp((u[:, None] + v[None, :] - p.C) / p.eta)) \
        + p.tau * cp.sum(cp.multiply(cp.exp(-u / p.tau), p.a)) \
        + p.tau * cp.sum(cp.multiply(cp.exp(-v / p.tau), p.b))
    obj = cp.Minimize(h)

    prob = cp.Problem(obj)
    prob.solve(verbose=verbose)

    return prob.value, u.value, v.value


def exact_entreg_rot_primal(p: EntropicROT, verbose: bool = False):
    n = p.C.shape[0]
    X = cp.Variable((n, n), nonneg=True)

    f = cp.sum(cp.multiply(p.C, X)) \
        + p.tau * cp.sum(cp.kl_div(cp.sum(X, 1), p.a)) \
        - p.eta * cp.sum(cp.entr(X))
    objective = cp.Minimize(f)

    constraints = [
        cp.sum(X) == 1.0,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)

    return prob.value, X.value
