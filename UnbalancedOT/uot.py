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


def exact_uot(p: UOT):
    n = p.C.shape[0]
    X = cp.Variable((n, n), nonneg=True)

    row_sums = cp.sum(X, axis=1)
    col_sums = cp.sum(X, axis=0)

    obj = cp.sum(cp.multiply(X, p.C))

    obj -= p.tau * cp.sum(cp.entr(row_sums))
    obj -= p.tau * cp.sum(cp.entr(col_sums))

    obj -= p.tau * cp.sum(cp.multiply(row_sums, cp.log(p.a)))
    obj -= p.tau * cp.sum(cp.multiply(col_sums, cp.log(p.b)))

    obj -= 2 * p.tau * cp.sum(X)
    obj += p.tau * cp.sum(p.a + p.b)

    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()

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


def calc_g(p: EntRegUOT,
           u: np.ndarray,
           v: np.ndarray) -> float:
    B = calc_B(p, u, v)
    return calc_f(p, B) - p.eta * calc_entropy(B)


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
