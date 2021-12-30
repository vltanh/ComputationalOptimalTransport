from dataclasses import dataclass

import numpy as np
import cvxpy as cp
from scipy.special import logsumexp

from rsot import EntropicRSOT, calc_g_rsot
from utils import calc_KL, calc_entropy

# ======================================= #
# ======================================= #


@dataclass()
class RSBP:
    C: np.ndarray
    p: np.ndarray
    w: np.ndarray
    tau: float


def calc_f_rsbp(p: RSBP,
                X: np.ndarray) -> float:
    f = 0
    for i in range(len(p.w)):
        f += p.w[i] * ((p.C[i] * X[i]).sum()
                       + p.tau * calc_KL(X[i].sum(-1), p.p[i]))
    return f


def exact_rsbp(p: RSBP, verbose: bool = False):
    n = p.C.shape[1]
    m = p.w.shape[0]
    X = [
        cp.Variable((n, n), nonneg=True)
        for _ in range(m)
    ]

    f = 0
    for i in range(m):
        f += p.w[i] \
            * (cp.sum(cp.multiply(p.C[i], X[i]))
               + p.tau * cp.sum(cp.kl_div(cp.sum(X[i], 1), p.p[i])))
    objective = cp.Minimize(f)

    constraints = [
        cp.sum(X[i]) == 1
        for i in range(m)
    ]
    constraints += [
        cp.sum(X[i], 0) == cp.sum(X[i+1], 0)
        for i in range(m-1)
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)

    return prob.value, np.array([XX.value for XX in X])


# ======================================= #
# ======================================= #


@dataclass()
class EntropicRSBP(RSBP):
    eta: float


def calc_B(p: EntropicRSBP,
           u: np.ndarray,
           v: np.ndarray) -> np.ndarray:
    m = p.w.shape[0]
    return np.array([
        np.exp((u[i][:, np.newaxis] + v[i][np.newaxis, :] - p.C[i]) / p.eta)
        for i in range(m)
    ])


def calc_logB(p: EntropicRSBP,
              u: np.ndarray,
              v: np.ndarray) -> np.ndarray:
    m = p.w.shape[0]
    return np.array([
        (u[i][:, np.newaxis] + v[i][np.newaxis, :] - p.C[i]) / p.eta
        for i in range(m)
    ])


def calc_g_rsbp(p: EntropicRSBP,
                X: np.ndarray) -> float:
    m = p.w.shape[0]
    g = 0
    for i in range(m):
        ersot = EntropicRSOT(p.C[i], p.p[i], None, p.tau, p.eta)
        g += p.w[i] * calc_g_rsot(ersot, X[i])
    return g


def calc_h_rsbp(p: EntropicRSBP,
                u: np.ndarray,
                v: np.ndarray) -> float:
    m = p.w.shape[0]
    h = 0
    logX = calc_logB(p, u, v)
    for i in range(m):
        h += p.w[i] * (
            p.eta * logsumexp(logX[i])
            + p.tau * (np.exp(- u[i] / p.tau) @ p.p[i])
        )
    return h


def exact_entreg_rsbp(p: EntropicRSBP, verbose: bool = False):
    n = p.C.shape[1]
    m = p.w.shape[0]

    u = cp.Variable(shape=(m, n))
    v = cp.Variable(shape=(m, n))

    h = 0
    for i in range(m):
        h += p.w[i] * (
            p.eta
            * cp.sum(cp.exp((u[i][:, None] + v[i][None, :] - p.C[i]) / p.eta))
            + p.tau * cp.sum(cp.multiply(cp.exp(-u[i] / p.tau), p.p[i]))
        )
    obj = cp.Minimize(h)

    constraints = [
        v.T @ p.w == 0
    ]

    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=verbose)

    return prob.value, u.value, v.value


def exact_entreg_rsbp_primal(p: EntropicRSBP,
                             with_norm_constraint: bool = False,
                             verbose: bool = False):
    n = p.C.shape[1]
    m = p.w.shape[0]
    X = [
        cp.Variable((n, n), nonneg=True)
        for _ in range(m)
    ]

    f = 0
    for i in range(m):
        f += p.w[i] \
            * (cp.sum(cp.multiply(p.C[i], X[i]))
               + p.tau * cp.sum(cp.kl_div(cp.sum(X[i], 1), p.p[i])))
    objective = cp.Minimize(f)

    constraints = [
        cp.sum(X[i], 0) == cp.sum(X[i+1], 0)
        for i in range(m-1)
    ]
    if with_norm_constraint:
        constraints += [
            cp.sum(X[i]) == 1
            for i in range(m)
        ]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)

    return prob.value, np.array([XX.value for XX in X])
