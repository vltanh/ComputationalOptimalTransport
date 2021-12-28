import numpy as np
from scipy.special import logsumexp

from uot import UOT, EntRegUOT
from uot import calc_f, calc_B, calc_logB
from utils import norm_inf


def sinkhorn_raw(p: EntRegUOT,
                 niters: int,
                 float_type=np.float64):
    # Find problem dimension
    n = p.C.shape[0]

    # Initialize
    u = np.zeros(n, dtype=float_type)
    v = np.zeros(n, dtype=float_type)

    # Loop
    scale = p.eta * p.tau / (p.eta + p.tau)
    for k in range(niters):
        X = calc_B(p, u, v)

        # Update
        if k % 2 == 0:
            ak = X.sum(-1)
            u = (u / p.eta + np.log(p.a / ak)) * scale
        else:
            bk = X.sum(0)
            v = (v / p.eta + np.log(p.b / bk)) * scale

    return calc_B(p, u, v)


def sinkhorn(p: EntRegUOT,
             k_stop: int,
             save_uv: bool = True,
             float_type=np.float64,
             verbose: bool = False):
    log = dict()
    log['f'] = []
    if save_uv:
        log['u'] = []
        log['v'] = []

    # Find problem dimension
    n = p.C.shape[0]

    # Initialize
    u = np.zeros(n, dtype=float_type)
    v = np.zeros(n, dtype=float_type)

    if save_uv:
        log['u'].append(u)
        log['v'].append(v)

    # Loop
    scale = p.eta * p.tau / (p.eta + p.tau)

    k = 0
    while True:
        Xk = calc_logB(p, u, v)

        f = calc_f(p, np.exp(Xk))
        log['f'].append(f)

        if verbose and k % 1000 == 0:
            print(k, f)

        if k >= k_stop:
            break

        # Update
        if k % 2 == 0:
            log_ak = logsumexp(Xk, -1)
            u = (u / p.eta + np.log(p.a) - log_ak) * scale
        else:
            log_bk = logsumexp(Xk, 0)
            v = (v / p.eta + np.log(p.b) - log_bk) * scale

        if save_uv:
            log['u'].append(u)
            log['v'].append(v)

        k += 1

    if save_uv:
        log['u'] = np.vstack(log['u'])
        log['v'] = np.vstack(log['v'])

    return Xk, log


def sinkhorn_eps(p: EntRegUOT,
                 f_optimal: float,
                 eps: float,
                 patience: int = 0,
                 save_uv: bool = True,
                 float_type=np.float64,
                 verbose: bool = False):
    log = dict()
    log['f'] = []
    if save_uv:
        log['u'] = []
        log['v'] = []

    # Find problem dimension
    n = p.C.shape[0]

    # Initialize
    u = np.zeros(n, dtype=float_type)
    v = np.zeros(n, dtype=float_type)

    if save_uv:
        log['u'].append(u)
        log['v'].append(v)

    # Loop
    scale = p.eta * p.tau / (p.eta + p.tau)

    k = 0
    c = 0
    while True:
        Xk = calc_logB(p, u, v)

        f = calc_f(p, np.exp(Xk))
        log['f'].append(f)

        if verbose and k % 1000 == 0:
            print(k, f)

        if f - f_optimal <= eps:
            c += 1
            if c > patience:
                break
        else:
            c = 1

        # Update
        if k % 2 == 0:
            log_ak = logsumexp(Xk, -1)
            u = (u / p.eta + np.log(p.a) - log_ak) * scale
        else:
            log_bk = logsumexp(Xk, 0)
            v = (v / p.eta + np.log(p.b) - log_bk) * scale

        if save_uv:
            log['u'].append(u)
            log['v'].append(v)

        k += 1

    if save_uv:
        log['u'] = np.vstack(log['u'])
        log['v'] = np.vstack(log['v'])

    return Xk, log

# =========================================================


def calc_R(p: EntRegUOT) -> float:
    n = p.C.shape[0]
    R = max(norm_inf(np.log(p.a)), norm_inf(np.log(p.b))) + \
        max(np.log(n), norm_inf(p.C) / p.eta - np.log(n))
    return R


def calc_U(p: UOT, eps: float) -> float:
    n = p.C.shape[0]
    alpha, beta = p.a.sum(), p.b.sum()

    S = 0.5 * (alpha + beta) + 0.5 + 0.25 / np.log(n)
    T = 0.5 * (alpha + beta) * (np.log(0.5 * (alpha + beta)) +
                                2 * np.log(n) - 1) + np.log(n) + 2.5
    U = max(S + T, 2 * eps, 4 * eps * np.log(n) / p.tau,
            4 * eps * (alpha + beta) * np.log(n) / p.tau)

    return U


def calc_k_stop(p: EntRegUOT, eps: float) -> float:
    R = calc_R(p)
    U = calc_U(p, eps)
    k_float = (p.tau * U / eps + 1) * (np.log(8 * p.eta * R) +
                                       np.log(p.tau * (p.tau + 1)) + 3 * np.log(U / eps))
    return k_float
