import numpy as np

from uot import UOT, EntRegUOT
from uot import calc_B
from utils import norm_inf


def sinkhorn(p: EntRegUOT, niters: int) -> np.ndarray:
    # Find problem dimension
    n = p.C.shape[0]

    log = {
        'u': np.empty((niters + 1, n), dtype=np.float64),
        'v': np.empty((niters + 1, n), dtype=np.float64),
    }

    # Initialize
    u = np.zeros(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.float64)

    # Loop
    scale = p.eta * p.tau / (p.eta + p.tau)
    for k in range(niters):
        log['u'][k] = u
        log['v'][k] = v

        X = calc_B(p, u, v)

        # Update
        if k % 2 == 0:
            ak = X.sum(-1)
            u = (u / p.eta + np.log(p.a / ak)) * scale
        else:
            bk = X.sum(0)
            v = (v / p.eta + np.log(p.b / bk)) * scale

    log['u'][-1] = u
    log['v'][-1] = v

    return calc_B(p, u, v), log


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


def sinkhorn_entreg_uot(p: EntRegUOT,
                        eps: float) -> np.ndarray:
    # Find stopping condition
    k_stop = calc_k_stop(p, eps)
    niters = 1 + int(k_stop)

    # Perform Sinkhorn iterations
    X, log = sinkhorn(p, niters)

    # Add k_stop to logger
    log['k_stop'] = k_stop

    return X, log
