import numpy as np

from rsot import RSOT, EntropicRSOT
from rsot import calc_f_rsot, calc_B, calc_logB
from utils import norm_inf

from scipy.special import logsumexp


def robust_semisinkhorn_raw(p: EntropicRSOT,
                            k_stop: int,
                            float_type=np.float64):
    # Find problem dimension
    n = p.C.shape[0]

    # Initialize
    u = np.zeros(n, dtype=float_type)
    v = np.zeros(n, dtype=float_type)

    # Loop
    k = 0
    while True:
        Xk = calc_B(p, u, v)

        if k >= k_stop:
            break

        # Update
        if k % 2 == 0:
            ak = Xk.sum(-1)
            u = (u / p.eta + np.log(p.a / ak)) \
                * (p.eta * p.tau) / (p.eta + p.tau)
        else:
            bk = Xk.sum(0)
            v = (v / p.eta + np.log(p.b / bk)) * p.eta

        k += 1

    return Xk


def robust_semisinkhorn(p: EntropicRSOT,
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
    k = 0
    c = 0
    while True:
        Xk = calc_logB(p, u, v)

        f = calc_f_rsot(p, np.exp(Xk))
        log['f'].append(f)

        if verbose and k % 1000 == 0:
            print(k, f)

        if k >= k_stop:
            if verbose:
                print(k, f)
            break

        # Update
        if k % 2 == 0:
            log_ak = logsumexp(Xk, -1)
            u = (u / p.eta + np.log(p.a) - log_ak) \
                * (p.eta * p.tau) / (p.eta + p.tau)
        else:
            log_bk = logsumexp(Xk, 0)
            v = (v / p.eta + np.log(p.b) - log_bk) * p.eta

        if save_uv:
            log['u'].append(u)
            log['v'].append(v)

        k += 1

    if save_uv:
        log['u'] = np.vstack(log['u'])
        log['v'] = np.vstack(log['v'])

    return Xk, log


def robust_semisinkhorn_eps(p: EntropicRSOT,
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
    k = 0
    c = 0
    while True:
        Xk = calc_logB(p, u, v)

        f = calc_f_rsot(p, np.exp(Xk))
        log['f'].append(f)

        if verbose and k % 1000 == 0:
            print(k, f)

        if f - f_optimal <= eps:
            c += 1
            if c > patience:
                if verbose:
                    print(k, f)
                break
        else:
            c = 1

        # Update
        if k % 2 == 0:
            log_ak = logsumexp(Xk, -1)
            u = (u / p.eta + np.log(p.a) - log_ak) \
                * (p.eta * p.tau) / (p.eta + p.tau)
        else:
            log_bk = logsumexp(Xk, 0)
            v = (v / p.eta + np.log(p.b) - log_bk) * p.eta

        if save_uv:
            log['u'].append(u)
            log['v'].append(v)

        k += 1

    if save_uv:
        log['u'] = np.vstack(log['u'])
        log['v'] = np.vstack(log['v'])

    return Xk, log


# =========================================================


def calc_R(p: EntropicRSOT) -> float:
    n = p.C.shape[0]
    R = max(norm_inf(np.log(p.a)), norm_inf(np.log(p.b))) + \
        max(np.log(n), norm_inf(p.C) / p.eta - np.log(n))
    return R


def calc_U(p: RSOT, eps: float) -> float:
    n = p.C.shape[0]
    U = max(3 * np.log(n), eps / p.tau)
    return U


def calc_k_formula(p: EntropicRSOT, eps: float) -> float:
    n = p.C.shape[0]

    R = calc_R(p)
    U = calc_U(p, eps)

    eta = eps / U

    k1 = np.log(8 * R * (2 * p.tau + eta) / (3 * eta)) \
        / np.log((p.tau + eta) / p.tau)
    k2 = (1 + p.tau / eta) \
        * np.log(3 * p.tau * R
                 * (2*(eta + p.tau) + 3*R*(2*p.tau + eta))
                 / (eta ** 2 * np.log(n)))

    k = 1 + 2 * max(k1, k2)
    return k
