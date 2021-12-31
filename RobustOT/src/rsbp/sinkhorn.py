import numpy as np

from src.rsbp.problem import RSBP, EntropicRSBP
from src.utils import norm_inf

from scipy.special import logsumexp


def robust_ibp_raw(p: EntropicRSBP,
                   k_stop: int,
                   float_type=np.float64):
    # Initialize
    u = np.zeros((p.m, p.n), dtype=float_type)
    v = np.zeros((p.m, p.n), dtype=float_type)

    # Loop
    k = 0
    while True:
        Xk = p.calc_logB(u, v)

        if k >= k_stop:
            break

        # Update
        if k % 2 == 0:
            log_ak = logsumexp(Xk, 2)
            for i in range(p.m):
                u[i] = (u[i] / p.eta + np.log(p.p[i]) - log_ak[i]) \
                    * (p.eta * p.tau) / (p.eta + p.tau)
        else:
            log_bk = logsumexp(Xk, 1)
            v_old = v.copy()
            for i in range(p.m):
                v[i] = (v_old[i] / p.eta - log_bk[i]) * p.eta
                for j in range(p.m):
                    v[i] -= p.w[j] * (v_old[j] / p.eta - log_bk[j]) * p.eta

        k += 1

    _Xk = np.exp(Xk - logsumexp(Xk, (1, 2), keepdims=True))
    return _Xk


def robust_ibp(p: EntropicRSBP,
               k_stop: int,
               save_uv: bool = True,
               float_type=np.float64,
               verbose: bool = False):
    log = dict()
    log['f'] = []
    if save_uv:
        log['u'] = []
        log['v'] = []

    # Initialize
    u = np.zeros((p.m, p.n), dtype=float_type)
    v = np.zeros((p.m, p.n), dtype=float_type)

    if save_uv:
        log['u'].append(u.copy())
        log['v'].append(v.copy())

    # Loop
    k = 0
    while True:
        Xk = p.calc_logB(u, v)

        _Xk = np.exp(Xk - logsumexp(Xk, (1, 2), keepdims=True))
        f = p.calc_f(_Xk)
        log['f'].append(f)

        if verbose and k % 1000 == 0:
            print(k, f)

        if k >= k_stop:
            if verbose:
                print(k, f)
            break

        # Update
        if k % 2 == 0:
            log_ak = logsumexp(Xk, 2)
            for i in range(p.m):
                u[i] = (u[i] / p.eta + np.log(p.p[i]) - log_ak[i]) \
                    * (p.eta * p.tau) / (p.eta + p.tau)
        else:
            log_bk = logsumexp(Xk, 1)
            v_old = v.copy()
            for i in range(p.m):
                v[i] = (v_old[i] / p.eta - log_bk[i]) * p.eta
                for j in range(p.m):
                    v[i] -= p.w[j] * (v_old[j] / p.eta - log_bk[j]) * p.eta

        if save_uv:
            log['u'].append(u.copy())
            log['v'].append(v.copy())

        k += 1

    if save_uv:
        log['u'] = np.stack(log['u'])
        log['v'] = np.stack(log['v'])

    return _Xk, log


def robust_ibp_eps(p: EntropicRSBP,
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

    # Initialize
    u = np.zeros((p.m, p.n), dtype=float_type)
    v = np.zeros((p.m, p.n), dtype=float_type)

    if save_uv:
        log['u'].append(u.copy())
        log['v'].append(v.copy())

    # Loop
    k = 0
    c = 0
    while True:
        Xk = p.calc_logB(u, v)

        _Xk = np.exp(Xk - logsumexp(Xk, (1, 2), keepdims=True))
        f = p.calc_f(_Xk)
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
            log_ak = logsumexp(Xk, 2)
            for i in range(p.m):
                u[i] = (u[i] / p.eta + np.log(p.p[i]) - log_ak[i]) \
                    * (p.eta * p.tau) / (p.eta + p.tau)
        else:
            log_bk = logsumexp(Xk, 1)
            v_old = v.copy()
            for i in range(p.m):
                v[i] = (v_old[i] / p.eta - log_bk[i]) * p.eta
                for j in range(p.m):
                    v[i] -= p.w[j] * (v_old[j] / p.eta - log_bk[j]) * p.eta

        if save_uv:
            log['u'].append(u.copy())
            log['v'].append(v.copy())

        k += 1

    if save_uv:
        log['u'] = np.stack(log['u'])
        log['v'] = np.stack(log['v'])

    return _Xk, log


# # =========================================================


# def calc_R(p: EntropicROT) -> float:
#     n = p.C.shape[0]
#     R = max(norm_inf(np.log(p.a)), norm_inf(np.log(p.b))) + \
#         max(np.log(n), norm_inf(p.C) / p.eta - np.log(n))
#     return R


def calc_U(p: RSBP, eps: float) -> float:
    n = p.C.shape[1]
    U = max(
        2 + 2 * np.log(n),
        2 * eps,
        3 * eps * np.log(n) / p.tau
    )
    return U


# def calc_k_formula(p: EntropicROT, eps: float) -> float:
#     R = calc_R(p)
#     U = calc_U(p, eps)

#     k = 1 + (p.tau * U / eps + 1) \
#         * np.log(8 * p.eta * R * p.tau * (p.tau + 1) * U / eps)
#     return k
