import numpy as np

from rsbp import RSBP, EntropicRSBP
from rsbp import calc_f_rsbp, calc_B, calc_logB
from utils import norm_inf

from scipy.special import logsumexp


def robust_ibp_raw(p: EntropicRSBP,
                   k_stop: int,
                   float_type=np.float64):
    # Find problem dimension
    n = p.C.shape[1]
    m = p.w.shape[0]

    # Initialize
    u = np.zeros((m, n), dtype=float_type)
    v = np.zeros((m, n), dtype=float_type)

    # Loop
    k = 0
    while True:
        Xk = calc_logB(p, u, v)

        if k >= k_stop:
            break

        # Update
        if k % 2 == 0:
            for i in range(m):
                log_ak = logsumexp(Xk[i], -1)
                u[i] = (u[i] / p.eta + np.log(p.p[i]) - log_ak) \
                    * (p.eta * p.tau) / (p.eta + p.tau)
        else:
            for i in range(m):
                log_bk = logsumexp(Xk, 0)
                v[i] = (v[i] / p.eta - log_bk) * p.eta
                for j in range(m):
                    j -= p.w[j] * (v[j] / p.eta - log_bk) * p.eta

        k += 1

    return np.exp(Xk - logsumexp(Xk))


# def robust_sinkhorn(p: EntropicROT,
#                     k_stop: int,
#                     save_uv: bool = True,
#                     float_type=np.float64,
#                     verbose: bool = False):
#     log = dict()
#     log['f'] = []
#     if save_uv:
#         log['u'] = []
#         log['v'] = []

#     # Find problem dimension
#     n = p.C.shape[0]

#     # Initialize
#     u = np.zeros(n, dtype=float_type)
#     v = np.zeros(n, dtype=float_type)

#     if save_uv:
#         log['u'].append(u)
#         log['v'].append(v)

#     # Loop
#     scale = (p.eta * p.tau) / (p.eta + p.tau)

#     k = 0
#     while True:
#         Xk = calc_logB(p, u, v)

#         _Xk = np.exp(Xk - logsumexp(Xk))
#         f = calc_f_rot(p, _Xk)
#         log['f'].append(f)

#         if verbose and k % 1000 == 0:
#             print(k, f)

#         if k >= k_stop:
#             if verbose:
#                 print(k, f)
#             break

#         # Update
#         if k % 2 == 0:
#             log_ak = logsumexp(Xk, -1)
#             u = (u / p.eta + np.log(p.a) - log_ak) * scale
#         else:
#             log_bk = logsumexp(Xk, 0)
#             v = (v / p.eta + np.log(p.a) - log_bk) * scale

#         if save_uv:
#             log['u'].append(u)
#             log['v'].append(v)

#         k += 1

#     if save_uv:
#         log['u'] = np.vstack(log['u'])
#         log['v'] = np.vstack(log['v'])

#     return Xk, log


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

    # Find problem dimension
    n = p.C.shape[1]
    m = p.w.shape[0]

    # Initialize
    u = np.zeros((m, n), dtype=float_type)
    v = np.zeros((m, n), dtype=float_type)

    if save_uv:
        log['u'].append(u)
        log['v'].append(v)

    # Loop
    k = 0
    c = 0
    while True:
        Xk = calc_logB(p, u, v)

        _Xk = np.exp(Xk - logsumexp(Xk))
        f = calc_f_rsbp(p, _Xk)
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
            for i in range(m):
                u[i] = (u[i] / p.eta + np.log(p.p[i]) - log_ak[i]) \
                    * (p.eta * p.tau) / (p.eta + p.tau)
        else:
            log_bk = logsumexp(Xk, 1)
            v_old = v.copy()
            for i in range(m):
                v[i] = (v_old[i] / p.eta - log_bk[i]) * p.eta
                for j in range(m):
                    v[i] -= p.w[j] * (v_old[j] / p.eta - log_bk[j]) * p.eta

        if save_uv:
            log['u'].append(u)
            log['v'].append(v)

        k += 1

    if save_uv:
        log['u'] = np.vstack(log['u'])
        log['v'] = np.vstack(log['v'])

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
