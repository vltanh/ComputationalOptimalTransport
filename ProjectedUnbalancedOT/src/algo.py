import numpy as np
from scipy.special import logsumexp

from src.puot import EntropicPUOT
from src.utils import norm_inf


def round_plan(pi, a, b):
    x = np.clip(a / pi.sum(-1), a_min=None, a_max=1.0)
    X = np.diag(x)
    pi_ = X @ pi

    y = np.clip(b / pi.sum(0), a_min=None, a_max=1.0)
    Y = np.diag(y)
    pi__ = pi_ @ Y

    err_r = a - pi__.sum(-1)
    err_c = b - pi__.sum(0)
    return pi__ + err_r[:, None] @ err_c[None, :] / np.sum(np.abs(err_r))


def solve(p: EntropicPUOT,
          u0: np.ndarray, v0: np.ndarray,
          U0: np.ndarray,
          delta: np.float64,
          eps_1: np.float64, eps_2: np.float64,
          save_uv: bool = False,
          save_U: bool = False):
    u, v = u0.copy(), v0.copy()
    U = U0.copy()

    log = dict()
    log['f'] = []
    if save_uv:
        log['u'] = []
        log['v'] = []
    if save_U:
        log['U'] = []

    rawC_inf = norm_inf(p.C)

    k = 0
    while True:
        # Compute projected cost matrix
        C = p.calc_proj_cost(U)

        # Update u/v
        for _ in range(1):
            # Compute pi using current U and previous u, v
            log_pi = p.calc_logpi(u, v, C)

            # Update u
            log_ak = logsumexp(log_pi, -1)
            u = u + np.log(p.a) - log_ak

            # Compute pi using current U, u and previous v
            log_pi = p.calc_logpi(u, v, C)

            # Update v
            log_bk = logsumexp(log_pi, 0)
            v = v + np.log(p.b) - log_bk

        # Update U
        for _ in range(1):
            # Compute pi using current U, u, v
            pi = p.calc_pi(u, v, C)

            # Compute Vpi using new u and v
            A = p.X.T @ pi @ p.Y
            Vpi = p.X.T @ np.diag(pi.sum(-1)) @ p.X  \
                + p.Y.T @ np.diag(pi.sum(0)) @ p.Y \
                - A - A.T

            # Compute xi using new Vpi and current U
            G = - 2. / p.eta * Vpi @ U
            temp = G.T @ U
            xi = G - U @ (temp + temp.T) / 2.

            # Update U
            U, _ = np.linalg.qr(U - delta * xi)

        # Log
        f = np.trace(U.T @ Vpi @ U)
        log['f'].append(f)
        if save_uv:
            log['u'].append(u.copy())
            log['v'].append(v.copy())
        if save_U:
            log['U'].append(U.copy())

        k += 1
        if k % 200 == 0:
            print(k, f)

        # Check stopping condition
        if 4 * p.eta * np.linalg.norm(xi) <= eps_1 \
                and 8 * rawC_inf * np.linalg.norm(p.a - pi.sum(-1)) <= eps_2 \
                and 8 * rawC_inf * np.linalg.norm(p.b - pi.sum(0)) <= eps_2:
            break

    log['f'] = np.array(log['f'])
    if save_uv:
        log['u'] = np.array(log['u'])
        log['v'] = np.array(log['v'])
    if save_U:
        log['U'] = np.array(log['U'])

    return log
