import numpy as np
from scipy.special import logsumexp

from src.prw import EntropicPRW
from src.utils import norm_inf


def RBCD(p: EntropicPRW,
         u0: np.ndarray, v0: np.ndarray,
         U0: np.ndarray,
         tau: np.float64,
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

    rawC_inf = norm_inf(((p.X[..., None] - p.Y[..., None].T) ** 2).sum(1))

    while True:
        # Compute pi using current U
        log_pi = p.calc_logpi(u, v, U)

        # Update u
        log_ak = logsumexp(log_pi, -1)
        u = u + np.log(p.a) - log_ak

        # Update v
        log_bk = logsumexp(log_pi, 0)
        v = v + np.log(p.b) - log_bk

        # Compute Vpi using new u and v
        pi = np.exp(log_pi)
        A = p.X.T @ pi @ p.Y
        Vpi = p.X.T @ np.diag(pi.sum(-1)) @ p.X  \
            + p.Y.T @ np.diag(pi.sum(0)) @ p.Y \
            - A - A.T

        # Compute xi using new Vpi and current U
        G = 2. / p.eta * Vpi @ U
        temp = G.T @ U
        xi = G - U @ (temp + temp.T) / 2.

        # Compute U^{(t+1)}
        U, _ = np.linalg.qr(U + tau * xi)

        # Log
        f = np.trace(U.T @ Vpi @ U)
        log['f'].append(f)
        if save_uv:
            log['u'].append(u.copy())
            log['v'].append(v.copy())
        if save_U:
            log['U'].append(U.copy())

        # Check stopping condition
        if 4 * p.eta * np.linalg.norm(xi) <= eps_1 \
                and 8 * rawC_inf * np.linalg.norm(p.a - pi.sum(-1)) <= eps_2 \
                and 8 * rawC_inf * np.linalg.norm(p.b - pi.sum(0)) <= eps_2:
            break

    return log
