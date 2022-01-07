import numpy as np
from scipy.special import logsumexp

from src.uot import UOT, EntropicUOT
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


def solve_entropic_uot(p: EntropicUOT,
                       u0: np.ndarray,
                       v0: np.ndarray,
                       niters: int):
    # Initialize
    u = u0.copy()
    v = v0.copy()

    # Loop
    for k in range(niters):
        log_pi = p.calc_logpi(u, v)

        # Update
        if k % 2 == 0:
            log_ak = logsumexp(log_pi, -1)
            u = (u / p.eta + np.log(p.a) - log_ak) \
                * p.eta * p.tau[0] / (p.eta + p.tau[0])
        else:
            log_bk = logsumexp(log_pi, 0)
            v = (v / p.eta + np.log(p.b) - log_bk) \
                * p.eta * p.tau[1] / (p.eta + p.tau[1])

    return u.copy(), v.copy()


def calc_R(p: EntropicUOT) -> float:
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
    U = max(S + T, 2 * eps, 4 * eps * np.log(n) / p.tau[0],
            4 * eps * (alpha + beta) * np.log(n) / p.tau[0])

    return U


def calc_k_stop(p: EntropicUOT, eps: float) -> float:
    R = calc_R(p)
    U = calc_U(p, eps)
    k_float = (p.tau[0] * U / eps + 1) * (np.log(8 * p.eta * R) +
                                          np.log(p.tau[0] * (p.tau[0] + 1)) + 3 * np.log(U / eps))
    return 1 + int(k_float)


def solve_entropic_puot(p: EntropicPUOT,
                        u0: np.ndarray, v0: np.ndarray,
                        U0: np.ndarray,
                        delta: np.float64,
                        eps_uv: np.float64, eps_U: np.float64,
                        max_uv: int = None, max_U: int = None,
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

    k = 0
    while True:
        # === Update u/v ===

        # Compute projected cost matrix
        C = p.calc_proj_cost(U)

        # UOT problem
        uot = UOT(C, p.a, p.b, p.tau)

        # Entropic UOT problem
        eta = eps_uv / calc_U(uot, eps_uv)
        euot = uot.entropic_regularize(eta)

        # Calculate k
        n_uv = calc_k_stop(euot, eps_uv)
        if max_uv is not None:
            n_uv = min(n_uv, max_uv)

        # Solve Entropic UOT
        u, v = solve_entropic_uot(euot, u, v, n_uv)

        # === Update U ===
        n_U = 0
        while True:
            # Compute projected cost matrix
            C = p.calc_proj_cost(U)

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

            n_U += 1

            grad_norm = np.linalg.norm(xi)
            if grad_norm <= eps_U or (max_U is not None and n_U >= max_U):
                break

        # Log
        C = p.calc_proj_cost(U)
        pi = p.calc_pi(u, v, C)
        A = p.X.T @ pi @ p.Y
        Vpi = p.X.T @ np.diag(pi.sum(-1)) @ p.X  \
            + p.Y.T @ np.diag(pi.sum(0)) @ p.Y \
            - A - A.T
        f = np.trace(U.T @ Vpi @ U)

        log['f'].append(f)
        if save_uv:
            log['u'].append(u.copy())
            log['v'].append(v.copy())
        if save_U:
            log['U'].append(U.copy())

        k += 1
        if k % 1 == 0:
            print(k, f, n_uv, n_U, grad_norm)

        # Check stopping condition
        if grad_norm <= eps_U and n_U == 1:
            break

    log['f'] = np.array(log['f'])
    if save_uv:
        log['u'] = np.array(log['u'])
        log['v'] = np.array(log['v'])
    if save_U:
        log['U'] = np.array(log['U'])

    return log
