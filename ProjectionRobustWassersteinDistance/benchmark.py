import numpy as np
from tqdm import tqdm

from src.prw import PRW
from src.sinkhorn import RBCD, RBCD_benchmark, round_plan
from demo import gen_gaussian, gen_fragmented_hypercube

import time

SEED = 3698
np.random.seed(SEED)

nexps = 1000

d = 10

n = 10
m = 20

k_star = 3
k = 2

a = np.ones(n, dtype=np.float64) / n
b = np.ones(m, dtype=np.float64) / m

T = []
pbar = tqdm(range(nexps))
for t in pbar:
    X, Y = gen_gaussian(n, m, d, k_star)
    # X, Y = gen_fragmented_hypercube(n, m, d, k_star, seed=SEED)

    prw = PRW(X, Y, a, b)

    eta = 10.0
    eprw = prw.entropic_regularize(eta)

    # Initial u and v
    u0, v0 = np.zeros(n), np.zeros(m)

    # Initial U
    U0, _ = np.linalg.qr(np.random.randn(d, k))

    # Regularization parameters
    tau = np.float64(0.1)

    # Accuracy threshold
    eps_1 = np.float64(0.001)
    eps_2 = np.float64(0.001)

    # Run algorithm
    tic = time.perf_counter()
    RBCD_benchmark(eprw, u0, v0, U0, tau, eps_1, eps_2)
    toc = time.perf_counter()
    T.append(toc - tic)
    pbar.set_description(f'{np.mean(T):.04f} +/- {np.std(T):.05f}')
