import numpy as np

from src.prw import PRW
from src.sinkhorn import RBCD

import pickle
import time

data = pickle.load(open('data/mnist_features.pkl', 'rb'))
N = len(data)

for i in range(N):
    for j in range(i + 1, N):
        print(f'[{i}][{j}]')

        tic = time.perf_counter()

        X, Y = data[i].astype(np.float64), data[j].astype(np.float64)

        n, d = X.shape
        m = Y.shape[0]

        k = 2

        a = np.ones(n, dtype=np.float64) / n
        b = np.ones(m, dtype=np.float64) / m

        prw = PRW(X, Y, a, b)

        eta = np.float64(8.0)
        eprw = prw.entropic_regularize(eta)

        # Initial u and v
        u0, v0 = np.zeros(n), np.zeros(m)

        # Initial U
        U0, _ = np.linalg.qr(np.random.randn(d, k))

        # Regularization parameters
        tau = np.float64(0.0005)

        # Accuracy threshold
        eps_1 = np.float64(0.1)
        eps_2 = np.float64(0.1)

        # Run algorithm
        log = RBCD(eprw,
                   u0, v0,
                   U0,
                   tau,
                   eps_1, eps_2, save_uv=True, save_U=True)

        toc = time.perf_counter()

        print(f"distance: {log['f'][-1]}, elapsed: {toc - tic}")

        pickle.dump(log, open(f'{i}-{j}.pkl', 'wb'))
