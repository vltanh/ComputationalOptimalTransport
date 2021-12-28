import numpy as np
import matplotlib.pyplot as plt

from rsot import RSOT, EntropicRSOT
from rsot import exact_rsot
from rsot import calc_f_rsot
from sinkhorn import calc_U, calc_k_formula, robust_semisinkhorn_eps

# Dimension
n = 100

# Regularization
tau = np.float128(1.0)

# Number of eps
neps = 20

np.random.seed(3698)

# Cost matrix
C = np.random.uniform(low=1.0, high=50.0, size=(n, n)).astype(np.float128)
C = (C + C.T) / 2.0

# Marginal vectors
a = np.random.uniform(0.1, 1.0, size=n).astype(np.float128)
b = np.random.uniform(0.1, 1.0, size=n).astype(np.float128)

a = a / a.sum()
b = b / b.sum()

# Original UOT problem
rsot = RSOT(C, a, b, tau)

# Optimal solution
f_optimal, X_optimal = exact_rsot(rsot)

print('Optimal:', f_optimal)

eps = 1e-3

U = calc_U(rsot, eps)
eta = eps / U

# Convert to Entropic Regularized UOT
ersot = EntropicRSOT(C, a, b, tau, eta)

# Sinkhorn
_, log = robust_semisinkhorn_eps(ersot, f_optimal, eps, patience=1000,
                                 save_uv=False, verbose=True)
