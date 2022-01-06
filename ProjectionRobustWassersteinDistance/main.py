import numpy as np
import matplotlib.pyplot as plt

from src.prw import PRW
from src.sinkhorn import RBCD, round_plan
from src.utils import norm_inf
from demo import gen_gaussian, gen_fragmented_hypercube
from visualize import visualize_2D_plan, visualize_3D_plan

SEED = 3698

d = 30

n = 200
m = 100

k_star = 5
k = 10

a = np.ones(n, dtype=np.float64) / n
b = np.ones(m, dtype=np.float64) / m

X, Y = gen_gaussian(n, m, d, k_star, seed=SEED)
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
eps_1 = np.float64(0.01)
eps_2 = np.float64(0.01)

# Run algorithm
log = RBCD(eprw, u0, v0, U0, tau, eps_1, eps_2, save_uv=True, save_U=True)

u, v = log['u'][-1], log['v'][-1]
U = log['U'][-1]

C = eprw.calc_proj_cost(U)
pi = round_plan(eprw.calc_pi(u, v, C), a, b)

f = log['f'][-1]

fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
ax.imshow(pi)
plt.show()
