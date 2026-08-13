"""
Correctness checks for tau_ij = min_x max(d_i(x), d_j(x)).

Three independent checks:
  1. Closed form. If Sigma_i == Sigma_j == S, symmetry puts the touching point
     at the midpoint, so tau = 0.5 * d_M(mu_i, mu_j) under S. Exact target.
  2. KKT / equal-distance. At the optimum the two distances must be equal
     (otherwise step toward the farther mean and lower the max).
  3. Optimality vs random probing. No sampled point beats the solution.
"""
import os
import pickle
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

PKL = os.path.join(os.path.dirname(__file__), "..", "artifacts", "support_set_40phone.pkl")
with open(PKL, "rb") as f:
    b = pickle.load(f)
phones = b["phonemes"]
mus = np.asarray(b["class_mu"], float)
covs = np.asarray(b["class_cov"], float)
inv = np.stack([np.linalg.inv(c) for c in covs])
K, D = mus.shape


def dm(x, mu, ci):
    d = x - mu
    return float(np.sqrt(max(d @ ci @ d, 0.0)))


def tau_pair(mu_i, ci_i, mu_j, ci_j):
    x0 = 0.5 * (mu_i + mu_j)
    z0 = np.concatenate([x0, [max(dm(x0, mu_i, ci_i), dm(x0, mu_j, ci_j))]])
    cons = []
    for mu, ci in ((mu_i, ci_i), (mu_j, ci_j)):
        def c(z, mu=mu, ci=ci):
            d = z[:-1] - mu
            return z[-1] - np.sqrt(max(d @ ci @ d, 1e-300))

        def gc(z, mu=mu, ci=ci):
            d = z[:-1] - mu
            q = np.sqrt(max(d @ ci @ d, 1e-300))
            g = np.zeros_like(z)
            g[:-1] = -(ci @ d) / q
            g[-1] = 1.0
            return g
        cons.append({"type": "ineq", "fun": c, "jac": gc})
    r = minimize(lambda z: z[-1], z0,
                 jac=lambda z: np.eye(len(z0))[-1],
                 constraints=cons, method="SLSQP",
                 options={"maxiter": 500, "ftol": 1e-10})
    x = r.x[:-1]
    return float(max(dm(x, mu_i, ci_i), dm(x, mu_j, ci_j))), x


# ---- check 1: closed form when covariances are shared ----
print("=== 1. closed form, shared covariance (tau should equal 0.5 * d_M) ===")
rng = np.random.default_rng(0)
worst = 0.0
for i, j in [(0, 1), (3, 7), (12, 25), (5, 30), (18, 22)]:
    S = covs[i]                      # force both classes onto the same covariance
    ci = np.linalg.inv(S)
    got, x = tau_pair(mus[i], ci, mus[j], ci)
    d = mus[i] - mus[j]
    exact = 0.5 * float(np.sqrt(d @ ci @ d))
    err = abs(got - exact)
    worst = max(worst, err)
    print(f"  {phones[i]:>3s}/{phones[j]:<3s}  solver={got:.9f}  exact={exact:.9f}  err={err:.2e}")
print(f"  worst error: {worst:.2e}")

# ---- check 2: distances equal at the optimum ----
print("\n=== 2. equal-distance condition at the optimum ===")
gaps = []
for i, j in combinations(range(K), 2):
    t, x = tau_pair(mus[i], inv[i], mus[j], inv[j])
    gaps.append(abs(dm(x, mus[i], inv[i]) - dm(x, mus[j], inv[j])))
gaps = np.asarray(gaps)
print(f"  pairs checked        : {len(gaps)}")
print(f"  max |d_i - d_j|      : {gaps.max():.3e}")
print(f"  mean |d_i - d_j|     : {gaps.mean():.3e}")

# ---- check 3: no random point beats the solution ----
print("\n=== 3. random probing cannot beat the solution ===")
beaten = 0
tested = 0
for i, j in [(0, 1), (3, 7), (12, 25), (5, 30), (18, 22), (2, 9), (14, 31)]:
    t, x = tau_pair(mus[i], inv[i], mus[j], inv[j])
    # probe around the solution and along the line between the means
    P = x + rng.normal(scale=0.25, size=(20000, D))
    lam = rng.uniform(-0.5, 1.5, size=(20000, 1))
    P = np.vstack([P, mus[i] + lam * (mus[j] - mus[i])])
    di = np.sqrt(np.einsum("ni,ij,nj->n", P - mus[i], inv[i], P - mus[i]))
    dj = np.sqrt(np.einsum("ni,ij,nj->n", P - mus[j], inv[j], P - mus[j]))
    best = np.maximum(di, dj).min()
    tested += 1
    flag = ""
    if best < t - 1e-9:
        beaten += 1
        flag = "  <-- BEATEN"
    print(f"  {phones[i]:>3s}/{phones[j]:<3s}  solver={t:.6f}  best probe={best:.6f}{flag}")
print(f"  beaten in {beaten}/{tested} pairs")

print("\nPASS" if worst < 1e-6 and gaps.max() < 1e-4 and beaten == 0 else "\nCHECK FAILED")
