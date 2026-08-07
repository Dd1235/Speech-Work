"""
E1: Pairwise critical core radius, computed geometrically.

tau_ij = min_x max( d_M(x, mu_i; Sig_i), d_M(x, mu_j; Sig_j) )

Both d_M^2 are convex quadratics; sqrt is monotone; max of convex is convex.
So this is a convex program with a unique optimum -- solved as
    min t  s.t.  d_M(x,mu_i) <= t,  d_M(x,mu_j) <= t.

Interpretation: if the truncation radius tau < tau_ij, the two class cores are
disjoint sets. min over all pairs is therefore a purely geometric prediction of
the critical core radius, with no clustering algorithm involved.
"""
import os
import json
import pickle
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

PKL = os.path.join(os.path.dirname(__file__), "..", "artifacts", "support_set_40phone.pkl")
OUT = os.path.join(os.path.dirname(__file__), "results", "e1_results.json")

with open(PKL, "rb") as f:
    b = pickle.load(f)

phones = b["phonemes"]
mus = np.asarray(b["class_mu"], dtype=np.float64)
covs = np.asarray(b["class_cov"], dtype=np.float64)
K = len(phones)
inv = np.stack([np.linalg.inv(c) for c in covs])


def dm(x, k):
    d = x - mus[k]
    return np.sqrt(max(d @ inv[k] @ d, 0.0))


def tau_pair(i, j):
    """min_x max(dM_i(x), dM_j(x)) via SLSQP on the epigraph form."""
    # variable z = [x (12), t (1)]
    x0 = 0.5 * (mus[i] + mus[j])
    t0 = max(dm(x0, i), dm(x0, j))
    z0 = np.concatenate([x0, [t0]])

    def obj(z):
        return z[-1]

    def gobj(z):
        g = np.zeros_like(z)
        g[-1] = 1.0
        return g

    cons = []
    for k in (i, j):
        def c(z, k=k):
            d = z[:-1] - mus[k]
            return z[-1] - np.sqrt(max(d @ inv[k] @ d, 1e-300))

        def gc(z, k=k):
            d = z[:-1] - mus[k]
            q = np.sqrt(max(d @ inv[k] @ d, 1e-300))
            g = np.zeros_like(z)
            g[:-1] = -(inv[k] @ d) / q
            g[-1] = 1.0
            return g

        cons.append({"type": "ineq", "fun": c, "jac": gc})

    r = minimize(obj, z0, jac=gobj, constraints=cons, method="SLSQP",
                 options={"maxiter": 500, "ftol": 1e-10})
    x = r.x[:-1]
    # report the true value at the solution (guard against constraint slack)
    return float(max(dm(x, i), dm(x, j)))


# Full pairwise matrix
T = np.zeros((K, K))
pairs = []
for i, j in combinations(range(K), 2):
    t = tau_pair(i, j)
    T[i, j] = T[j, i] = t
    pairs.append({"i": phones[i], "j": phones[j], "tau": t,
                  "mu_dist": float(np.linalg.norm(mus[i] - mus[j]))})
np.fill_diagonal(T, np.inf)

pairs.sort(key=lambda p: p["tau"])
print("=== 15 smallest pairwise critical radii (hardest pairs) ===")
for p in pairs[:15]:
    print(f"  {p['i']:>3s}/{p['j']:<3s}  tau_ij={p['tau']:.4f}   ||mu_i-mu_j||={p['mu_dist']:.4f}")
print("\n=== 5 largest ===")
for p in pairs[-5:]:
    print(f"  {p['i']:>3s}/{p['j']:<3s}  tau_ij={p['tau']:.4f}   ||mu_i-mu_j||={p['mu_dist']:.4f}")

tau_geo_39 = pairs[0]["tau"]
print(f"\nGeometric prediction, all 39 classes: tau* <= {tau_geo_39:.4f}"
      f"  (limited by {pairs[0]['i']}/{pairs[0]['j']})")

# 4-class subset used in the April 22 experiment
sub4 = ["aa", "n", "sh", "sil"]
idx4 = [phones.index(p) for p in sub4]
t4 = min(T[i, j] for i, j in combinations(idx4, 2))
lim4 = min(((T[i, j], phones[i], phones[j]) for i, j in combinations(idx4, 2)))
print(f"Geometric prediction, 4-class (aa,n,sh,sil): tau* <= {t4:.4f}"
      f"  (limited by {lim4[1]}/{lim4[2]})")

# tau*_geo as a function of inventory size K: min over pairs in a random subset
rng = np.random.default_rng(0)
curve = []
for k in range(2, K + 1):
    vals = []
    n_draw = 400 if k < K else 1
    for _ in range(n_draw):
        s = rng.choice(K, size=k, replace=False) if k < K else np.arange(K)
        sub = T[np.ix_(s, s)]
        vals.append(float(sub.min()))
    curve.append({"K": k, "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                  "p10": float(np.percentile(vals, 10)), "p90": float(np.percentile(vals, 90)),
                  "min": float(np.min(vals))})
print("\n=== tau*_geo vs inventory size (mean over 400 random subsets) ===")
for c in curve:
    if c["K"] in (2, 4, 8, 12, 16, 24, 32, 39):
        print(f"  K={c['K']:>2d}   tau*_geo = {c['mean']:.3f} +/- {c['std']:.3f}"
              f"   [p10 {c['p10']:.3f}, p90 {c['p90']:.3f}]")

# correlation between tau_ij and mean distance
tv = np.array([p["tau"] for p in pairs])
dv = np.array([p["mu_dist"] for p in pairs])
print(f"\ncorr(tau_ij, ||mu_i-mu_j||) = {np.corrcoef(tv, dv)[0,1]:.4f}")

# Which phones are involved in the tightest pairs?
absorbed = ["ih", "ay", "z", "sh", "m", "oy", "jh", "th"]
rank = {}
for r, p in enumerate(pairs):
    for ph in (p["i"], p["j"]):
        rank.setdefault(ph, r)
print("\n=== rank of each phone's tightest pair (0 = tightest in corpus) ===")
order = sorted(rank.items(), key=lambda kv: kv[1])
for ph, r in order:
    mark = "  <-- absorbed by DBSCAN" if ph in absorbed else ""
    print(f"  {ph:>3s}  best-pair rank {r:>3d}{mark}")

json.dump({"phones": phones, "T": T.tolist(), "pairs": pairs, "curve": curve,
           "tau_geo_39": tau_geo_39, "tau_geo_4": t4,
           "absorbed": absorbed,
           "phone_pair_rank": rank},
          open(OUT, "w"), indent=1)
print(f"\nsaved -> {OUT}")
