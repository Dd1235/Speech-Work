"""
E2: Empirical tau sweep on MCMC cores, reusing the saved per-class Gaussians.
Finer tau grid than the original April 29 run, to locate tau* precisely and to
compare against the geometric certificate tau_geo from E1.

Also runs the 4-class subset (aa, n, sh, sil) on the same protocol so the two
inventory sizes are directly comparable.
"""
import os
import json
import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             homogeneity_score, completeness_score)

BASE = os.path.join(os.path.dirname(__file__), "results")
PKL = os.path.join(os.path.dirname(__file__), "..", "artifacts", "support_set_40phone.pkl")
SEED = 0
N_PER = 250          # MCMC samples per class (smaller than the original 400-500; keeps the sweep fast)
BURN, THIN = 500, 3

with open(PKL, "rb") as f:
    b = pickle.load(f)
phones_all = b["phonemes"]
mus_all = np.asarray(b["class_mu"], dtype=np.float64)
covs_all = np.asarray(b["class_cov"], dtype=np.float64)


def mh_core(mu, cov, n, tau, scale, seed):
    rng = np.random.default_rng(seed)
    d = len(mu)
    ci = np.linalg.inv(cov)
    L = np.linalg.cholesky(cov * scale ** 2)
    t2 = tau ** 2

    def m2(x):
        v = x - mu
        return float(v @ ci @ v)

    x = mu.copy()
    out = np.empty((n, d))
    filled = step = acc = tot = 0
    while filled < n:
        xp = x + L @ rng.standard_normal(d)
        mp = m2(xp)
        tot += 1
        if mp <= t2:
            if np.log(rng.random()) < -0.5 * (mp - m2(x)):
                x, acc = xp, acc + 1
        step += 1
        if step > BURN and (step - BURN) % THIN == 0:
            out[filled] = x
            filled += 1
    return out, acc / max(tot, 1)


def build(idx, tau, scale):
    X, y, accs = [], [], []
    for k in idx:
        s, a = mh_core(mus_all[k], covs_all[k], N_PER, tau, scale, SEED + k)
        X.append(s)
        y.append(np.full(N_PER, phones_all[k], dtype=object))
        accs.append(a)
    return np.concatenate(X), np.concatenate(y), float(np.mean(accs))


def sweep(X, y, target, eps_grid, ms_grid=(5, 10)):
    rows = []
    for ms in ms_grid:
        for eps in eps_grid:
            lab = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1).fit_predict(X)
            nc = len(set(lab)) - (1 if -1 in lab else 0)
            rows.append(dict(ms=int(ms), eps=float(eps), clusters=int(nc),
                             gap=int(abs(nc - target)),
                             noise=float((lab == -1).mean()),
                             nmi=float(normalized_mutual_info_score(y, lab)),
                             ari=float(adjusted_rand_score(y, lab)),
                             hom=float(homogeneity_score(y, lab)),
                             comp=float(completeness_score(y, lab))))
    ok = [r for r in rows if r["noise"] < 0.4]
    if not ok:
        ok = rows
    best = max(ok, key=lambda r: (r["nmi"], -r["gap"]))
    return best, rows


def run(idx, label, tau_grid, eps_grid):
    target = len(idx)
    res = []
    print(f"\n===== {label}  (K={target}) =====")
    print(f"{'tau':>6} {'acc':>6} {'eps':>5} {'ms':>3} {'clus':>5} {'noise':>7} "
          f"{'NMI':>7} {'ARI':>7} {'hom':>6} {'comp':>6}")
    for tau in tau_grid:
        scale = 0.25 if tau >= 1.2 else 0.15
        X, y, acc = build(idx, tau, scale)
        best, _ = sweep(X, y, target, eps_grid)
        res.append(dict(tau=float(tau), accept=acc, **best))
        print(f"{tau:6.2f} {acc:6.3f} {best['eps']:5.2f} {best['ms']:3d} "
              f"{best['clusters']:5d} {best['noise']:7.4f} {best['nmi']:7.4f} "
              f"{best['ari']:7.4f} {best['hom']:6.3f} {best['comp']:6.3f}")
    return res


# ---- 39 classes ----
idx39 = list(range(len(phones_all)))
tau39 = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00, 1.25, 1.50]
eps39 = [round(v, 2) for v in np.arange(0.10, 1.05, 0.05)]
r39 = run(idx39, "39 phone classes", tau39, eps39)

# ---- 4 classes (April 22 subset) ----
sub4 = ["aa", "n", "sh", "sil"]
idx4 = [phones_all.index(p) for p in sub4]
tau4 = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50]
eps4 = [round(v, 2) for v in np.arange(0.10, 2.05, 0.10)]
r4 = run(idx4, "4 classes: aa, n, sh, sil", tau4, eps4)

json.dump({"r39": r39, "r4": r4, "n_per_class": N_PER},
          open(f"{BASE}/e2_results.json", "w"), indent=1)
print(f"\nsaved -> {BASE}/e2_results.json")
