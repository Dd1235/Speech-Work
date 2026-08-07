"""E5: test the certificate at the level it is actually defined -- pairs.

Does tau_ij predict how often the held-out classifier confuses i with j?
Also tests aggregate 'crowding' statistics as per-phone predictors, since
min_j tau_ij alone was a weak per-phone predictor.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

BASE = os.path.join(os.path.dirname(__file__), "results")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "assets")
plt.rcParams.update({"figure.dpi": 160, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .3, "axes.axisbelow": True})

e1 = json.load(open(f"{BASE}/e1_results.json"))
e3 = json.load(open(f"{BASE}/e3_results.json"))
phones = e1["phones"]; K = len(phones)
T = np.array(e1["T"]); np.fill_diagonal(T, np.inf)
absorbed = set(e1["absorbed"])
idx = {p: i for i, p in enumerate(phones)}

yh = np.load(f"{BASE}/e3_true.npy", allow_pickle=True)
pq = np.load(f"{BASE}/e3_pred_qda.npy", allow_pickle=True)
C = np.zeros((K, K))
for t, p in zip(yh, pq):
    C[idx[t], idx[p]] += 1
n_per = C.sum(1)
R = C / n_per[:, None]                     # row-normalized confusion
S = (R + R.T) / 2                          # symmetric confusion rate
np.fill_diagonal(S, 0)

iu = np.triu_indices(K, 1)
tau_v = T[iu]; conf_v = S[iu]
rho, pv = spearmanr(tau_v, conf_v)
print(f"PAIR LEVEL  tau_ij vs symmetric confusion: spearman={rho:.4f} p={pv:.3e}")
mask = conf_v > 0
print(f"  (restricted to {mask.sum()} pairs ever confused): "
      f"spearman={spearmanr(tau_v[mask], conf_v[mask])[0]:.4f}")

# top confused pairs vs their rank in the certificate
order_conf = np.argsort(-conf_v)
tau_rank = {tuple(sorted(p)): r for r, p in enumerate(
    sorted([(phones[i], phones[j]) for i, j in zip(*iu)],
           key=lambda ij: T[idx[ij[0]], idx[ij[1]]]))}
print("\n=== 15 most-confused held-out pairs, and their certificate rank ===")
rows = []
for k in order_conf[:15]:
    i, j = iu[0][k], iu[1][k]
    key = tuple(sorted((phones[i], phones[j])))
    rows.append(dict(pair=f"{phones[i]}/{phones[j]}", confusion=conf_v[k],
                     tau=T[i, j], tau_rank=tau_rank[key]))
    print(f"  {rows[-1]['pair']:>8s}  conf={conf_v[k]:.3f}  "
          f"tau_ij={T[i,j]:.3f}  (rank {tau_rank[key]}/{len(tau_v)})")

# how many of the 20 tightest pairs are in the top-20 most confused?
tight20 = set(np.argsort(tau_v)[:20].tolist())
conf20 = set(order_conf[:20].tolist())
print(f"\noverlap |tightest-20 ∩ most-confused-20| = {len(tight20 & conf20)}")

# ---- per-phone: aggregate crowding statistics ----
per = pd.DataFrame(e3["per_phone"])
acc = per.set_index("phone").acc_qda
cands = {}
cands["min_tau"] = {p: float(T[idx[p]].min()) for p in phones}
for k in (3, 5, 10):
    cands[f"mean_{k}_smallest_tau"] = {
        p: float(np.sort(T[idx[p]])[:k].mean()) for p in phones}
cands["crowding_sum_inv_tau"] = {
    p: float(-np.sum(1.0 / T[idx[p]][np.isfinite(T[idx[p]])])) for p in phones}
cands["crowding_sum_exp"] = {
    p: float(-np.sum(np.exp(-T[idx[p]][np.isfinite(T[idx[p]])]))) for p in phones}
cands["n_pairs_below_0.6"] = {
    p: float(-(T[idx[p]] < 0.6).sum()) for p in phones}

print("\n=== per-phone predictors of held-out QDA accuracy ===")
best = None
for name, d in cands.items():
    x = np.array([d[p] for p in acc.index]); y = acc.values
    r, rp = spearmanr(x, y)
    print(f"  {name:<26s} spearman={r:+.3f}  p={rp:.3e}")
    if best is None or abs(r) > abs(best[1]):
        best = (name, r, rp, x, y)

print(f"\nbest per-phone predictor: {best[0]}  (spearman {best[1]:+.3f}, p={best[2]:.2e})")

# ---- figure: pair-level scatter + best per-phone predictor ----
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
ax = axes[0]
col = np.array(["tab:blue"] * len(tau_v), dtype=object)
for n, (i, j) in enumerate(zip(*iu)):
    if phones[i] in absorbed or phones[j] in absorbed:
        col[n] = "tab:red"
ax.scatter(tau_v, conf_v, s=9, c=list(col), alpha=.5, lw=0)
for k in order_conf[:8]:
    i, j = iu[0][k], iu[1][k]
    ax.annotate(f"{phones[i]}/{phones[j]}", (T[i, j], conf_v[k]), fontsize=6.5,
                xytext=(3, 2), textcoords="offset points")
ax.set_xlabel(r"pairwise critical radius $\tau_{ij}$")
ax.set_ylabel("symmetric held-out confusion rate")
ax.set_yscale("symlog", linthresh=1e-3)
ax.set_title(f"Certificate predicts confusion\n"
             fr"Spearman $\rho$ = {rho:.3f} (p = {pv:.1e}), n = {len(tau_v)} pairs",
             fontsize=9)

ax = axes[1]
name, r, rp, x, y = best
xs = -x if name.startswith(("crowding", "n_pairs")) else x
ax.scatter(xs, y, s=26, c=["tab:red" if p in absorbed else "tab:blue"
                           for p in acc.index], zorder=3)
for p, xx, yy in zip(acc.index, xs, y):
    ax.annotate(p, (xx, yy), fontsize=6.5, xytext=(2.5, 2.5),
                textcoords="offset points")
z = np.polyfit(xs, y, 1)
gx = np.linspace(xs.min(), xs.max(), 20)
ax.plot(gx, np.polyval(z, gx), color="0.4", lw=1.2, zorder=2)
pretty = {"mean_3_smallest_tau": r"crowding: mean of 3 smallest $\tau_{ij}$",
          "mean_5_smallest_tau": r"crowding: mean of 5 smallest $\tau_{ij}$",
          "mean_10_smallest_tau": r"crowding: mean of 10 smallest $\tau_{ij}$",
          "crowding_sum_inv_tau": r"crowding  $\sum_j 1/\tau_{ij}$",
          "crowding_sum_exp": r"crowding  $\sum_j e^{-\tau_{ij}}$",
          "n_pairs_below_0.6": r"# neighbours with $\tau_{ij}<0.6$",
          "min_tau": r"$\min_j \tau_{ij}$"}.get(name, name)
ax.set_xlabel(pretty); ax.set_ylabel("held-out accuracy (QDA)")
ax.set_title(f"Per-phone: crowding, not nearest neighbour\n"
             fr"Spearman $\rho$ = {r:+.3f} (p = {rp:.1e})", fontsize=9)
ax.scatter([], [], c="tab:red", s=26, label="absorbed by DBSCAN")
ax.legend(fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_certificate_validation.png", bbox_inches="tight")
plt.close()

json.dump({"pair_spearman": float(rho), "pair_p": float(pv),
           "n_pairs": int(len(tau_v)),
           "overlap_top20": int(len(tight20 & conf20)),
           "top_confused": rows,
           "per_phone_predictors": {k: [float(v) for v in spearmanr(
               [d[p] for p in acc.index], acc.values)]
               for k, d in cands.items()},
           "best_predictor": best[0]},
          open(f"{BASE}/e5_results.json", "w"), indent=1)
print("\nsaved fig_certificate_validation.png")
