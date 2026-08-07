"""E4: figures for the paper + the geometry-to-accuracy validation."""
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

BASE = os.path.join(os.path.dirname(__file__), "results")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "assets")
import os
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 160, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .3, "axes.axisbelow": True})

e1 = json.load(open(f"{BASE}/e1_results.json"))
e2 = json.load(open(f"{BASE}/e2_results.json"))
e3 = json.load(open(f"{BASE}/e3_results.json"))
phones = e1["phones"]
T = np.array(e1["T"])
np.fill_diagonal(T, np.inf)
absorbed = set(e1["absorbed"])

# ---------------------------------------------------------------- FIG 1
# NMI vs tau with acceptance rate; degenerate regime shaded; certificate marked
r39 = pd.DataFrame(e2["r39"]); r4 = pd.DataFrame(e2["r4"])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, r, lab, tgt, tgeo in [
        (axes[0], r39, "39 phone classes", 39, e1["tau_geo_39"]),
        (axes[1], r4, "4 classes (aa, n, sh, sil)", 4, e1["tau_geo_4"])]:
    ax.plot(r.tau, r.nmi, "o-", color="tab:blue", lw=2, label="NMI", zorder=3)
    ax.plot(r.tau, r.ari, "s-", color="tab:cyan", lw=1.5, label="ARI", zorder=3)
    ax.plot(r.tau, r.accept, "^--", color="tab:red", lw=1.5,
            label="MH acceptance", zorder=3)
    deg = r.tau[r.accept < 0.02]
    if len(deg):
        ax.axvspan(0, deg.max() * 1.05, color="grey", alpha=.22, lw=0, zorder=0)
        ax.text(deg.max() * .52, .45, "chain frozen\n(degenerate)", ha="center",
                fontsize=7.5, color="0.25")
    ax.axvline(tgeo, color="tab:green", ls=":", lw=2, zorder=2)
    ax.text(tgeo, 1.03, r"$\tau_{\rm geo}$=" + f"{tgeo:.2f}", color="tab:green",
            ha="center", fontsize=8)
    ax.set_xlabel(r"core radius $\tau$ ($\sigma$)")
    ax.set_title(lab, fontsize=9.5)
    ax.set_ylim(-.03, 1.1)
    ax.set_xlim(0, r.tau.max() * 1.03)
axes[0].set_ylabel("score")
axes[0].legend(fontsize=7.5, loc="center right")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_tau_regimes.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- FIG 2
curve = pd.DataFrame(e1["curve"])
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.fill_between(curve.K, curve.p10, curve.p90, alpha=.22, color="tab:blue",
                lw=0, label="10-90th pct over random subsets")
ax.plot(curve.K, curve["mean"], "o-", color="tab:blue", ms=3.5, lw=1.8,
        label=r"mean $\tau^*_{\rm geo}$")
ax.plot(curve.K, curve["min"], "-", color="tab:red", lw=1.2, alpha=.8,
        label="worst-case subset")
ax.scatter([4], [e1["tau_geo_4"]], marker="*", s=180, color="tab:orange",
           zorder=5, label="aa/n/sh/sil (chosen subset)")
ax.scatter([39], [e1["tau_geo_39"]], marker="D", s=45, color="black", zorder=5,
           label="full inventory")
ax.set_xlabel("number of phone classes $K$")
ax.set_ylabel(r"critical core radius $\tau^*_{\rm geo}$ ($\sigma$)")
ax.set_title("Separability collapses as the inventory grows", fontsize=9.5)
ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_tau_vs_K.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- FIG 3
pairs = e1["pairs"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9),
                         gridspec_kw={"width_ratios": [1.15, 1]})
top = pairs[:16][::-1]
lbl = [f"{p['i']}/{p['j']}" for p in top]
val = [p["tau"] for p in top]
col = ["tab:red" if (p["i"] in absorbed or p["j"] in absorbed) else "tab:blue"
       for p in top]
axes[0].barh(range(len(top)), val, color=col, alpha=.85)
axes[0].set_yticks(range(len(top))); axes[0].set_yticklabels(lbl, fontsize=7.5)
axes[0].axvline(e1["tau_geo_39"], color="k", ls=":", lw=1)
axes[0].set_xlabel(r"pairwise critical radius $\tau_{ij}$ ($\sigma$)")
axes[0].set_title("16 least-separable phone pairs\n"
                  "(red = contains a phone DBSCAN absorbed)", fontsize=8.5)

tv = np.array([p["tau"] for p in pairs]); dv = np.array([p["mu_dist"] for p in pairs])
axes[1].scatter(dv, tv, s=7, alpha=.4, color="tab:blue", lw=0)
inv_ = [p for p in pairs if p["i"] in absorbed or p["j"] in absorbed][:20]
axes[1].scatter([p["mu_dist"] for p in inv_], [p["tau"] for p in inv_],
                s=22, color="tab:red", lw=0, label="tightest absorbed-phone pairs")
axes[1].set_xlabel(r"$\|\mu_i-\mu_j\|$ (standardized 12-D)")
axes[1].set_ylabel(r"$\tau_{ij}$")
axes[1].set_title(f"Certificate vs mean distance\n"
                  f"Pearson r = {pearsonr(dv, tv)[0]:.3f}", fontsize=8.5)
axes[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_pairs.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- FIG 4
per = pd.DataFrame(e3["per_phone"])
tau_min = {p: float(np.min(T[i])) for i, p in enumerate(phones)}
per["tau_min"] = per.phone.map(tau_min)
rho, pv = spearmanr(per.tau_min, per.acc_qda)
pr, ppv = pearsonr(per.tau_min, per.acc_qda)
fig, ax = plt.subplots(figsize=(5.4, 3.9))
c = ["tab:red" if p in absorbed else "tab:blue" for p in per.phone]
ax.scatter(per.tau_min, per.acc_qda, c=c, s=26, zorder=3)
for _, r in per.iterrows():
    ax.annotate(r.phone, (r.tau_min, r.acc_qda), fontsize=6.5,
                xytext=(2.5, 2.5), textcoords="offset points")
z = np.polyfit(per.tau_min, per.acc_qda, 1)
xs = np.linspace(per.tau_min.min(), per.tau_min.max(), 20)
ax.plot(xs, np.polyval(z, xs), "-", color="0.4", lw=1.2, zorder=2)
ax.set_xlabel(r"$\min_j \tau_{ij}$  (geometric separability of the phone)")
ax.set_ylabel("held-out accuracy (QDA)")
ax.set_title(f"Geometry predicts held-out accuracy\n"
             fr"Spearman $\rho$={rho:.3f} (p={pv:.1e}),  Pearson r={pr:.3f}",
             fontsize=9)
ax.scatter([], [], c="tab:red", label="absorbed by DBSCAN", s=26)
ax.legend(fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_geometry_vs_accuracy.png", bbox_inches="tight"); plt.close()
print(f"geometry->accuracy: spearman={rho:.4f} p={pv:.3e}  pearson={pr:.4f} p={ppv:.3e}")

# ---------------------------------------------------------------- FIG 5
ks = pd.DataFrame(e3["kshot"])
old = {(5, 1): .8073, (5, 5): .9073, (10, 1): .6930, (10, 5): .8163,
       (20, 1): .5468, (20, 5): .7052}
fig, ax = plt.subplots(figsize=(5.4, 3.7))
cols = {5: "tab:blue", 10: "tab:orange", 20: "tab:green", 39: "tab:red"}
for n in sorted(ks.n_way.unique()):
    s = ks[ks.n_way == n]
    ax.errorbar(s.k_shot, s["mean"], yerr=s["std"], marker="o", ms=4, lw=1.8,
                capsize=2, color=cols[n], label=f"{n}-way (held-out queries)")
    pts = [(k, v) for (nn, k), v in old.items() if nn == n]
    if pts:
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "--", marker="x",
                ms=5, lw=1, color=cols[n], alpha=.65)
ax.plot([], [], "--", marker="x", color="0.4",
        label="same-pool queries (inflated)")
ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10]); ax.set_xticklabels([1, 2, 5, 10])
ax.set_xlabel("shots $K$"); ax.set_ylabel("episode accuracy")
ax.set_title("Few-shot baselines: honest vs leaked evaluation", fontsize=9.5)
ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_kshot.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- FIG 6
yh = np.load(f"{BASE}/e3_true.npy", allow_pickle=True)
pq = np.load(f"{BASE}/e3_pred_qda.npy", allow_pickle=True)
idx = {p: i for i, p in enumerate(phones)}
cm = np.zeros((len(phones), len(phones)))
for t, p in zip(yh, pq):
    cm[idx[t], idx[p]] += 1
cm = cm / cm.sum(1, keepdims=True).clip(min=1)
Tf = T.copy(); np.fill_diagonal(Tf, 0)
Z = linkage(squareform(Tf, checks=False), method="average")
lo = leaves_list(Z)
fig, ax = plt.subplots(figsize=(6.6, 5.6))
im = ax.imshow(cm[np.ix_(lo, lo)], cmap="viridis", vmin=0, vmax=.75)
ax.set_xticks(range(len(phones))); ax.set_xticklabels(np.array(phones)[lo], fontsize=6, rotation=90)
ax.set_yticks(range(len(phones))); ax.set_yticklabels(np.array(phones)[lo], fontsize=6)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(f"Held-out confusion, QDA (acc={e3['overall']['gaussian_loglik_qda']:.3f}), "
             r"ordered by $\tau_{ij}$", fontsize=9)
ax.grid(False)
plt.colorbar(im, ax=ax, label="fraction of true class", shrink=.85)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_heldout_confusion.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- stats dump
absorbed_ranks = sorted(e1["phone_pair_rank"][p] for p in absorbed)
stats = {
    "geometry_vs_accuracy": {"spearman": float(rho), "spearman_p": float(pv),
                             "pearson": float(pr), "pearson_p": float(ppv)},
    "absorbed_pair_ranks": absorbed_ranks,
    "n_pairs": len(pairs),
    "corr_tau_mudist": float(pearsonr(dv, tv)[0]),
    "tau_geo_39": e1["tau_geo_39"], "tau_geo_4": e1["tau_geo_4"],
    "heldout": e3["overall"],
    "absorbed_mean_acc": float(per[per.phone.isin(absorbed)].acc_qda.mean()),
    "other_mean_acc": float(per[~per.phone.isin(absorbed)].acc_qda.mean()),
}
json.dump(stats, open(f"{BASE}/e4_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
print("\nfigures ->", OUT)
print(sorted(os.listdir(OUT)))
