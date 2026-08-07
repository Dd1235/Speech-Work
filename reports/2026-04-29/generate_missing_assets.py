"""Generate the 5 missing report figures directly, without needing to re-run notebook cells.

Reuses the per-class Gaussians stored in support_set_40phone.pkl.
Runs MCMC + DBSCAN to reproduce the plots, saves them to assets/.

Run from the report_apr29/ directory:
    python generate_missing_assets.py
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import json
import pickle
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score,
    homogeneity_score, completeness_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist, squareform

plt.style.use("ggplot")

ROOT = Path(__file__).parent.parent
ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 0
FRAMES_PER_CLASS = 1500
SAMPLES_PER_COMPONENT = 500   # for tau=1.5 (matches notebook)
SAMPLES_PER_COMPONENT_SWEEP = 400
TSNE_SIZE = 6000
MH_BURN_IN = 500
MH_THIN = 3


# ---------- helpers ----------

def expand_frame_labels(utt):
    T = len(utt["features"])
    fl = np.full(T, "unlabeled", dtype=object)
    for i, (lab, s, e) in enumerate(utt["phn_transcript"]):
        s = max(0, min(int(s), T)); e = max(s, min(int(e), T))
        if i == len(utt["phn_transcript"]) - 1 and e == T - 1: e = T
        if e > s: fl[s:e] = lab
    return fl


def mh_sample_core(mu, cov, n_samples, threshold_sigma,
                   burn_in=500, thin=3, proposal_scale=0.25, seed=0):
    rng = np.random.default_rng(seed)
    d = len(mu)
    cov_inv = np.linalg.inv(cov)
    chol_prop = np.linalg.cholesky(cov * (proposal_scale ** 2))
    threshold_sq = threshold_sigma ** 2
    def mahal_sq(x):
        diff = x - mu; return float(diff @ cov_inv @ diff)
    x = mu.copy()
    samples = np.empty((n_samples, d))
    filled, step, accepts, total = 0, 0, 0, 0
    while filled < n_samples:
        x_prop = x + chol_prop @ rng.standard_normal(d)
        m_prop = mahal_sq(x_prop); total += 1
        if m_prop <= threshold_sq:
            m_curr = mahal_sq(x)
            if np.log(rng.random()) < -0.5 * (m_prop - m_curr):
                x = x_prop; accepts += 1
        step += 1
        if step > burn_in and ((step - burn_in) % thin == 0):
            samples[filled] = x; filled += 1
    return samples, accepts / max(total, 1)


def dbscan_sweep(X, y, eps_grid, min_samples_grid, target_clusters):
    rows = []
    for ms in min_samples_grid:
        for eps in eps_grid:
            labels = DBSCAN(eps=eps, min_samples=ms,
                            algorithm="ball_tree", n_jobs=-1).fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            rows.append({
                "min_samples": int(ms), "eps": float(eps),
                "clusters": int(n_clusters),
                "cluster_gap": int(abs(n_clusters - target_clusters)),
                "noise_fraction": float((labels == -1).mean()),
                "nmi": float(normalized_mutual_info_score(y, labels)),
                "ari": float(adjusted_rand_score(y, labels)),
                "homogeneity": float(homogeneity_score(y, labels)),
                "completeness": float(completeness_score(y, labels)),
            })
    return pd.DataFrame(rows)


# ---------- load + rebuild standardized real frames ----------

print("loading train_dim12.json ...")
t0 = time.time()
with open(ROOT / "train_dim12.json", "r") as f:
    data = json.load(f)
print(f"  done in {time.time()-t0:.1f}s, {len(data)} utts")

print("loading support_set_40phone.pkl ...")
with open(ROOT / "support_set_40phone.pkl", "rb") as f:
    bundle = pickle.load(f)
phones = bundle["phonemes"]
mus = np.asarray(bundle["class_mu"])
covs = np.asarray(bundle["class_cov"])
print(f"  39 phones, mu={mus.shape}, cov={covs.shape}")

# rebuild real-scaled subset for the t-SNE that overlays support frames (we already have that one)
# we mainly need it for the cluster_majority computation; using the cached scaler from the pickle
print("rebuilding stratified frame subset ...")
sel = set(phones)
X_full, y_full = [], []
for utt in data.values():
    feats = np.asarray(utt["features"], dtype=np.float32)
    fl = expand_frame_labels(utt)
    m = np.isin(fl, list(sel))
    X_full.append(feats[m]); y_full.append(fl[m])
X_full = np.concatenate(X_full); y_full = np.concatenate(y_full)
rng = np.random.default_rng(RANDOM_SEED)
parts = []
for p in phones:
    idx = np.where(y_full == p)[0]
    parts.append(rng.choice(idx, size=min(FRAMES_PER_CLASS, len(idx)), replace=False))
sub = np.concatenate(parts); rng.shuffle(sub)
X_real = X_full[sub]; y_real = y_full[sub]
# reuse the saved scaler so geometry matches the pickle exactly
scaler_mean = np.asarray(bundle["scaler_mean"])
scaler_scale = np.asarray(bundle["scaler_scale"])
X_real_scaled = (X_real - scaler_mean) / scaler_scale
print(f"  X_real_scaled = {X_real_scaled.shape}")


# ---------- 1) pairwise mean distance heatmap (cheapest) ----------

print("\n[1/5] pairwise_mean_distance.png ...")
D = squareform(pdist(mus, metric="euclidean"))
phones_arr = np.array(phones)
Z = linkage(pdist(mus, metric="euclidean"), method="average")
leaf_order = leaves_list(Z)
D_sorted = D[np.ix_(leaf_order, leaf_order)]
phones_sorted = phones_arr[leaf_order]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(D_sorted, cmap="magma_r", aspect="auto")
ax.set_xticks(range(len(phones_sorted))); ax.set_xticklabels(phones_sorted, fontsize=8, rotation=90)
ax.set_yticks(range(len(phones_sorted))); ax.set_yticklabels(phones_sorted, fontsize=8)
ax.set_title("Pairwise Euclidean distance between class means (standardized 12D MFCC)")
plt.colorbar(im, ax=ax, label=r"$\|\mu_i - \mu_j\|$")
plt.tight_layout()
plt.savefig(ASSETS / "pairwise_mean_distance.png", dpi=120)
plt.close()
print("  saved")


# ---------- 2) MCMC at tau=1.5, DBSCAN, t-SNE -> dbscan_clusters_tau15_tsne.png ----------

print("\n[2/5] dbscan_clusters_tau15_tsne.png (sampling tau=1.5 cores) ...")
t0 = time.time()
Xc15_parts, yc15_parts = [], []
for k, p in enumerate(phones):
    s, _ = mh_sample_core(mus[k], covs[k], SAMPLES_PER_COMPONENT, 1.5,
                          burn_in=MH_BURN_IN, thin=MH_THIN,
                          proposal_scale=0.25, seed=RANDOM_SEED + k)
    Xc15_parts.append(s); yc15_parts.append(np.full(SAMPLES_PER_COMPONENT, p, dtype=object))
X_core15 = np.concatenate(Xc15_parts); y_core15 = np.concatenate(yc15_parts)
print(f"  cores shape {X_core15.shape} in {time.time()-t0:.1f}s")

# Use the original (notebook) ranker so the picture shows the actual failure mode reported.
sweep15 = dbscan_sweep(
    X_core15, y_core15,
    eps_grid=[round(v, 2) for v in np.arange(0.3, 1.61, 0.1)],
    min_samples_grid=[5, 8, 10, 15],
    target_clusters=39,
)
ranked15 = sweep15.sort_values(
    ["cluster_gap", "nmi", "noise_fraction"],
    ascending=[True, False, True],
).reset_index(drop=True)
best15 = ranked15.iloc[0]
print(f"  best15: eps={best15.eps} ms={best15.min_samples} "
      f"clusters={best15.clusters} noise={best15.noise_fraction:.3f} "
      f"NMI={best15.nmi:.3f}")
labels15 = DBSCAN(eps=float(best15.eps), min_samples=int(best15.min_samples),
                  algorithm="ball_tree", n_jobs=-1).fit_predict(X_core15)

vis_n = min(TSNE_SIZE, len(X_core15))
vis_i = np.random.default_rng(RANDOM_SEED).choice(len(X_core15), size=vis_n, replace=False)
emb15 = TSNE(n_components=2, perplexity=30, init="pca",
             learning_rate="auto", random_state=RANDOM_SEED).fit_transform(X_core15[vis_i])
true_codes = pd.Categorical(y_core15[vis_i], categories=phones).codes
cluster_codes = labels15[vis_i]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].scatter(emb15[:, 0], emb15[:, 1], c=true_codes, cmap="tab20", s=10, alpha=0.75)
axes[0].set_title(r"MCMC core ($\tau=1.5\sigma$): t-SNE colored by true phone")
axes[0].set_xlabel("dim 1"); axes[0].set_ylabel("dim 2")
nm = cluster_codes == -1
axes[1].scatter(emb15[nm, 0], emb15[nm, 1], c="lightgray", s=10, alpha=0.5, label="noise (-1)")
axes[1].scatter(emb15[~nm, 0], emb15[~nm, 1], c=cluster_codes[~nm], cmap="tab20", s=10, alpha=0.85)
axes[1].set_title(f"DBSCAN clusters @ eps={best15.eps}, ms={int(best15.min_samples)}  "
                  f"(NMI={normalized_mutual_info_score(y_core15, labels15):.3f}, "
                  f"noise={(labels15==-1).mean():.0%})")
axes[1].set_xlabel("dim 1"); axes[1].set_ylabel("dim 2")
axes[1].legend(loc="best")
plt.tight_layout()
plt.savefig(ASSETS / "dbscan_clusters_tau15_tsne.png", dpi=120)
plt.close()
print("  saved")


# ---------- 3) MCMC at tau=0.5 + DBSCAN ----------

print("\n[3/5] sampling tau=0.5 cores ...")
t0 = time.time()
Xc05_parts, yc05_parts = [], []
for k, p in enumerate(phones):
    s, _ = mh_sample_core(mus[k], covs[k], SAMPLES_PER_COMPONENT_SWEEP, 0.5,
                          burn_in=MH_BURN_IN, thin=MH_THIN,
                          proposal_scale=0.15, seed=RANDOM_SEED + k)
    Xc05_parts.append(s); yc05_parts.append(np.full(SAMPLES_PER_COMPONENT_SWEEP, p, dtype=object))
X_core05 = np.concatenate(Xc05_parts); y_core05 = np.concatenate(yc05_parts)
print(f"  cores shape {X_core05.shape} in {time.time()-t0:.1f}s")

sweep05 = dbscan_sweep(
    X_core05, y_core05,
    eps_grid=[round(v, 2) for v in np.arange(0.2, 1.21, 0.1)],
    min_samples_grid=[5, 8, 10, 15],
    target_clusters=39,
)
sweep05_filt = sweep05[sweep05["noise_fraction"] < 0.4].sort_values(
    ["nmi", "cluster_gap"], ascending=[False, True],
).reset_index(drop=True)
best05 = sweep05_filt.iloc[0]
labels05 = DBSCAN(eps=float(best05.eps), min_samples=int(best05.min_samples),
                  algorithm="ball_tree", n_jobs=-1).fit_predict(X_core05)
print(f"  best05: eps={best05.eps} ms={best05.min_samples} "
      f"clusters={best05.clusters} noise={best05.noise_fraction:.3f} "
      f"NMI={best05.nmi:.3f} ARI={best05.ari:.3f}")


# ---------- 4) confusion_matrix.png ----------

print("\n[4/5] confusion_matrix.png ...")
ct = pd.crosstab(pd.Series(y_core05, name="phone"),
                 pd.Series(labels05, name="cluster"))
phone_order = list(phones)
ordered_cols = []
for p in phone_order:
    cands = [c for c in ct.columns if ct[c].idxmax() == p and c not in ordered_cols]
    cands.sort(key=lambda c: -ct.loc[p, c])
    for c in cands: ordered_cols.append(c)
for c in ct.columns:
    if c not in ordered_cols: ordered_cols.append(c)
ct_sorted = ct.loc[phone_order, ordered_cols]
ct_norm = ct_sorted.div(ct_sorted.sum(axis=1).replace(0, 1), axis=0)

fig, ax = plt.subplots(figsize=(13, 10))
im = ax.imshow(ct_norm.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(ordered_cols)))
ax.set_xticklabels([str(c) for c in ordered_cols], fontsize=8, rotation=90)
ax.set_yticks(range(len(phone_order)))
ax.set_yticklabels(phone_order, fontsize=9)
ax.set_xlabel("DBSCAN cluster id (reordered)")
ax.set_ylabel("true phone")
ax.set_title(f"Cluster x Phone confusion (row-normalized)  "
             rf"$\tau=0.5$, $\varepsilon$={best05.eps}, ms={int(best05.min_samples)}, "
             f"NMI={normalized_mutual_info_score(y_core05, labels05):.3f}")
plt.colorbar(im, ax=ax, label="fraction of phone in cluster")
plt.tight_layout()
plt.savefig(ASSETS / "confusion_matrix.png", dpi=120)
plt.close()
print("  saved")


# ---------- 5) labeled_cluster_tsne.png ----------

print("\n[5/5] labeled_cluster_tsne.png ...")
vis_n = min(TSNE_SIZE, len(X_core05))
vis_i = np.random.default_rng(RANDOM_SEED).choice(len(X_core05), size=vis_n, replace=False)
emb05 = TSNE(n_components=2, perplexity=30, init="pca",
             learning_rate="auto", random_state=RANDOM_SEED).fit_transform(X_core05[vis_i])
true_codes = pd.Categorical(y_core05[vis_i], categories=phones).codes
cluster_codes = labels05[vis_i]

cluster_majority = {}
for cid in sorted(set(labels05.tolist())):
    if cid == -1: continue
    mask = labels05 == cid
    cluster_majority[cid] = pd.Series(y_core05[mask]).value_counts().idxmax()

fig, axes = plt.subplots(1, 2, figsize=(17, 8))
axes[0].scatter(emb05[:, 0], emb05[:, 1], c=true_codes, cmap="tab20", s=10, alpha=0.7)
for k, p in enumerate(phones):
    pmask = pd.Categorical(y_core05[vis_i], categories=phones).codes == k
    if pmask.sum() == 0: continue
    cx, cy = emb05[pmask, 0].mean(), emb05[pmask, 1].mean()
    axes[0].annotate(p, (cx, cy), fontsize=9, fontweight="bold",
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.18", fc="white", alpha=0.7, lw=0))
axes[0].set_title(r"MCMC core ($\tau=0.5\sigma$): t-SNE labeled by true phone")
axes[0].set_xlabel("dim 1"); axes[0].set_ylabel("dim 2")

nm = cluster_codes == -1
axes[1].scatter(emb05[nm, 0], emb05[nm, 1], c="lightgray", s=10, alpha=0.5, label="noise")
axes[1].scatter(emb05[~nm, 0], emb05[~nm, 1], c=cluster_codes[~nm], cmap="tab20", s=10, alpha=0.85)
for cid, phone in cluster_majority.items():
    cmask = cluster_codes == cid
    if cmask.sum() == 0: continue
    cx, cy = emb05[cmask, 0].mean(), emb05[cmask, 1].mean()
    axes[1].annotate(phone, (cx, cy), fontsize=9, fontweight="bold",
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.18", fc="white", alpha=0.85, lw=0))
axes[1].set_title(f"DBSCAN clusters labeled by majority phone  "
                  f"(NMI={normalized_mutual_info_score(y_core05, labels05):.3f}, "
                  f"noise={(labels05==-1).mean():.0%})")
axes[1].set_xlabel("dim 1"); axes[1].set_ylabel("dim 2")
axes[1].legend(loc="best")
plt.tight_layout()
plt.savefig(ASSETS / "labeled_cluster_tsne.png", dpi=120)
plt.close()
print("  saved")


# ---------- 6) tau_progression.png ----------

print("\n[6/6] tau_progression.png (running tau=0.75 and tau=1.0) ...")
def run_tau(tau, scale=0.15):
    Xc_parts, yc_parts = [], []
    for k, p in enumerate(phones):
        s, _ = mh_sample_core(mus[k], covs[k], SAMPLES_PER_COMPONENT_SWEEP, tau,
                              burn_in=MH_BURN_IN, thin=MH_THIN,
                              proposal_scale=scale, seed=RANDOM_SEED + k)
        Xc_parts.append(s); yc_parts.append(np.full(SAMPLES_PER_COMPONENT_SWEEP, p, dtype=object))
    Xc = np.concatenate(Xc_parts); yc = np.concatenate(yc_parts)
    sw = dbscan_sweep(Xc, yc,
                      eps_grid=[round(v, 2) for v in np.arange(0.2, 1.21, 0.1)],
                      min_samples_grid=[5, 8, 10, 15],
                      target_clusters=39)
    sw_f = sw[sw["noise_fraction"] < 0.4].sort_values(["nmi", "cluster_gap"],
                                                       ascending=[False, True])
    if len(sw_f) == 0:
        sw_f = sw.sort_values(["nmi", "cluster_gap"], ascending=[False, True])
    top = sw_f.iloc[0]
    lab = DBSCAN(eps=float(top.eps), min_samples=int(top.min_samples),
                 algorithm="ball_tree", n_jobs=-1).fit_predict(Xc)
    return {
        "tau": tau,
        "eps": float(top.eps), "min_samples": int(top.min_samples),
        "clusters": int(len(set(lab)) - (1 if -1 in lab else 0)),
        "noise": float((lab == -1).mean()),
        "nmi": float(normalized_mutual_info_score(yc, lab)),
        "ari": float(adjusted_rand_score(yc, lab)),
    }

# We already ran tau=0.5 (best05) and tau=1.5 (best15). Just run 0.75 and 1.0.
prog = []
prog.append({
    "tau": 1.5,
    "eps": float(best15.eps), "min_samples": int(best15.min_samples),
    "clusters": int(len(set(labels15)) - (1 if -1 in labels15 else 0)),
    "noise": float((labels15 == -1).mean()),
    "nmi": float(normalized_mutual_info_score(y_core15, labels15)),
    "ari": float(adjusted_rand_score(y_core15, labels15)),
})
print("  tau=1.0 ..."); prog.append(run_tau(1.0, scale=0.15))
print("  tau=0.75 ..."); prog.append(run_tau(0.75, scale=0.15))
prog.append({
    "tau": 0.5,
    "eps": float(best05.eps), "min_samples": int(best05.min_samples),
    "clusters": int(len(set(labels05)) - (1 if -1 in labels05 else 0)),
    "noise": float((labels05 == -1).mean()),
    "nmi": float(normalized_mutual_info_score(y_core05, labels05)),
    "ari": float(adjusted_rand_score(y_core05, labels05)),
})
prog_df = pd.DataFrame(prog).sort_values("tau").reset_index(drop=True)
print(prog_df)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(prog_df["tau"], prog_df["nmi"], "o-", label="NMI", linewidth=2)
axes[0].plot(prog_df["tau"], prog_df["ari"], "s-", label="ARI", linewidth=2)
axes[0].set_xlabel(r"MCMC core threshold $\tau$ ($\sigma$)")
axes[0].set_ylabel("clustering quality")
axes[0].set_title("DBSCAN agreement with truth vs core tightness")
axes[0].invert_xaxis()
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(prog_df["tau"], prog_df["clusters"], "o-", color="tab:purple", label="clusters")
axes[1].axhline(39, color="black", linestyle="--", alpha=0.4, label="target = 39")
ax2 = axes[1].twinx()
ax2.plot(prog_df["tau"], prog_df["noise"], "s-", color="tab:red", label="noise frac")
ax2.set_ylabel("noise fraction", color="tab:red")
ax2.set_ylim(0, 1)
axes[1].set_xlabel(r"MCMC core threshold $\tau$ ($\sigma$)")
axes[1].set_ylabel("# clusters", color="tab:purple")
axes[1].set_title(r"Cluster count and noise vs $\tau$")
axes[1].invert_xaxis()
axes[1].legend(loc="upper left"); ax2.legend(loc="upper right")
plt.tight_layout()
plt.savefig(ASSETS / "tau_progression.png", dpi=120)
plt.close()
print("  saved")

print("\nall 5 missing assets generated.")
