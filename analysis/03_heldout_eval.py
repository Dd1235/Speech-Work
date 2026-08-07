"""
E3: The held-out evaluation that never ran last semester.

Loads the real corpus, removes the 390 prototype frames, and scores four
no-learning classifiers plus proper N-way K-shot episodes with HELD-OUT queries
(the original notebook drew queries from the prototype pool itself, which is a
leak in spirit and inflates the numbers).
"""
import os
import json
import pickle
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "results")
ROOT = os.environ.get("SPEECH_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))
SEED = 0
N_HELD = 300      # held-out frames per phone

with open(f"{ROOT}/submission/support_set_40phone.pkl", "rb") as f:
    b = pickle.load(f)
phones = b["phonemes"]
K_STORE = b["K"]
mus = np.asarray(b["class_mu"], dtype=np.float64)
covs = np.asarray(b["class_cov"], dtype=np.float64)
sc_m, sc_s = b["scaler_mean"], b["scaler_scale"]
Kn = len(phones)

excluded = set()
for p in phones:
    md = b["support_set"][p]["meta"]
    for uid, fi in zip(md["utt_id"], md["frame_idx"]):
        excluded.add((uid, int(fi)))
print("prototype frames excluded:", len(excluded))

print("loading corpus ...", flush=True)
with open(f"{ROOT}/train_dim12.json") as f:
    data = json.load(f)
print("utterances:", len(data), flush=True)


def frame_labels(utt):
    T = len(utt["features"])
    fl = np.full(T, "unlabeled", dtype=object)
    tr = utt["phn_transcript"]
    for i, (lab, s, e) in enumerate(tr):
        s = max(0, min(int(s), T))
        e = max(s, min(int(e), T))
        if i == len(tr) - 1 and e == T - 1:
            e = T
        if e > s:
            fl[s:e] = lab
    return fl


sel = set(phones)
pool = {p: [] for p in phones}
for uid, utt in data.items():
    feats = np.asarray(utt["features"], dtype=np.float32)
    fl = frame_labels(utt)
    for fi in range(len(feats)):
        ph = fl[fi]
        if ph in sel and (uid, fi) not in excluded:
            pool[ph].append(feats[fi])
del data
print("pool built. sizes (first 5):", {p: len(pool[p]) for p in phones[:5]}, flush=True)

rng = np.random.default_rng(SEED + 1)
Xh, yh = [], []
for p in phones:
    arr = np.asarray(pool[p], dtype=np.float32)
    take = min(N_HELD, len(arr))
    pick = rng.choice(len(arr), size=take, replace=False)
    Xh.append(arr[pick])
    yh.append(np.full(take, p, dtype=object))
Xh = np.concatenate(Xh)
yh = np.concatenate(yh)
del pool
Xs = (Xh - sc_m) / sc_s
print("held-out matrix:", Xs.shape, flush=True)

# ---------------- classifiers ----------------
proto = np.stack([b["support_set"][p]["X"].mean(axis=0) for p in phones]).astype(np.float64)
inv = np.stack([np.linalg.inv(c) for c in covs])
logdet = np.array([np.linalg.slogdet(c)[1] for c in covs])
w = np.asarray(b["class_weight"], dtype=np.float64)
ph_arr = np.array(phones)

res = {}

d_proto = np.linalg.norm(Xs[:, None, :] - proto[None], axis=-1)
pred_proto = ph_arr[d_proto.argmin(1)]
res["nearest_prototype_mean_euclidean"] = float((pred_proto == yh).mean())

d_mu = np.linalg.norm(Xs[:, None, :] - mus[None], axis=-1)
pred_mu = ph_arr[d_mu.argmin(1)]
res["nearest_class_mean_euclidean"] = float((pred_mu == yh).mean())


def mahal(X, mu, ci):
    d = X - mu
    return np.einsum("ni,ij,nj->n", d, ci, d)


M = np.stack([mahal(Xs, mus[k], inv[k]) for k in range(Kn)], 1)
pred_mah = ph_arr[M.argmin(1)]
res["min_mahalanobis"] = float((pred_mah == yh).mean())

# full Gaussian log-likelihood (QDA), uniform priors -- classes are stratified
ll = -0.5 * (M + logdet[None, :])
pred_qda = ph_arr[ll.argmax(1)]
res["gaussian_loglik_qda"] = float((pred_qda == yh).mean())
res["chance"] = 1.0 / Kn

print("\n=== held-out top-1 accuracy (39-way, %d frames) ===" % len(yh))
for k, v in res.items():
    print(f"  {k:<36s} {v:.4f}")

# ---------------- per-phone ----------------
rows = []
for p in phones:
    m = yh == p
    rows.append(dict(phone=p, n=int(m.sum()),
                     acc_proto=float((pred_proto[m] == p).mean()),
                     acc_mahal=float((pred_mah[m] == p).mean()),
                     acc_qda=float((pred_qda[m] == p).mean())))
per = pd.DataFrame(rows).sort_values("acc_qda", ascending=False)
print("\n=== best / worst phones (QDA) ===")
print(per.head(8).to_string(index=False))
print(per.tail(8).to_string(index=False))

absorbed = ["ih", "ay", "z", "sh", "m", "oy", "jh", "th"]
print("\nmean QDA acc, 8 DBSCAN-absorbed phones:",
      round(float(per[per.phone.isin(absorbed)].acc_qda.mean()), 4))
print("mean QDA acc, other 31 phones      :",
      round(float(per[~per.phone.isin(absorbed)].acc_qda.mean()), 4))

# ---------------- N-way K-shot with HELD-OUT queries ----------------
def episodes(n_way, k_shot, n_ep=400, seed=0):
    r = np.random.default_rng(seed)
    idx_by_phone = {p: np.where(yh == p)[0] for p in phones}
    accs = []
    for _ in range(n_ep):
        cs = r.choice(Kn, size=n_way, replace=False)
        cent, qx, qy = [], [], []
        for lab, c in enumerate(cs):
            sup = b["support_set"][phones[c]]["X"][r.permutation(K_STORE)[:k_shot]]
            cent.append(sup.mean(0))
            qi = r.choice(idx_by_phone[phones[c]], size=1)
            qx.append(Xs[qi])
            qy.append(lab)
        cent = np.stack(cent)
        qx = np.concatenate(qx)
        pr = np.linalg.norm(qx[:, None, :] - cent[None], axis=-1).argmin(1)
        accs.append(float((pr == np.array(qy)).mean()))
    return float(np.mean(accs)), float(np.std(accs))


print("\n=== N-way K-shot, nearest centroid, HELD-OUT queries (400 episodes) ===")
kshot = []
for n_way in (5, 10, 20, 39):
    line = []
    for k_shot in (1, 2, 5, 10):
        m, s = episodes(n_way, k_shot, seed=n_way * 100 + k_shot)
        kshot.append(dict(n_way=n_way, k_shot=k_shot, mean=m, std=s))
        line.append(f"K={k_shot}: {m:.3f}+/-{s:.3f}")
    print(f"  N={n_way:>2d}  " + "   ".join(line))

json.dump({"overall": res, "per_phone": rows, "kshot": kshot,
           "n_heldout": int(len(yh))},
          open(f"{BASE}/e3_results.json", "w"), indent=1)
np.save(f"{BASE}/e3_pred_qda.npy", pred_qda)
np.save(f"{BASE}/e3_true.npy", yh)
print(f"\nsaved -> {BASE}/e3_results.json")
