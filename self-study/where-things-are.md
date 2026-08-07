# Where things are

For future me, after a gap.

## This repo

`~/Speech-Work` — deliberately **outside** OneDrive. Git inside a syncing folder corrupts `.git`.

## Original working folder

`OneDrive-iiit-b/sem6_asr_pe/` — the messy original, 2.2 GB, still intact. Nothing was deleted.

Contains everything here plus:
- the corpora (`train.json`, `train.pkl`, `train_dim12.json`, `dev*`, `test.json`)
- `submission.zip`, `report_apr*/report_dump.md` (generated dumps, skipped here)
- notebooks under their original names — mapping below

| here | original |
|---|---|
| `notebooks/00_starter_clustering` | `clustering.ipynb` |
| `notebooks/01_eda` | `eda_train_dim12.ipynb` |
| `notebooks/02–05` | `dbscan_0`, `dbscan`, `dbscan_train_dim12`, `dbscan_frame_level_dim12` |
| `notebooks/06–09` | `dbscan_5phone_classes_dim12`, `include sil`, `dbscan_apr1`, `dbscan_variants_5phone` |
| `notebooks/10–12` | `dbscan_4phone_classes_dim12`, `dbscan_4phone_mcmc_gmm_dim12`, `dbscan_40phone_mcmc_gmm_dim12` |
| `notebooks/13–14` | `use_support_set_40phone`, `submission/inspect_support_set` |

## Data

Not in git (~2.1 GB). See [`../data/README.md`](../data/README.md).

Originally pulled from the lab server over rsync — **host, user and path are in `resources/server_notes.md`, local only, never committed.** That password was sitting in plaintext in a synced folder; rotate it if it still works.

To run scripts against data kept elsewhere:

```bash
SPEECH_DATA=/path/to/data python analysis/03_heldout_eval.py
```

`analysis/01_geometric_certificate.py` needs **only** `artifacts/support_set_40phone.pkl` — no corpus. Good starting point after a long gap.

## Not in git

`resources/` — both proposal versions, the Indic TTS survey, the DBSCAN presentation, server notes. Local only; the GitHub repo is public.

## Picking it back up

1. `python analysis/01_geometric_certificate.py` — runs in a minute off the pickle alone, confirms the environment works
2. Read `paper/paper.pdf` §7–8 — the certificate and its validation, which is where the ideas are
3. `self-study/README.md` for what's next

## Known issues to not re-trip on

- The "39 clusters at τ=0.5" figure from April is **grid-sensitive** — a finer ε grid gives 52. Don't quote the cluster count; the monotone τ trend is what's robust.
- Old episode accuracies drew queries from the 390-prototype pool. Inflated ~0.2 absolute. Held-out numbers are in `paper/paper.pdf` Table 8.
- `train_dim12.json` load once timed out under OneDrive hydration — not a code bug. If it hangs, let the file materialise first.
