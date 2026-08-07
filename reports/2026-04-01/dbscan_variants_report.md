# Report: DBSCAN Variants for 5-Class TIMIT Phoneme Clustering

## 1. Objective

This notebook studies whether unsupervised clustering can separate a small set of TIMIT phoneme classes using frame-level 12D MFCC features. The five selected phones were:

- `s`
- `ih`
- `aa`
- `iy`
- `n`

`sil` was intentionally excluded because it dominates the corpus and would overwhelm density-based clustering.

## 2. Data and Experimental Setup

### 2.1 Dataset snapshot

- Source file: `train_dim12.json`
- Total utterances: `4,620`
- Total labeled frames across train set: `1,419,678`
- Unique phones in train set: `39`
- Example feature shape per utterance: `(349, 12)` = `349` frames, each with `12` MFCC values

### 2.2 Global phoneme frequency

The most frequent phone in the train set is `sil`, which occupies `367,150` frames (`25.86%`). This is the main reason it was removed from the 5-class study.

Top frame counts from the notebook:

| Phone | Frames | Percent |
|---|---:|---:|
| `sil` | 367,150 | 25.86% |
| `s` | 84,491 | 5.95% |
| `ih` | 84,270 | 5.94% |
| `aa` | 73,885 | 5.20% |
| `iy` | 62,901 | 4.43% |
| `n` | 46,996 | 3.31% |

### 2.3 5-class dataset used in the notebook

After filtering to the 5 target phones, the notebook builds a frame-level dataset with `352,543` frames.

| Phone | Frames | Percent |
|---|---:|---:|
| `s` | 84,491 | 23.97% |
| `ih` | 84,270 | 23.90% |
| `aa` | 73,885 | 20.96% |
| `iy` | 62,901 | 17.84% |
| `n` | 46,996 | 13.33% |

### 2.4 Subsampling and preprocessing

- Balanced subset used for sweeps: `10,000` frames
- Sampling strategy: stratified, `2,000` frames per class
- Feature preprocessing for 12D experiments: `StandardScaler`
- t-SNE was used only for visualization, not for clustering

### 2.5 Baseline DBSCAN distance scale

The notebook computed k-distance quantiles on the standardized 12D MFCC subset to guide `eps`.

| k | q50 | q75 | q90 | q95 | q99 |
|---|---:|---:|---:|---:|---:|
| 4 | 1.5774 | 1.8195 | 2.0511 | 2.2070 | 2.5433 |
| 6 | 1.6584 | 1.9085 | 2.1519 | 2.3148 | 2.6402 |
| 10 | 1.7647 | 2.0236 | 2.2764 | 2.4437 | 2.7815 |
| 16 | 1.8636 | 2.1346 | 2.4086 | 2.5737 | 2.9263 |

This is why the main DBSCAN sweep focused on `eps` roughly in the `1.6` to `2.4` range.

## 3. What We Tried and What We Got

### 3.1 Standard DBSCAN on 12D standardized MFCC

Sweep used in the notebook:

- `eps_grid = 1.6, 1.7, ..., 2.4`
- `min_samples = [5, 8, 10, 15, 20]`
- Total configs tried: `45`

Top notebook outputs from the sweep:

| min_samples | eps | n_clusters | coverage | NMI | AMI | ARI | homogeneity | completeness | v_measure | clustered_purity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.6 | 5 | 0.6326 | 0.1086 | 0.1077 | 0.0780 | 0.0106 | 0.3116 | 0.0206 | 0.3022 |
| 5 | 1.9 | 4 | 0.9239 | 0.0264 | 0.0255 | 0.0038 | 0.0013 | 0.1933 | 0.0025 | 0.2155 |
| 5 | 1.8 | 4 | 0.8729 | 0.0448 | 0.0439 | 0.0120 | 0.0013 | 0.1863 | 0.0025 | 0.2266 |
| 8 | 1.8 | 4 | 0.8355 | 0.0632 | 0.0624 | 0.0230 | 0.0014 | 0.1980 | 0.0028 | 0.2361 |
| 10 | 1.7 | 4 | 0.7194 | 0.1064 | 0.1057 | 0.0628 | 0.0109 | 0.3531 | 0.0212 | 0.2758 |
| 15 | 1.7 | 4 | 0.6445 | 0.1395 | 0.1389 | 0.0983 | 0.0142 | 0.4099 | 0.0275 | 0.3033 |
| 5 | 1.7 | 7 | 0.8005 | 0.0665 | 0.0650 | 0.0280 | 0.0037 | 0.1898 | 0.0073 | 0.2467 |
| 8 | 1.7 | 3 | 0.7469 | 0.0873 | 0.0867 | 0.0483 | 0.0020 | 0.2388 | 0.0039 | 0.2611 |

#### Best exact-5-cluster DBSCAN result

The notebook selected `eps = 1.6`, `min_samples = 8` as the best configuration that produced exactly 5 clusters.

Metrics:

| Metric | Value |
|---|---:|
| n_clusters | 5 |
| noise_fraction | 0.3674 |
| coverage | 0.6326 |
| NMI | 0.1086 |
| AMI | 0.1077 |
| ARI | 0.0780 |
| homogeneity | 0.0106 |
| completeness | 0.3116 |
| v_measure | 0.0206 |
| clustered_purity | 0.3022 |

Cluster composition:

| Cluster | Size | Majority phone | Purity | Top phones |
|---|---:|---|---:|---|
| 0 | 6,279 | `s` | 0.298 | `s(1869), iy(1416), ih(1383)` |
| 1 | 6 | `n` | 0.833 | `n(5), s(1)` |
| 2 | 26 | `aa` | 1.000 | `aa(26)` |
| 3 | 8 | `aa` | 0.625 | `aa(5), n(3)` |
| 4 | 7 | `aa` | 1.000 | `aa(7)` |
| noise | 3,674 | - | 1.000 | - |

Interpretation:

- The algorithm technically found 5 clusters, but almost all clustered points fell into one large mixed cluster.
- This is not a meaningful 5-way phoneme separation.
- The very low homogeneity (`0.0106`) confirms that the clusters do not align cleanly with the true phones.

### 3.2 Standard DBSCAN follow-up with smaller eps

The notebook then manually tried a smaller `eps = 1.4` while keeping `min_samples = 8`.

Metrics:

| Metric | Value |
|---|---:|
| n_clusters | 11 |
| noise_fraction | 0.6487 |
| coverage | 0.3513 |
| NMI | 0.1633 |
| AMI | 0.1614 |
| ARI | 0.1084 |
| homogeneity | 0.0318 |
| completeness | 0.2353 |
| v_measure | 0.0560 |
| clustered_purity | 0.4859 |

Cluster composition:

| Cluster | Size | Majority phone | Purity | Top phones |
|---|---:|---|---:|---|
| 0 | 3,422 | `s` | 0.476 | `s(1629), iy(726), ih(668)` |
| 1 | 6 | `iy` | 0.833 | `iy(5), ih(1)` |
| 2 | 4 | `n` | 0.750 | `n(3), ih(1)` |
| 3 | 28 | `iy` | 0.964 | `iy(27), ih(1)` |
| 4 | 8 | `iy` | 0.625 | `iy(5), ih(1), s(1)` |
| 5 | 8 | `aa` | 1.000 | `aa(8)` |
| 6 | 8 | `iy` | 0.875 | `iy(7), ih(1)` |
| 7 | 6 | `n` | 1.000 | `n(6)` |
| 8 | 7 | `n` | 0.857 | `n(6), ih(1)` |
| 9 | 9 | `n` | 0.556 | `n(5), ih(3), iy(1)` |
| 10 | 7 | `iy` | 0.857 | `iy(6), ih(1)` |
| noise | 6,487 | - | 1.000 | - |

Interpretation:

- Lowering `eps` improved NMI and purity, but only by fragmenting the data into many tiny clusters.
- Coverage dropped to `35.13%`, so most points became noise.
- This is a sensitivity check, not a good final solution.

### 3.3 HDBSCAN on 12D standardized MFCC

Sweep used in the notebook:

- `min_cluster_size = [30, 50, 100, 200, 300, 500, 750, 1000]`

Sweep results:

| min_cluster_size | n_clusters | coverage | NMI | AMI | ARI | clustered_purity | cluster_gap_to_5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 50 | 2 | 0.1656 | 0.2329 | 0.2326 | 0.0692 | 0.8472 | 3 |
| 100 | 2 | 0.1128 | 0.1919 | 0.1916 | 0.0459 | 0.9424 | 3 |
| 30 | 2 | 0.1029 | 0.1785 | 0.1781 | 0.0391 | 0.9563 | 3 |
| 200 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5 |
| 300 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5 |
| 500 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5 |
| 750 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5 |
| 1000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5 |

#### Best HDBSCAN result

Best notebook result: `min_cluster_size = 50`

| Metric | Value |
|---|---:|
| n_clusters | 2 |
| noise_fraction | 0.8344 |
| coverage | 0.1656 |
| NMI | 0.2329 |
| AMI | 0.2326 |
| ARI | 0.0692 |
| homogeneity | 0.6477 |
| completeness | 0.9495 |
| v_measure | 0.7701 |
| clustered_purity | 0.8472 |

Cluster composition:

| Cluster | Size | Majority phone | Purity | Top phones |
|---|---:|---|---:|---|
| 0 | 1,201 | `s` | 0.999 | `s(1200), ih(1)` |
| 1 | 455 | `iy` | 0.446 | `iy(203), ih(177), n(67)` |
| noise | 8,344 | - | 1.000 | - |

Interpretation:

- HDBSCAN was more honest than DBSCAN: it did not force 5 clusters.
- It mainly identified one extremely clean `s` cluster and one mixed vowel/nasal cluster.
- Among density-based methods on 12D MFCC, HDBSCAN gave the best NMI, but it covered only `16.56%` of the data.

### 3.4 Context-window DBSCAN: 132D context features, PCA + whitening

The notebook then added temporal context:

- Left context = `5` frames
- Right context = `5` frames
- Total context dimension = `132`
- Valid context frames: `9,999 / 10,000`

PCA settings used in the final context section:

- `30` PCs after whitening
- Variance retained: `95.0%`

Context-feature k-distance quantiles:

| k | q50 | q75 | q90 | q95 | q99 |
|---|---:|---:|---:|---:|---:|
| 4 | 4.3134 | 4.7531 | 5.1707 | 5.4382 | 5.9755 |
| 6 | 4.4465 | 4.8903 | 5.3092 | 5.5740 | 6.1128 |
| 10 | 4.6039 | 5.0574 | 5.4736 | 5.7636 | 6.3070 |

Sweep used in the notebook:

- `eps = 2.4, 2.5, ..., 4.0`
- `min_samples = [5, 8, 9, 10, 12, 14, 15]`
- Total configs tried: `119`

Representative sweep outputs printed by the notebook:

| eps | min_samples | n_clusters | NMI | coverage |
|---|---:|---:|---:|---:|
| 2.70 | 5 | 23 | 0.196 | 0.379 |
| 3.10 | 5 | 20 | 0.074 | 0.737 |
| 3.50 | 5 | 1 | 0.020 | 0.929 |
| 2.60 | 8 | 9 | 0.212 | 0.227 |
| 3.00 | 8 | 3 | 0.104 | 0.579 |
| 2.50 | 9 | 6 | 0.218 | 0.158 |
| 2.90 | 9 | 5 | 0.205 | 0.466 |
| 2.40 | 10 | 2 | 0.215 | 0.118 |
| 2.70 | 12 | 3 | 0.218 | 0.242 |
| 2.60 | 14 | 2 | 0.220 | 0.171 |
| 2.50 | 15 | 2 | 0.233 | 0.130 |

#### Best context-DBSCAN result

Best notebook result: `eps = 2.4`, `min_samples = 8`

| Metric | Value |
|---|---:|
| n_clusters | 5 |
| noise_fraction | 0.8726 |
| coverage | 0.1274 |
| NMI | 0.2250 |
| AMI | 0.2241 |
| ARI | 0.0642 |
| homogeneity | 0.7346 |
| completeness | 0.8127 |
| v_measure | 0.7717 |
| clustered_purity | 0.9780 |

Cluster composition:

| Cluster | Size | Majority phone | Purity | Top phones |
|---|---:|---|---:|---|
| 0 | 1,179 | `s` | 0.992 | `s(1169), ih(4), n(3)` |
| 1 | 74 | `iy` | 0.784 | `iy(58), ih(14), n(2)` |
| 2 | 5 | `iy` | 0.800 | `iy(4), ih(1)` |
| 3 | 8 | `ih` | 1.000 | `ih(8)` |
| 4 | 8 | `ih` | 0.875 | `ih(7), n(1)` |
| noise | 8,726 | - | 1.000 | - |

Interpretation:

- Adding temporal context produced very pure clusters.
- The cost was extremely high noise: only `12.74%` of frames were clustered.
- So context helps precision much more than recall.

### 3.5 OPTICS on 12D standardized MFCC

Notebook setup:

- `min_samples = 10`
- `xi = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]`

The run was interrupted by `KeyboardInterrupt`, but the notebook printed partial results:

| xi | n_clusters | NMI | coverage | gap_to_5 |
|---|---:|---:|---:|---:|
| 0.03 | 3 | 0.010 | 0.005 | 2 |
| 0.05 | 1 | 0.000 | 1.000 | 4 |
| 0.08 | 1 | 0.000 | 1.000 | 4 |
| 0.10 | 1 | 0.000 | 1.000 | 4 |
| 0.15 | 1 | 0.000 | 1.000 | 4 |

After that failure, the notebook explicitly recorded OPTICS as failed and replaced its labels with all-noise.

Recorded final OPTICS result in the notebook:

| Metric | Value |
|---|---:|
| n_clusters | 0 |
| noise_fraction | 1.0000 |
| coverage | 0.0000 |
| NMI | 0.0000 |
| AMI | 0.0000 |
| ARI | 0.0000 |
| homogeneity | 0.0000 |
| completeness | 0.0000 |
| v_measure | 0.0000 |
| clustered_purity | 0.0000 |

Interpretation:

- OPTICS did not reveal clean valleys in reachability space.
- The notebook conclusion is that the density transitions are too flat for xi-extraction to work well on these features.

### 3.6 GMM baseline on 12D standardized MFCC

Although the notebook is centered on DBSCAN variants, it also included a GMM baseline.

#### Model-selection numbers

| Components | BIC | AIC |
|---|---:|---:|
| 2 | 293,954 | 292,648 |
| 3 | 285,481 | 283,520 |
| 4 | 281,885 | 279,268 |
| 5 | 279,131 | 275,858 |
| 6 | 278,193 | 274,263 |
| 7 | 277,351 | 272,765 |
| 8 | 276,479 | 271,238 |
| 9 | 276,338 | 270,440 |
| 10 | 275,984 | 269,429 |
| 11 | 275,669 | 268,459 |

The notebook notes that BIC keeps improving up to `11` components, which already suggests that the 5 phones are not well modeled as 5 simple spherical or density-separated groups.

#### GMM with 5 components

| Metric | Value |
|---|---:|
| n_clusters | 5 |
| noise_fraction | 0.0000 |
| coverage | 1.0000 |
| NMI | 0.5546 |
| AMI | 0.5544 |
| ARI | 0.5247 |
| homogeneity | 0.5531 |
| completeness | 0.5562 |
| v_measure | 0.5546 |
| clustered_purity | 0.7096 |

Cluster composition:

| Cluster | Size | Majority phone | Purity | Top phones |
|---|---:|---|---:|---|
| 0 | 1,580 | `iy` | 0.453 | `iy(715), ih(593), n(220)` |
| 1 | 2,311 | `iy` | 0.496 | `iy(1146), ih(926), n(150)` |
| 2 | 2,255 | `n` | 0.687 | `n(1550), ih(410), aa(149)` |
| 3 | 1,951 | `s` | 0.974 | `s(1901), n(24), ih(21)` |
| 4 | 1,903 | `aa` | 0.937 | `aa(1784), n(56), ih(50)` |
| noise | 0 | - | 1.000 | - |

Interpretation:

- GMM gave the best global agreement metrics by a large margin.
- `s` and `aa` are modeled very cleanly.
- `ih` and `iy` still overlap heavily even in GMM.
- This suggests the MFCC geometry is better captured by elliptical probabilistic components than by a single global density threshold.

## 4. Overall Comparison

### 4.1 Side-by-side metrics

| Method | Main params | n_clusters | coverage | noise | NMI | AMI | ARI | homogeneity | completeness | v_measure | clustered_purity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DBSCAN | `eps=1.6, ms=8` | 5 | 0.6326 | 0.3674 | 0.1086 | 0.1077 | 0.0780 | 0.0106 | 0.3116 | 0.0206 | 0.3022 |
| DBSCAN follow-up | `eps=1.4, ms=8` | 11 | 0.3513 | 0.6487 | 0.1633 | 0.1614 | 0.1084 | 0.0318 | 0.2353 | 0.0560 | 0.4859 |
| HDBSCAN | `mcs=50` | 2 | 0.1656 | 0.8344 | 0.2329 | 0.2326 | 0.0692 | 0.6477 | 0.9495 | 0.7701 | 0.8472 |
| DBSCAN-132D | `eps=2.4, ms=8` | 5 | 0.1274 | 0.8726 | 0.2250 | 0.2241 | 0.0642 | 0.7346 | 0.8127 | 0.7717 | 0.9780 |
| OPTICS | `min_samples=10, xi sweep failed` | 0 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| GMM | `5 full-cov components` | 5 | 1.0000 | 0.0000 | 0.5546 | 0.5544 | 0.5247 | 0.5531 | 0.5562 | 0.5546 | 0.7096 |

### 4.2 Main conclusions from the notebook

1. Single-frame 12D MFCC is not sufficient for clean density-based separation of these 5 phones.
2. `s` is consistently the easiest phone to cluster. It forms the cleanest cluster in every method.
3. `ih` and `iy` are the hardest pair. They overlap in almost every method.
4. Standard DBSCAN has a basic trade-off:
   - larger `eps`: better coverage, but it merges phones into one big mixed cluster
   - smaller `eps`: better purity, but most points become noise
5. HDBSCAN is more realistic than DBSCAN on this data. It prefers finding only the dense cores instead of forcing 5 clusters.
6. Adding context helps purity a lot, but still does not give broad coverage.
7. GMM fits the feature geometry much better than density-based clustering on this task.

## 5. Practical Takeaway

If the goal is to show what density-based clustering can and cannot do on frame-level MFCC:

- Standard DBSCAN does not recover the 5 phones meaningfully.
- HDBSCAN finds only a few stable dense regions, mainly a very pure `s` cluster.
- Context-window DBSCAN gives very pure clusters, but only for a small minority of frames.
- The best overall clustering quality in the notebook comes from GMM, not from DBSCAN-family methods.

So the main story is not "we found the right DBSCAN settings"; it is "the feature space itself limits what density-based clustering can recover."

## 6. Summary to Send to Prof

We ran a focused 5-phone clustering study on TIMIT using frame-level 12D MFCCs, restricting the classes to `s`, `ih`, `aa`, `iy`, and `n` and removing `sil` because it dominates the frame distribution. On a balanced 10k-frame subset, standard DBSCAN could be tuned to return 5 clusters (`eps=1.6`, `min_samples=8`), but the result was not meaningful: one large mixed cluster absorbed most clustered points, and the alignment with true phones was weak (`NMI=0.1086`, `clustered purity=0.3022`). Making `eps` smaller improved purity somewhat but caused most points to become noise.

We then tried HDBSCAN, OPTICS, and a context-window variant of DBSCAN. HDBSCAN was more conservative and produced only 2 stable clusters, with better `NMI=0.2329` but only `16.56%` coverage; it mainly isolated a very pure `s` cluster. Context-window DBSCAN (132D features with PCA whitening) also improved cluster purity substantially (`0.9780`) and reached `NMI=0.2250`, but coverage dropped further to `12.74%`, so it mostly retained only high-confidence cores. OPTICS did not produce usable structure on this feature space. For comparison, a 5-component full-covariance GMM gave the strongest overall result (`NMI=0.5546`, `ARI=0.5247`, 100% coverage), suggesting that these MFCC features are better modeled by overlapping elliptical distributions than by a single density threshold. Overall, the main conclusion is that DBSCAN-style clustering is strongly feature-limited on single-frame MFCCs; context helps precision, but not enough to recover all 5 phones cleanly.
