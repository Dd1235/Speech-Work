# Data

Not tracked in git (~2.1 GB). TIMIT-derived dumps, 12-dim MFCC per frame.

| file | size | contents |
|---|---|---|
| `train_dim12.json` | 334 MB | 4,620 utterances, `features` + `phn_transcript` |
| `train.json` / `train.pkl` | 1.0 GB / 552 MB | full-dimension originals |
| `dev.json` / `dev_12dim.pkl` | 116 MB / 63 MB | dev split |
| `test.json` | 55 MB | test split |
| `segmentation_results_12dim_{train,dev}.json` | | frame→phone segment boundaries |

Point scripts at a different location with `SPEECH_DATA=/path/to/data`.

Source: lab server, `/mnt/disk1/` (credentials not stored here).

Original working folder (everything, unpruned): `OneDrive-iiit-b/sem6_asr_pe/`.
See [`../self-study/where-things-are.md`](../self-study/where-things-are.md).
