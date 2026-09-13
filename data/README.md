# data/

Holds the real WM811K archive and derived artifacts. **Nothing in here is
committed** except this file.

| Path | What | How it gets here |
| --- | --- | --- |
| `LSWMD.pkl` | The real WM811K archive, 811,457 wafers | `python scripts/fetch_dataset.py` |
| `registry/` | Content-addressed model artifacts | Written by training |
| `registry/wafer.index` | FAISS index and its manifest | `python scripts/seed_from_real_labels.py` |

## The dataset

WM811K (MIR-WM811K) — real wafer maps from a production fab, with human defect
pattern labels on 21.3% of them. Obtained from Kaggle; the dataset page carries
its own licence and terms, which you accept there.

The archive is verified by sha256 on every load. If it does not match
`YIELDLOOP_WM811K_SHA256`, loading fails rather than warning: metrics computed
from a different file are not comparable to the committed baselines.

## Why there is no fallback

`scripts/fetch_dataset.py` has no generated-data path. Without credentials it
prints setup instructions and exits non-zero.

A fallback is how a repository ends up publishing metrics computed against
something that is not the dataset. Every number in the eval report is meant to be
traceable to real wafer maps and real human labels, and that guarantee only holds
if there is no code path that quietly substitutes anything else.

See [../docs/data_contract.md](../docs/data_contract.md) for the full field
mapping and every derivation rule.
