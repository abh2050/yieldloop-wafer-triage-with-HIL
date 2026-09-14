# data/

This directory holds the real WM811K archive and the artifacts derived from it.
Git tracks nothing in here except this file.

| Path | What | How it gets here |
| --- | --- | --- |
| `LSWMD.pkl` | The real WM811K archive, 811,457 wafers | `python scripts/fetch_dataset.py` |
| `registry/` | Content-addressed model artifacts | Written by training |
| `registry/wafer.index` | FAISS index and its manifest | `python scripts/seed_from_real_labels.py` |

## The dataset

WM811K, also called MIR-WM811K, holds real wafer maps from a production fab.
Human annotators labeled the defect pattern on 21.3% of them. You obtain the
archive from Kaggle, and the dataset page carries its own licence and terms,
which you accept there.

The loader verifies the archive by sha256 on every load. If the digest does not
match `YIELDLOOP_WM811K_SHA256`, loading fails rather than warning, because
metrics computed from a different file do not compare to the committed
baselines.

## Why no fallback exists

`scripts/fetch_dataset.py` has no generated-data path. Without credentials it
prints setup instructions and exits non-zero.

A fallback is how a repository ends up publishing metrics computed against
something that is not the dataset. Every number in the eval report traces to real
wafer maps and real human labels, and that guarantee holds only while no code
path quietly substitutes anything else.

See [../docs/data_contract.md](../docs/data_contract.md) for the full field
mapping and every derivation rule.
