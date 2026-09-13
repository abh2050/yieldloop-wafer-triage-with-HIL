"""Deterministic, lot-keyed train/validation/holdout partitioning.

Rule ``PARTITION_ASSIGN`` in ``docs/data_contract.md``.

Partitioning is keyed on the lot, never the wafer. Wafers from one lot share
process history and are strongly correlated; splitting at the wafer level would
put near-duplicates on both sides of the boundary and inflate every metric in
the eval report. The dataset's own ``trainTestLabel`` is deliberately unused for
the same reason: it does not respect lot boundaries.

The assignment is a hash of the lot name, not a shuffle, which means it is
stable under dataset growth. Adding lots never moves an existing lot to a
different split, so a model trained last month can still be evaluated against a
holdout it has provably never seen.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

from yieldloop.db.enums import SplitName

#: Width of the hash prefix used to build the unit interval position.
_HASH_BYTES = 8
_HASH_SPACE = float(1 << (_HASH_BYTES * 8))


def lot_position(lot_name: str, seed: int) -> float:
    """Map a lot name onto ``[0, 1)`` deterministically.

    Uses blake2b rather than :func:`hash`, which is randomized per process and
    would make partitions irreproducible across runs.
    """
    payload = f"{seed}:{lot_name}".encode()
    digest = hashlib.blake2b(payload, digest_size=_HASH_BYTES).digest()
    return int.from_bytes(digest, byteorder="big") / _HASH_SPACE


def assign_split(lot_name: str, seed: int, train_fraction: float, val_fraction: float) -> SplitName:
    """Assign one lot to a split.

    Raises:
        ValueError: if the fractions leave no room for a holdout split.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1); got {train_fraction}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            "train_fraction + val_fraction must be < 1.0 to leave a holdout split; got "
            f"{train_fraction} + {val_fraction}"
        )

    position = lot_position(lot_name, seed)
    if position < train_fraction:
        return SplitName.TRAIN
    if position < train_fraction + val_fraction:
        return SplitName.VAL
    return SplitName.HOLDOUT


def assign_splits(
    lot_names: Iterable[str], seed: int, train_fraction: float, val_fraction: float
) -> dict[str, SplitName]:
    """Assign every lot in ``lot_names`` to a split."""
    return {
        name: assign_split(name, seed, train_fraction, val_fraction)
        for name in dict.fromkeys(lot_names)
    }


def split_counts(assignments: Mapping[str, SplitName]) -> dict[SplitName, int]:
    """Count lots per split. Every split is present, zero included."""
    counts = dict.fromkeys(SplitName, 0)
    for split in assignments.values():
        counts[split] += 1
    return counts


def lot_ordinals(lot_names: Iterable[str]) -> dict[str, int]:
    """Assign each lot a 0-based ordinal under a stable byte ordering.

    Rule ``LOT_ORDINAL``. WM811K carries no timestamps, so the sorted lot name is
    the only total order the dataset admits. Sorting is on the UTF-8 encoding
    rather than the string, so the result does not depend on locale.
    """
    unique = sorted(set(lot_names), key=lambda name: name.encode("utf-8"))
    return {name: index for index, name in enumerate(unique)}
