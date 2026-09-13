"""Property tests for lot-keyed partitioning.

These exercise pure functions over generated *inputs*. No generated wafer data is
stored anywhere; the arrays and names here exist only to probe the function under
test, which is exactly what the no-fabricated-data rule permits and what fixed
example tests could not do as thoroughly.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from yieldloop.db.enums import SplitName
from yieldloop.ingest.partition import (
    assign_split,
    assign_splits,
    lot_ordinals,
    lot_position,
    split_counts,
)

lot_names = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1, max_size=32
)
seeds = st.integers(min_value=0, max_value=2**31 - 1)


@given(name=lot_names, seed=seeds)
def test_position_is_in_unit_interval(name: str, seed: int) -> None:
    assert 0.0 <= lot_position(name, seed) < 1.0


@given(name=lot_names, seed=seeds)
def test_assignment_is_deterministic(name: str, seed: int) -> None:
    first = assign_split(name, seed, 0.70, 0.15)
    second = assign_split(name, seed, 0.70, 0.15)
    assert first is second


@given(names=st.lists(lot_names, min_size=1, max_size=200, unique=True), seed=seeds)
def test_adding_lots_never_moves_an_existing_lot(names: list[str], seed: int) -> None:
    """Stability under dataset growth is what lets an old model be evaluated on a
    holdout it provably never saw."""
    before = assign_splits(names, seed, 0.70, 0.15)
    after = assign_splits([*names, "lot-newly-arrived"], seed, 0.70, 0.15)
    for name in names:
        assert after[name] is before[name]


@given(names=st.lists(lot_names, min_size=1, max_size=200, unique=True), seed=seeds)
def test_every_lot_receives_exactly_one_split(names: list[str], seed: int) -> None:
    assignments = assign_splits(names, seed, 0.70, 0.15)
    assert set(assignments) == set(names)
    assert sum(split_counts(assignments).values()) == len(names)


@given(names=st.lists(lot_names, min_size=1, max_size=300, unique=True))
def test_ordinals_are_a_contiguous_permutation(names: list[str]) -> None:
    ordinals = lot_ordinals(names)
    assert sorted(ordinals.values()) == list(range(len(names)))


@given(names=st.lists(lot_names, min_size=1, max_size=300, unique=True))
def test_ordinals_follow_byte_order_not_locale(names: list[str]) -> None:
    ordinals = lot_ordinals(names)
    ordered = [name for name, _ in sorted(ordinals.items(), key=lambda kv: kv[1])]
    assert ordered == sorted(names, key=lambda n: n.encode("utf-8"))


@hyp_settings(max_examples=25)
@given(seed=seeds)
def test_split_proportions_track_the_configured_fractions(seed: int) -> None:
    """Over many lots the hash should land close to the requested fractions."""
    names = [f"lot{i:06d}" for i in range(8000)]
    counts = split_counts(assign_splits(names, seed, 0.70, 0.15))
    total = sum(counts.values())
    assert counts[SplitName.TRAIN] / total == pytest.approx(0.70, abs=0.03)
    assert counts[SplitName.VAL] / total == pytest.approx(0.15, abs=0.03)
    assert counts[SplitName.HOLDOUT] / total == pytest.approx(0.15, abs=0.03)


@pytest.mark.parametrize(
    ("train", "val"),
    [(0.9, 0.1), (0.95, 0.06), (0.0, 0.5), (1.0, 0.1), (0.5, 0.0)],
)
def test_fractions_leaving_no_holdout_are_rejected(train: float, val: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        assign_split("lot00001", 1, train, val)
