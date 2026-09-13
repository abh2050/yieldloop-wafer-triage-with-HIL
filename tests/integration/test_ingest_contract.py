"""The real WM811K archive must match the documented data contract.

These read the actual file. They are skipped only when it is absent -- never
satisfied by a generated substitute -- because their entire purpose is to detect
the archive differing from what ``docs/data_contract.md`` claims.

The most valuable assertion here is
``test_die_total_reproduces_the_datasets_own_die_size``: WM811K carries a
``dieSize`` field that is the die count for the wafer, which gives an independent
check on the counting rule. If normalization ever ran before counting, or the
alphabet were misread, that test breaks rather than every failure rate in the
system shifting quietly.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yieldloop.config import Settings
from yieldloop.db.enums import DefectPattern, SplitName
from yieldloop.ingest.loader import (
    REQUIRED_COLUMNS,
    SOURCE_SPLIT_COLUMN,
    iter_records,
    read_frame,
)
from yieldloop.ingest.normalize import (
    WAFER_MAP_ALPHABET,
    encode_grid,
    normalize_grid,
)
from yieldloop.ingest.partition import assign_splits, lot_ordinals, split_counts

pytestmark = pytest.mark.dataset

#: How many rows to walk for the row-level contract checks. The full file is
#: 811,457 wafers; a prefix keeps the suite usable in the fast loop while still
#: covering many lots and every common map shape.
SAMPLE_ROWS = 20_000

EXPECTED_ROWS = 811_457
EXPECTED_LOTS = 46_293


@pytest.fixture(scope="module")
def frame(wm811k_path: Path) -> pd.DataFrame:
    return read_frame(wm811k_path, Settings())


def test_archive_has_the_documented_columns(frame: pd.DataFrame) -> None:
    assert set(frame.columns) >= REQUIRED_COLUMNS


def test_train_test_column_is_the_misspelled_one(frame: pd.DataFrame) -> None:
    """The typo is in the published dataset.

    Pinned so that a future release fixing it surfaces here, rather than as a
    column that quietly starts arriving empty.
    """
    assert "trianTestLabel" == SOURCE_SPLIT_COLUMN  # noqa: SIM300
    assert SOURCE_SPLIT_COLUMN in frame.columns
    assert "trainTestLabel" not in frame.columns


def test_archive_size_matches_the_data_contract(frame: pd.DataFrame) -> None:
    assert len(frame) == EXPECTED_ROWS
    assert frame["lotName"].nunique() == EXPECTED_LOTS


def test_wafer_maps_use_only_the_documented_alphabet(frame: pd.DataFrame) -> None:
    """0 outside the wafer, 1 passing die, 2 failing die. Nothing else."""
    for wafer_map in frame["waferMap"].head(2_000):
        values = set(np.unique(np.asarray(wafer_map)).tolist())
        assert values <= {float(v) for v in WAFER_MAP_ALPHABET} | WAFER_MAP_ALPHABET


def test_every_label_in_the_archive_is_a_known_defect_pattern(
    frame: pd.DataFrame,
) -> None:
    """The nine values in DefectPattern are the nine the dataset actually uses."""
    found: set[str] = set()
    for raw in frame["failureType"]:
        array = np.asarray(raw)
        if array.size:
            found.add(str(array.ravel()[0]))
    mapped = {DefectPattern.from_dataset_label(value) for value in found}
    assert mapped == set(DefectPattern)


def test_most_of_the_archive_is_unlabeled(frame: pd.DataFrame) -> None:
    """The premise of the whole project.

    If this ever inverted, active learning would stop being the right shape for
    the problem and the label efficiency curve would lose its meaning.
    """
    labeled = sum(1 for raw in frame["failureType"] if np.asarray(raw).size)
    assert labeled == 172_950
    assert labeled / len(frame) < 0.25


def test_die_total_reproduces_the_datasets_own_die_size(frame: pd.DataFrame) -> None:
    """An independent check on the DIE_STATISTICS rule.

    ``dieSize`` is the die count for the wafer, computed by whoever published the
    dataset. Counting non-zero cells in the raw map must reproduce it exactly.
    """
    checked = 0
    for record in itertools.islice(iter_records(frame), SAMPLE_ROWS):
        assert record.statistics.die_total == pytest.approx(record.die_size), (
            f"{record.wafer_identifier}: counted {record.statistics.die_total} die "
            f"but dieSize is {record.die_size}"
        )
        checked += 1
    assert checked == SAMPLE_ROWS


def test_die_fail_never_exceeds_die_total(frame: pd.DataFrame) -> None:
    for record in itertools.islice(iter_records(frame), SAMPLE_ROWS):
        assert 0 <= record.statistics.die_fail <= record.statistics.die_total
        assert 0.0 <= record.statistics.failure_rate <= 1.0


def test_wafer_identifiers_are_unique(frame: pd.DataFrame) -> None:
    identifiers = [
        record.wafer_identifier for record in itertools.islice(iter_records(frame), SAMPLE_ROWS)
    ]
    assert len(set(identifiers)) == len(identifiers)


def test_normalization_preserves_the_alphabet_on_real_maps(frame: pd.DataFrame) -> None:
    """Nearest-neighbour resampling must never invent a die value.

    Real maps vary in shape, so this exercises both upsampling and downsampling
    against the same fixed grid the classifier will see.
    """
    settings = Settings()
    shapes_seen: set[tuple[int, int]] = set()
    # Strided rather than a prefix: the archive is ordered by lot, and wafers in
    # one lot share a map shape, so the first N rows cover almost no variety.
    spread = frame.iloc[::271]
    for record in itertools.islice(iter_records(spread), 3_000):
        shapes_seen.add(record.raw_map.shape)
        grid = normalize_grid(record.raw_map, settings.grid_height, settings.grid_width)
        assert grid.shape == settings.grid_shape
        assert set(np.unique(grid).tolist()) <= WAFER_MAP_ALPHABET
        assert len(encode_grid(grid)) == settings.grid_height * settings.grid_width
    assert len(shapes_seen) > 50, "strided sample should cover many real map shapes"


def test_partitioning_never_splits_a_lot(frame: pd.DataFrame) -> None:
    """The assertion the whole evaluation rests on."""
    settings = Settings()
    lot_names = [str(name) for name in frame["lotName"].head(100_000)]
    assignments = assign_splits(
        lot_names,
        settings.partition_seed,
        settings.partition_train_fraction,
        settings.partition_val_fraction,
    )
    per_lot: dict[str, set[SplitName]] = {}
    for name in lot_names:
        per_lot.setdefault(name, set()).add(assignments[name])
    assert all(len(splits) == 1 for splits in per_lot.values())


def test_partition_proportions_hold_on_the_real_lot_names(frame: pd.DataFrame) -> None:
    settings = Settings()
    lot_names = [str(name) for name in frame["lotName"].unique()]
    assert len(lot_names) == EXPECTED_LOTS
    counts = split_counts(
        assign_splits(
            lot_names,
            settings.partition_seed,
            settings.partition_train_fraction,
            settings.partition_val_fraction,
        )
    )
    total = sum(counts.values())
    assert counts[SplitName.TRAIN] / total == pytest.approx(0.70, abs=0.01)
    assert counts[SplitName.VAL] / total == pytest.approx(0.15, abs=0.01)
    assert counts[SplitName.HOLDOUT] / total == pytest.approx(0.15, abs=0.01)


def test_lot_ordinals_cover_every_real_lot(frame: pd.DataFrame) -> None:
    lot_names = [str(name) for name in frame["lotName"].unique()]
    ordinals = lot_ordinals(lot_names)
    assert len(ordinals) == EXPECTED_LOTS
    assert sorted(ordinals.values()) == list(range(EXPECTED_LOTS))
