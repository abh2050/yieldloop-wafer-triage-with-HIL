"""Deterministic derivations from real WM811K fields.

Every function here is pure and is named by a rule in ``docs/data_contract.md``.
Nothing in this module invents a measurement. Where a value cannot be read from
the dataset it is either omitted or derived by a documented rule whose output is
labelled as derived in the schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final

import numpy as np
import numpy.typing as npt

from yieldloop.db.enums import CauseCategory, DefectPattern

#: WM811K wafer map alphabet. These values are preserved exactly by every
#: transformation in this module; a die never acquires a value the source map
#: did not contain.
OUTSIDE_WAFER: Final[int] = 0
PASSING_DIE: Final[int] = 1
FAILING_DIE: Final[int] = 2
WAFER_MAP_ALPHABET: Final[frozenset[int]] = frozenset({OUTSIDE_WAFER, PASSING_DIE, FAILING_DIE})

#: Rule LOT_DATE. A synthetic index expressed as a date so the retention window
#: and the process-event window have a total order. Not a production date.
LOT_DATE_EPOCH: Final[date] = date(2021, 1, 1)

#: Number of concentric rings used by DIE_STATISTICS.
RADIAL_BIN_COUNT: Final[int] = 5

#: Rule EXCURSION_RESOLUTION. The complete semantic content of the cause labels.
#: This is a documented convention of this repository, not fab knowledge, and the
#: console presents it as such.
PATTERN_TO_CAUSE: Final[dict[DefectPattern, CauseCategory]] = {
    DefectPattern.CENTER: CauseCategory.CHAMBER_CONDITION,
    DefectPattern.DONUT: CauseCategory.CHAMBER_CONDITION,
    DefectPattern.EDGE_RING: CauseCategory.UPSTREAM_PROCESS,
    DefectPattern.EDGE_LOC: CauseCategory.HANDLING_MECHANICAL,
    DefectPattern.LOC: CauseCategory.TOOL_DRIFT,
    DefectPattern.SCRATCH: CauseCategory.HANDLING_MECHANICAL,
    DefectPattern.RANDOM: CauseCategory.MATERIAL_LOT,
    DefectPattern.NEAR_FULL: CauseCategory.RECIPE_CHANGE,
    DefectPattern.NONE: CauseCategory.UNKNOWN,
}

WaferMap = npt.NDArray[np.uint8]


class WaferMapError(ValueError):
    """Raised when a source wafer map violates the documented data contract."""


@dataclass(frozen=True, slots=True)
class DieStatistics:
    """Die-level statistics measured on the raw map, before normalization.

    Computed pre-normalization so that reported counts are the real ones and not
    an artifact of resampling.
    """

    die_total: int
    die_fail: int
    #: Fail rate per concentric ring, innermost first. Length RADIAL_BIN_COUNT.
    radial_fail_rates: tuple[float, ...]

    @property
    def failure_rate(self) -> float:
        return self.die_fail / self.die_total if self.die_total else 0.0

    @property
    def edge_concentration(self) -> float:
        """Outermost ring fail rate divided by the whole-wafer fail rate.

        Above 1.0 means failures concentrate at the edge. Returns 0.0 when the
        wafer has no failures at all, where the ratio is undefined.
        """
        overall = self.failure_rate
        if overall <= 0.0:
            return 0.0
        return self.radial_fail_rates[-1] / overall


def as_wafer_map(raw: npt.ArrayLike) -> WaferMap:
    """Coerce a source ``waferMap`` to a validated 2-D uint8 array.

    Raises:
        WaferMapError: if the map is not 2-D, is empty, or contains a value
            outside the documented ``{0, 1, 2}`` alphabet. Unknown values are
            never coerced or clipped, because doing so would silently change the
            meaning of a die.
    """
    array = np.asarray(raw)
    if array.ndim != 2:
        raise WaferMapError(f"wafer map must be 2-D; got shape {array.shape}")
    if array.size == 0:
        raise WaferMapError("wafer map is empty")

    present = set(np.unique(array).tolist())
    unexpected = present - {float(v) for v in WAFER_MAP_ALPHABET} - WAFER_MAP_ALPHABET
    if unexpected:
        raise WaferMapError(
            f"wafer map contains values outside the documented alphabet "
            f"{sorted(WAFER_MAP_ALPHABET)}: {sorted(unexpected)}"
        )
    return array.astype(np.uint8, copy=False)


def die_statistics(raw_map: WaferMap) -> DieStatistics:
    """Rule ``DIE_STATISTICS``. Measure die counts and radial distribution.

    Radius is normalized against the maximum radius of any die on the wafer, so
    the binning is comparable across the varying map shapes in WM811K.
    """
    inside = raw_map != OUTSIDE_WAFER
    die_total = int(inside.sum())
    if die_total == 0:
        raise WaferMapError("wafer map contains no die inside the wafer")
    failing = raw_map == FAILING_DIE
    die_fail = int(failing.sum())

    rows, cols = np.nonzero(inside)
    centre_row = float(rows.mean())
    centre_col = float(cols.mean())
    radius = np.hypot(rows - centre_row, cols - centre_col)
    max_radius = float(radius.max())

    # A single-die wafer has zero spread; place every die in the innermost ring
    # rather than dividing by zero.
    normalized = radius / max_radius if max_radius > 0.0 else np.zeros_like(radius)
    bin_index = np.minimum((normalized * RADIAL_BIN_COUNT).astype(np.int64), RADIAL_BIN_COUNT - 1)

    fail_flags = failing[rows, cols]
    per_bin_total = np.bincount(bin_index, minlength=RADIAL_BIN_COUNT)
    per_bin_fail = np.bincount(bin_index, weights=fail_flags, minlength=RADIAL_BIN_COUNT)
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = np.where(per_bin_total > 0, per_bin_fail / per_bin_total, 0.0)

    return DieStatistics(
        die_total=die_total,
        die_fail=die_fail,
        radial_fail_rates=tuple(float(r) for r in rates),
    )


def normalize_grid(raw_map: WaferMap, height: int, width: int) -> WaferMap:
    """Rule ``GRID_NORMALIZE``. Resample a wafer map to a fixed grid.

    Nearest-neighbour index mapping, not interpolation. Interpolating would
    average a passing die (1) and a failing die (2) into a value of 1.5 that means
    nothing, so the alphabet is preserved exactly by construction.
    """
    if height <= 0 or width <= 0:
        raise WaferMapError(f"target grid must be positive; got {height}x{width}")

    source_height, source_width = raw_map.shape
    row_index = (np.arange(height) * source_height) // height
    col_index = (np.arange(width) * source_width) // width
    resampled = raw_map[np.ix_(row_index, col_index)]
    return np.ascontiguousarray(resampled, dtype=np.uint8)


def encode_grid(grid: WaferMap) -> bytes:
    """Serialize a normalized grid for the ``wafers.grid`` column."""
    return grid.astype(np.uint8, copy=False).tobytes(order="C")


def decode_grid(payload: bytes, height: int, width: int) -> WaferMap:
    """Inverse of :func:`encode_grid`."""
    expected = height * width
    if len(payload) != expected:
        raise WaferMapError(
            f"grid payload is {len(payload)} bytes but {height}x{width} needs {expected}"
        )
    return np.frombuffer(payload, dtype=np.uint8).reshape(height, width)


def lot_date(lot_ordinal: int, epoch: date = LOT_DATE_EPOCH) -> date:
    """Rule ``LOT_DATE``. Map a lot ordinal onto a synthetic calendar date.

    WM811K has no timestamps. This provides the total order the retention window
    and process-event window require. It is not a production date and nothing in
    the system treats it as one.
    """
    if lot_ordinal < 0:
        raise ValueError(f"lot_ordinal must be non-negative; got {lot_ordinal}")
    return epoch + timedelta(days=lot_ordinal)


def wafer_id(lot_name: str, wafer_index: int) -> str:
    """Build the stable wafer identifier ``{lot_name}-{wafer_index}``."""
    return f"{lot_name}-{wafer_index}"


def evidence_id_for_prediction(wafer_identifier: str) -> str:
    return f"cp:{wafer_identifier}"


def evidence_id_for_die_statistics(wafer_identifier: str) -> str:
    return f"ds:{wafer_identifier}"


def evidence_id_for_excursion(lot_name: str) -> str:
    return f"hx:{lot_name}"


def evidence_id_for_process_event(lot_name: str, ordinal: int) -> str:
    return f"pe:{lot_name}:{ordinal}"


def resolve_cause(pattern: DefectPattern) -> CauseCategory:
    """Rule ``EXCURSION_RESOLUTION``. Map a real defect label to a cause label."""
    return PATTERN_TO_CAUSE[pattern]


def resolution_text(
    lot_name: str,
    pattern: DefectPattern,
    pattern_share: float,
    labeled_wafers: int,
    total_wafers: int,
    failure_rate: float,
) -> str:
    """Render the resolution summary for a historical excursion.

    Every number in the output is a real measurement of the lot. The sentence
    contains no claim that is not a restatement of one of them, and it names the
    mapping as a convention so a reader cannot mistake it for a finding.
    """
    cause = resolve_cause(pattern)
    return (
        f"Lot {lot_name}: {labeled_wafers} of {total_wafers} wafers carry human defect "
        f"labels, of which {pattern_share:.0%} are '{pattern.value}'. Measured die "
        f"failure rate across the lot is {failure_rate:.2%}. Recorded cause category "
        f"'{cause.value}' follows the documented pattern-to-cause mapping in "
        f"docs/data_contract.md; it is a convention of this dataset build, not an "
        f"investigated finding."
    )
