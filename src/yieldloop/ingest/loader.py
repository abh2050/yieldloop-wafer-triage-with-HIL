"""Read the real WM811K archive from local disk.

The archive is a pickled pandas DataFrame, and three properties of the real file
shape this module.

**It is a Python 2 pickle written with pandas 0.x (2019).** The module paths it
references (``pandas.indexes.base``, ``pandas.indexes.range``) no longer exist,
and its strings are latin1. Modern pandas cannot read it, so
:class:`_LegacyUnpickler` maps the old module paths onto their current homes.
Converting the file once and committing the result is the obvious alternative and
is rejected deliberately: it would put a derived artifact between the published
dataset and the metrics, and the sha256 in the config would then verify our copy
rather than the real thing.

**Its columns are not all scalar.** ``failureType`` and the train/test label are
stored per row as small numpy arrays -- ``(1, 1)`` when present and ``(0, 0)``
when the wafer is unlabeled, which is the majority of the file. ``lotName`` and
``waferIndex`` arrive as numpy scalars. Unwrapping is explicit and anything
unexpected raises.

**It ships a typo.** The train/test column is spelled ``trianTestLabel`` in the
real archive. yieldloop reads that name and exposes it under the corrected one
rather than silently tolerating either, so a future file that fixes the spelling
surfaces as a contract failure to be looked at rather than as a column that
quietly starts arriving empty.

Nothing here falls back to generated data. If the file is absent, unverified, or
does not have the documented columns, loading fails.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Final

import numpy as np
import pandas as pd

from yieldloop.config import Settings, get_settings
from yieldloop.db.enums import DefectPattern
from yieldloop.ingest.normalize import (
    DieStatistics,
    WaferMap,
    WaferMapError,
    as_wafer_map,
    die_statistics,
    wafer_id,
)
from yieldloop.logging import get_logger

logger = get_logger(__name__)

#: The train/test column as the real archive spells it. The misspelling is in
#: the published dataset, not here. See the module docstring.
SOURCE_SPLIT_COLUMN: Final[str] = "trianTestLabel"

#: Columns the data contract requires to be present in LSWMD.pkl, exactly as the
#: real file spells them.
REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {"waferMap", "dieSize", "lotName", "waferIndex", SOURCE_SPLIT_COLUMN, "failureType"}
)

#: Module paths that moved between the pandas version that wrote the archive and
#: the one that reads it. Mapping them is what makes the original file loadable
#: without converting it first.
_LEGACY_MODULES: Final[dict[str, str]] = {
    "pandas.indexes.base": "pandas.core.indexes.base",
    "pandas.indexes.range": "pandas.core.indexes.range",
    "pandas.indexes.numeric": "pandas.core.indexes.base",
    "pandas.core.index": "pandas.core.indexes.base",
    "numpy.core.multiarray": "numpy._core.multiarray",
    "numpy.core.numeric": "numpy._core.numeric",
}


class _LegacyUnpickler(pickle.Unpickler):
    """Unpickler that resolves the pandas 0.x module layout.

    Only module *paths* are remapped. No class is substituted for a different
    one, so the objects reconstructed here are the ones the file describes.
    """

    def find_class(self, module: str, name: str) -> Any:
        resolved = _LEGACY_MODULES.get(module, module)
        try:
            return super().find_class(resolved, name)
        except (ModuleNotFoundError, AttributeError) as exc:
            raise DatasetContractError(
                f"LSWMD.pkl references {module}.{name}, which does not resolve under the "
                f"installed pandas/numpy (tried {resolved}.{name}). See the compatibility "
                "notes in yieldloop.ingest.loader."
            ) from exc


def load_legacy_pickle(handle: IO[bytes]) -> Any:
    """Unpickle the archive, tolerating its Python 2 / pandas 0.x provenance."""
    return _LegacyUnpickler(handle, encoding="latin1").load()


class DatasetContractError(RuntimeError):
    """Raised when the archive does not match the documented data contract."""


@dataclass(frozen=True, slots=True)
class RawWaferRecord:
    """One wafer as it exists in the real dataset, before normalization."""

    source_row: int
    lot_name: str
    wafer_index: int
    die_size: float
    raw_map: WaferMap
    statistics: DieStatistics
    #: None where WM811K carries no human label, which is most of the dataset.
    dataset_label: DefectPattern | None
    dataset_split_label: str | None

    @property
    def wafer_identifier(self) -> str:
        return wafer_id(self.lot_name, self.wafer_index)


def _unwrap_scalar(value: Any) -> Any:
    """Reduce a possibly-nested numpy container to a python scalar, or None.

    WM811K stores optional string fields as arrays: ``array([['Center']])`` when
    present and ``array([], dtype=float64)`` when absent. Returns None for the
    empty case rather than inventing a value.
    """
    current = value
    while isinstance(current, np.ndarray | list | tuple):
        if len(current) == 0:
            return None
        if len(current) > 1:
            raise DatasetContractError(
                f"expected a single value but found {len(current)} entries: {current!r}"
            )
        current = current[0]
    if isinstance(current, np.generic):
        current = current.item()
    if current is None:
        return None
    if isinstance(current, float) and np.isnan(current):
        return None
    if isinstance(current, str) and not current.strip():
        return None
    return current


def _coerce_label(value: Any) -> DefectPattern | None:
    raw = _unwrap_scalar(value)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise DatasetContractError(f"failureType must be a string; got {type(raw).__name__}")
    return DefectPattern.from_dataset_label(raw)


def _coerce_split_label(value: Any) -> str | None:
    raw = _unwrap_scalar(value)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise DatasetContractError(
            f"{SOURCE_SPLIT_COLUMN} must be a string; got {type(raw).__name__}"
        )
    return raw.strip()


def read_frame(path: Path | None = None, settings: Settings | None = None) -> pd.DataFrame:
    """Load the archive into a DataFrame and assert the documented columns.

    Raises:
        DatasetContractError: if the file is missing or its columns have changed.
    """
    resolved = settings if settings is not None else get_settings()
    target = path if path is not None else resolved.wm811k_path
    if not target.is_file():
        raise DatasetContractError(
            f"{target} does not exist. Run `python scripts/fetch_dataset.py` to obtain the "
            "real WM811K archive; yieldloop has no generated fallback."
        )

    logger.info("dataset_load_start", path=str(target))
    with target.open("rb") as handle:
        frame = load_legacy_pickle(handle)
    if not isinstance(frame, pd.DataFrame):
        raise DatasetContractError(
            f"{target} unpickled to {type(frame).__name__}, expected a pandas DataFrame"
        )

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise DatasetContractError(
            f"{target} is missing required columns {sorted(missing)}; found "
            f"{sorted(frame.columns)}. See docs/data_contract.md."
        )
    logger.info("dataset_loaded", rows=len(frame), columns=sorted(frame.columns))
    return frame


def iter_records(frame: pd.DataFrame, *, skip_invalid: bool = True) -> Iterator[RawWaferRecord]:
    """Yield one :class:`RawWaferRecord` per usable row.

    A small number of WM811K rows carry degenerate maps -- zero-size, or entirely
    outside the wafer. With ``skip_invalid`` they are counted and skipped, and the
    count is logged so the loss is visible rather than silent. With
    ``skip_invalid=False`` the first such row raises, which is what the ingest
    contract test asserts against.
    """
    skipped = 0
    for position, (_, row) in enumerate(frame.iterrows()):
        try:
            lot_name = _unwrap_scalar(row["lotName"])
            if not isinstance(lot_name, str) or not lot_name.strip():
                raise DatasetContractError(f"row {position} has an unusable lotName {lot_name!r}")

            wafer_index_raw = _unwrap_scalar(row["waferIndex"])
            if wafer_index_raw is None:
                raise DatasetContractError(f"row {position} has no waferIndex")
            wafer_index = int(wafer_index_raw)

            die_size_raw = _unwrap_scalar(row["dieSize"])
            if die_size_raw is None:
                raise DatasetContractError(f"row {position} has no dieSize")

            raw_map = as_wafer_map(row["waferMap"])
            statistics = die_statistics(raw_map)
        except (WaferMapError, DatasetContractError, TypeError, ValueError):
            if not skip_invalid:
                raise
            skipped += 1
            continue

        yield RawWaferRecord(
            source_row=position,
            lot_name=lot_name.strip(),
            wafer_index=wafer_index,
            die_size=float(die_size_raw),
            raw_map=raw_map,
            statistics=statistics,
            dataset_label=_coerce_label(row["failureType"]),
            dataset_split_label=_coerce_split_label(row[SOURCE_SPLIT_COLUMN]),
        )

    if skipped:
        logger.warning("dataset_rows_skipped", skipped=skipped, total=len(frame))
