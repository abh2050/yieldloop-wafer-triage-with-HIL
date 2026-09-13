"""Ingest the real WM811K archive into Postgres.

Reads ``LSWMD.pkl``, normalizes every wafer map to the configured grid, assigns
each lot deterministically to a split, and writes lots and wafers in bulk.

The full archive is 811,457 wafers across 46,293 lots. ``--max-lots`` ingests a
prefix of the lot ordering for a faster working set; because partitioning hashes
the lot name rather than shuffling, a subset is still a valid lot-keyed partition
with the same train/val/holdout proportions, and lots keep the splits they would
have had in a full ingest.

Idempotent: re-running truncates and reloads, so a partial run cannot leave the
database holding half a dataset with no indication of it.

Usage::

    python scripts/bootstrap_db.py                  # everything
    python scripts/bootstrap_db.py --max-lots 3000  # a working subset
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator, Mapping
from uuid import UUID

from sqlalchemy import insert, text
from sqlalchemy.orm import Session

from yieldloop.config import Settings, get_settings
from yieldloop.db.enums import SplitName
from yieldloop.db.models import Lot, Wafer
from yieldloop.db.session import build_engine
from yieldloop.ingest.loader import RawWaferRecord, iter_records, read_frame
from yieldloop.ingest.normalize import encode_grid, lot_date, normalize_grid
from yieldloop.ingest.partition import assign_splits, lot_ordinals
from yieldloop.logging import configure_logging, get_logger

logger = get_logger(__name__)

#: Rows per bulk insert. Large enough to amortize round trips, small enough that
#: the encoded grids for one batch stay comfortably in memory.
BATCH_SIZE = 5_000


def _wafer_rows(
    records: Iterator[RawWaferRecord],
    *,
    settings: Settings,
    lot_ids: Mapping[str, UUID],
    splits: Mapping[str, SplitName],
    keep_lots: frozenset[str] | None,
) -> Iterator[dict[str, object]]:
    """Yield insertable wafer rows, normalizing each map as it goes."""
    for record in records:
        if keep_lots is not None and record.lot_name not in keep_lots:
            continue
        grid = normalize_grid(record.raw_map, settings.grid_height, settings.grid_width)
        yield {
            "lot_id": lot_ids[record.lot_name],
            "wafer_id": record.wafer_identifier,
            "wafer_index": record.wafer_index,
            "die_size": record.die_size,
            "source_row": record.source_row,
            "raw_height": record.raw_map.shape[0],
            "raw_width": record.raw_map.shape[1],
            "grid_height": settings.grid_height,
            "grid_width": settings.grid_width,
            "grid": encode_grid(grid),
            "die_total": record.statistics.die_total,
            "die_fail": record.statistics.die_fail,
            "dataset_label": record.dataset_label,
            "dataset_split_label": record.dataset_split_label,
            "split": splits[record.lot_name],
        }


def bootstrap(session: Session, settings: Settings, *, max_lots: int | None) -> tuple[int, int]:
    """Load the archive. Returns ``(lots, wafers)`` written."""
    frame = read_frame(settings=settings)

    lot_names = [str(name) for name in frame["lotName"].unique()]
    ordinals = lot_ordinals(lot_names)
    splits = assign_splits(
        lot_names,
        settings.partition_seed,
        settings.partition_train_fraction,
        settings.partition_val_fraction,
    )

    keep_lots: frozenset[str] | None = None
    if max_lots is not None and max_lots < len(lot_names):
        keep_lots = frozenset(name for name, ordinal in ordinals.items() if ordinal < max_lots)
        lot_names = sorted(keep_lots, key=lambda n: ordinals[n])

    wafer_counts = frame["lotName"].value_counts()

    logger.info("bootstrap_start", lots=len(lot_names), wafers_in_archive=len(frame))

    # A partial run must not leave a half-loaded database behind.
    session.execute(text("TRUNCATE lots RESTART IDENTITY CASCADE"))

    lot_rows = [
        {
            "lot_name": name,
            "wafer_count": int(wafer_counts[name]),
            "split": splits[name],
            "lot_ordinal": ordinals[name],
            "derived_date": lot_date(ordinals[name]),
            "derivation_rule": "LOT_ORDINAL+LOT_DATE",
        }
        for name in lot_names
    ]
    session.execute(insert(Lot), lot_rows)
    session.flush()

    lot_ids: Mapping[str, UUID] = dict(
        session.execute(text("SELECT lot_name, id FROM lots")).all()  # type: ignore[arg-type]
    )

    written = 0
    batch: list[dict[str, object]] = []
    started = time.monotonic()
    for row in _wafer_rows(
        iter_records(frame),
        settings=settings,
        lot_ids=lot_ids,
        splits=splits,
        keep_lots=keep_lots,
    ):
        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            session.execute(insert(Wafer), batch)
            written += len(batch)
            batch.clear()
            logger.info(
                "bootstrap_progress",
                wafers=written,
                rate_per_second=round(written / max(time.monotonic() - started, 1e-6)),
            )
    if batch:
        session.execute(insert(Wafer), batch)
        written += len(batch)

    session.commit()
    logger.info(
        "bootstrap_complete",
        lots=len(lot_rows),
        wafers=written,
        seconds=round(time.monotonic() - started, 1),
    )
    return len(lot_rows), written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--max-lots",
        type=int,
        default=None,
        help="ingest only the first N lots by ordinal (default: all 46,293)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings)

    if args.max_lots is not None and args.max_lots < 1:
        print("--max-lots must be positive", file=sys.stderr)
        return 2

    engine = build_engine(settings)
    with Session(engine) as session:
        lots, wafers = bootstrap(session, settings, max_lots=args.max_lots)

    print(f"ingested {lots:,} lots and {wafers:,} wafers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
