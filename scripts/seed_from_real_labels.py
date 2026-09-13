"""Build the retrieval corpus from real labels.

Derives historical excursions and process events from real WM811K structure,
embeds each lot with the active classifier, and builds the FAISS index the agent
retrieves evidence from.

Every row written here is either a real measurement or a documented deterministic
function of one, and each carries the rule that produced it. Nothing in this
script invents a tool, a chamber, a recipe, or a timestamp, because the dataset
contains none of those -- which is exactly what makes the grounding suite
meaningful: the set of citable evidence is known exactly, so a citation to
anything else is provably a fabrication.

Usage::

    python scripts/seed_from_real_labels.py --max-lots 5000
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import date
from uuid import UUID

import numpy as np
import numpy.typing as npt
import torch
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from yieldloop.config import Settings, get_settings
from yieldloop.db.enums import DefectPattern
from yieldloop.db.models import HistoricalExcursion, Lot, ProcessEvent, Wafer
from yieldloop.db.session import build_engine
from yieldloop.ingest.normalize import (
    LotSummary,
    derive_process_events,
    dominant_pattern,
    evidence_id_for_excursion,
    evidence_id_for_process_event,
    resolution_text,
    resolve_cause,
)
from yieldloop.logging import configure_logging, get_logger
from yieldloop.models.classifier import WaferCNN
from yieldloop.models.embed import select_device
from yieldloop.models.train import load_trained
from yieldloop.retrieval.index import build_index, save_index

logger = get_logger(__name__)


def _lot_summaries(session: Session, max_lots: int | None) -> list[LotSummary]:
    """Read real per-lot structure straight from the database."""
    statement = text(
        """
        SELECT l.lot_name, l.lot_ordinal, count(w.id) AS wafer_count,
               array_agg(w.wafer_index ORDER BY w.wafer_index) AS indices,
               max(w.die_size) AS die_size,
               sum(w.die_total) AS die_total, sum(w.die_fail) AS die_fail
        FROM lots l JOIN wafers w ON w.lot_id = l.id
        GROUP BY l.lot_name, l.lot_ordinal
        ORDER BY l.lot_ordinal
        """
        + (f" LIMIT {int(max_lots)}" if max_lots else "")
    )
    return [
        LotSummary(
            lot_name=row.lot_name,
            lot_ordinal=int(row.lot_ordinal),
            wafer_count=int(row.wafer_count),
            wafer_indices=tuple(int(i) for i in row.indices),
            die_size=float(row.die_size),
            die_total=int(row.die_total),
            die_fail=int(row.die_fail),
        )
        for row in session.execute(statement)
    ]


def seed(session: Session, settings: Settings, *, max_lots: int | None) -> tuple[int, int, int]:
    """Write process events, excursions, and the index. Returns the three counts."""
    summaries = _lot_summaries(session, max_lots)
    if not summaries:
        raise RuntimeError("no lots found; run scripts/bootstrap_db.py first")

    lot_ids: dict[str, UUID] = {
        str(name): identifier for name, identifier in session.execute(select(Lot.lot_name, Lot.id))
    }

    # The scale a failure-rate step is judged against: the real spread of lot
    # failure rates across the dataset, not a chosen constant.
    rates = np.array([s.failure_rate for s in summaries], dtype=np.float64)
    failure_rate_std = float(rates.std())
    logger.info("seed_start", lots=len(summaries), failure_rate_std=round(failure_rate_std, 5))

    session.execute(text("TRUNCATE process_events, historical_excursions CASCADE"))

    event_count = 0
    previous: LotSummary | None = None
    for summary in summaries:
        for event in derive_process_events(summary, previous, failure_rate_std=failure_rate_std):
            session.add(
                ProcessEvent(
                    evidence_id=evidence_id_for_process_event(summary.lot_name, event.ordinal),
                    lot_id=lot_ids[summary.lot_name],
                    event_date=_lot_date(session, summary.lot_name),
                    category=event.category,
                    summary=event.summary,
                    attributes=dict(event.attributes),
                    is_derived=True,
                    derivation_rule=f"PROCESS_EVENT_DERIVE:{event.rule}",
                )
            )
            event_count += 1
        previous = summary
    session.flush()

    excursion_count, indexed = _seed_excursions(session, settings, summaries, lot_ids)
    session.commit()

    logger.info(
        "seed_complete",
        process_events=event_count,
        excursions=excursion_count,
        indexed=indexed,
    )
    return event_count, excursion_count, indexed


def _lot_date(session: Session, lot_name: str) -> date:
    return session.execute(select(Lot.derived_date).where(Lot.lot_name == lot_name)).scalar_one()


def _seed_excursions(
    session: Session,
    settings: Settings,
    summaries: list[LotSummary],
    lot_ids: dict[str, UUID],
) -> tuple[int, int]:
    """Write one excursion per lot that has a dominant human label, and index it.

    A lot with no labeled wafers is skipped rather than given a placeholder
    cause: precedent with no observed pattern is not precedent for anything.
    """
    loaded = load_trained(session, settings)
    if loaded is None:
        raise RuntimeError(
            "no active classifier; train one before seeding, since excursion centroids "
            "are embeddings and retrieval cannot work without them"
        )
    model, _ = loaded
    device = select_device()
    model.to(device)

    wanted = {s.lot_name for s in summaries}
    rows = session.execute(
        select(Wafer.wafer_id, Wafer.dataset_label, Wafer.grid, Wafer.grid_height, Wafer.grid_width)
        .join(Lot, Wafer.lot_id == Lot.id)
        .where(Lot.lot_name.in_(wanted), Wafer.dataset_label.is_not(None))
        .order_by(Wafer.wafer_id)
    ).all()

    by_lot: dict[str, list[DefectPattern]] = defaultdict(list)
    for wafer_id, label, _grid, _h, _w in rows:
        if label is not None:
            by_lot[wafer_id.rsplit("-", 1)[0]].append(label)

    summary_by_name = {s.lot_name: s for s in summaries}
    identifiers: list[str] = []
    centroids: list[np.ndarray] = []
    written = 0

    for lot_name, labels in by_lot.items():
        summary = summary_by_name.get(lot_name)
        if summary is None:
            continue
        dominant = dominant_pattern(labels)
        if dominant is None:
            continue
        pattern, share, labeled = dominant

        centroid = _lot_centroid(session, model, device, lot_name)
        if centroid is None:
            continue

        evidence_id = evidence_id_for_excursion(lot_name)
        session.add(
            HistoricalExcursion(
                evidence_id=evidence_id,
                lot_id=lot_ids[lot_name],
                observed_pattern=pattern,
                pattern_share=share,
                resolved_cause=resolve_cause(pattern),
                resolution_text=resolution_text(
                    lot_name, pattern, share, labeled, summary.wafer_count, summary.failure_rate
                ),
                centroid=centroid.astype(np.float32).tobytes(),
                is_derived=True,
                derivation_rule="EXCURSION_RESOLUTION",
            )
        )
        identifiers.append(evidence_id)
        centroids.append(centroid)
        written += 1

    session.flush()

    if centroids:
        index, manifest = build_index(np.vstack(centroids), identifiers, artifact_hash="0" * 64)
        save_index(index, manifest, settings.faiss_index_path)
    return written, len(identifiers)


def _lot_centroid(
    session: Session, model: WaferCNN, device: torch.device, lot_name: str
) -> npt.NDArray[np.float64] | None:
    """Mean embedding over a lot's labeled wafers, renormalized to unit length.

    Renormalized because the mean of unit vectors is not itself a unit vector,
    and the index is an inner-product index: skipping this would make similarity
    depend on how tightly a lot clusters rather than on where it sits.
    """
    rows = session.execute(
        select(Wafer.grid, Wafer.grid_height, Wafer.grid_width)
        .join(Lot, Wafer.lot_id == Lot.id)
        .where(Lot.lot_name == lot_name, Wafer.dataset_label.is_not(None))
        .limit(25)
    ).all()
    if not rows:
        return None

    grids = np.stack([np.frombuffer(g, dtype=np.uint8).reshape(h, w).copy() for g, h, w in rows])
    with torch.no_grad():
        embeddings = model.embed(torch.from_numpy(grids).to(device)).cpu().numpy()
    centroid: npt.NDArray[np.float64] = embeddings.mean(axis=0).astype(np.float64)
    norm = float(np.linalg.norm(centroid))
    return centroid / norm if norm > 1e-12 else centroid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--max-lots", type=int, default=5000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings)

    started = time.monotonic()
    with Session(build_engine(settings)) as session:
        try:
            events, excursions, indexed = seed(session, settings, max_lots=args.max_lots)
        except RuntimeError as exc:
            print(f"{exc}", file=sys.stderr)
            return 1

    print(
        f"wrote {events:,} process events and {excursions:,} historical excursions; "
        f"indexed {indexed:,} lot centroids in {time.monotonic() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
