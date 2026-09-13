"""Scheduled telemetry rollups. This module is the worker container's entrypoint.

Computes a drift snapshot over a recent window and writes it to
``drift_snapshots``, which is what the model health screen reads.

Run as a loop by default, because that is what a worker container does. A single
pass is available with ``--once`` for a cron-driven deployment or for a manual
check.

Nothing here changes behaviour. Telemetry that silently retunes thresholds would
mean the routing configuration no longer matches what is recorded in
``threshold_changes``, and the audit trail would stop explaining the system.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import FrameType

import numpy as np
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from yieldloop.config import Settings, get_settings
from yieldloop.db.enums import ArtifactKind, SplitName
from yieldloop.db.models import Decision, DriftSnapshot, ModelArtifact, Prediction, Wafer
from yieldloop.db.session import build_engine
from yieldloop.logging import configure_logging, get_logger
from yieldloop.models.embed import decode_embedding
from yieldloop.telemetry.drift import (
    CalibrationDrift,
    PSIResult,
    calibration_drift,
    embedding_psi,
)

logger = get_logger(__name__)

#: How far back a rollup looks by default.
DEFAULT_WINDOW_HOURS = 24

#: Snapshots below this many decisions are written but flagged as thin. A rate
#: computed from a handful of decisions is noise, and acting on it is worse than
#: having no number at all.
MIN_DECISIONS_FOR_CONFIDENCE = 30

#: Set by SIGTERM/SIGINT. An Event rather than a module-level bool: it is the
#: correct primitive for signalling across a handler, and it lets the sleep below
#: wake immediately instead of waiting out the remaining interval.
_shutdown = threading.Event()


def _handle_signal(signum: int, _frame: FrameType | None) -> None:
    """Finish the current rollup, then stop.

    Killing mid-write would leave a partial snapshot that looks like a real
    measurement.
    """
    _shutdown.set()
    logger.info("shutdown_requested", signal=signum)


@dataclass(frozen=True, slots=True)
class RollupResult:
    snapshot: DriftSnapshot
    psi: PSIResult
    calibration: CalibrationDrift
    thin: bool


def _window_query(window_start: datetime) -> Select[tuple[Decision]]:
    return select(Decision).where(Decision.created_at >= window_start)


def _reference_embeddings(session: Session, settings: Settings, limit: int = 5_000) -> np.ndarray:
    """Embeddings from the training split: the distribution the model learned."""
    rows = (
        session.execute(
            select(Prediction.embedding)
            .join(Wafer, Prediction.wafer_id == Wafer.id)
            .where(Wafer.split == SplitName.TRAIN, Prediction.embedding.is_not(None))
            .limit(limit)
        )
        .scalars()
        .all()
    )
    vectors = [decode_embedding(row, settings.embedding_dim) for row in rows if row]
    return np.vstack(vectors) if vectors else np.zeros((0, settings.embedding_dim))


def _current_embeddings(
    session: Session, settings: Settings, window_start: datetime, limit: int = 5_000
) -> np.ndarray:
    """Embeddings of wafers predicted on inside the window."""
    rows = (
        session.execute(
            select(Prediction.embedding)
            .where(Prediction.created_at >= window_start, Prediction.embedding.is_not(None))
            .limit(limit)
        )
        .scalars()
        .all()
    )
    vectors = [decode_embedding(row, settings.embedding_dim) for row in rows if row]
    return np.vstack(vectors) if vectors else np.zeros((0, settings.embedding_dim))


def compute_rollup(
    session: Session, settings: Settings, *, window_hours: int = DEFAULT_WINDOW_HOURS
) -> RollupResult | None:
    """Compute and persist one drift snapshot. Returns None when there is nothing
    to measure."""
    window_end = datetime.now(UTC)
    window_start = window_end - timedelta(hours=window_hours)

    decisions = session.execute(_window_query(window_start)).scalars().all()
    artifact = session.execute(
        select(ModelArtifact).where(
            ModelArtifact.kind == ArtifactKind.CLASSIFIER, ModelArtifact.is_active.is_(True)
        )
    ).scalar_one_or_none()

    if artifact is None:
        logger.info("rollup_skipped", reason="no active classifier")
        return None

    with_model = [d for d in decisions if d.model_label is not None]
    overrides = sum(1 for d in with_model if d.is_override)
    override_rate = overrides / len(with_model) if with_model else 0.0

    confidences = np.array(
        [d.model_confidence for d in with_model if d.model_confidence is not None],
        dtype=np.float64,
    )
    correct = np.array(
        [not d.is_override for d in with_model if d.model_confidence is not None],
        dtype=bool,
    )
    # "Correct" here means the reviewer agreed. That is the only ground truth
    # available at serving time, and it is not the same as the label being right;
    # the eval harness measures the latter against holdout.
    baseline_ece = float(artifact.metrics.get("val_ece_calibrated", 0.0) or 0.0)
    calibration = calibration_drift(baseline_ece, confidences, correct)

    psi = embedding_psi(
        _reference_embeddings(session, settings),
        _current_embeddings(session, settings, window_start),
    )

    per_class: dict[str, dict[str, float | int]] = {}
    for decision in with_model:
        if decision.model_label is None:
            continue
        key = decision.model_label.value
        entry = per_class.setdefault(key, {"decisions": 0, "overrides": 0})
        entry["decisions"] = int(entry["decisions"]) + 1
        if decision.is_override:
            entry["overrides"] = int(entry["overrides"]) + 1
    for entry in per_class.values():
        total = int(entry["decisions"])
        entry["override_rate"] = int(entry["overrides"]) / total if total else 0.0

    snapshot = DriftSnapshot(
        artifact_id=artifact.id,
        window_start=window_start,
        window_end=window_end,
        decision_count=len(decisions),
        override_rate=override_rate,
        expected_calibration_error=calibration.current_ece,
        input_psi=psi.value,
        per_class=per_class,
    )
    session.add(snapshot)
    session.commit()

    thin = len(with_model) < MIN_DECISIONS_FOR_CONFIDENCE
    logger.info(
        "rollup_written",
        decisions=len(decisions),
        override_rate=round(override_rate, 4),
        ece=round(calibration.current_ece, 4),
        ece_baseline=round(baseline_ece, 4),
        calibration_degraded=calibration.degraded,
        input_psi=round(psi.value, 4),
        psi_severity=psi.severity,
        thin_sample=thin,
    )

    if calibration.degraded and not thin:
        logger.warning(
            "calibration_degraded",
            baseline_ece=baseline_ece,
            current_ece=calibration.current_ece,
            detail="routing thresholds were tuned against the baseline figure",
        )
    if psi.actionable:
        logger.warning(
            "input_distribution_shifted",
            psi=psi.value,
            detail="incoming wafers differ from the training distribution",
        )

    return RollupResult(snapshot=snapshot, psi=psi, calibration=calibration, thin=thin)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--once", action="store_true", help="compute one rollup and exit")
    parser.add_argument("--window-hours", type=int, default=DEFAULT_WINDOW_HOURS)
    parser.add_argument(
        "--interval-seconds", type=int, default=3600, help="loop interval (ignored with --once)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings)

    if args.window_hours < 1:
        print("--window-hours must be positive", file=sys.stderr)
        return 2

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    engine = build_engine(settings)
    logger.info(
        "telemetry_start",
        once=args.once,
        window_hours=args.window_hours,
        interval_seconds=args.interval_seconds,
    )

    while True:
        with Session(engine) as session:
            try:
                compute_rollup(session, settings, window_hours=args.window_hours)
            except Exception:
                # A failed rollup must not take the worker down: telemetry is
                # observability, and losing it should never stop review work.
                logger.exception("rollup_failed")
                session.rollback()

        if args.once or _shutdown.is_set():
            break

        # Waits on the event rather than sleeping, so SIGTERM is honoured
        # immediately instead of after the remaining interval.
        if _shutdown.wait(timeout=args.interval_seconds):
            break

    logger.info("telemetry_stop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
