"""The human decision loop, closed.

Two properties the project is named after, neither of which held until now.

A reviewer decision must reach the next training round. Capturing decisions,
auditing them, and reporting override rates on them is not a human loop if the
model never reads them -- and it did not: training loaded `wafers.dataset_label`
and nothing else, so every correction a reviewer made was discarded.

All three gates must have a producer. Only the label gate did. The confirm and
escalation gates were queried by the API, rendered by the console, and had reason
codes defined for them, while nothing in the system ever created a task at
either, so those screens were permanently empty.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from yieldloop.config import Settings
from yieldloop.db.enums import (
    ArtifactKind,
    DecisionAction,
    DefectPattern,
    LabelSource,
    RoutingBand,
    SplitName,
    TaskGate,
    TaskState,
)
from yieldloop.db.models import Lot, ModelArtifact, Prediction, ReasonCode, ReviewTask, Wafer
from yieldloop.guardrails.input_filter import InputFilter
from yieldloop.guardrails.thresholds import RoutingBands
from yieldloop.models.embed import load_samples, reviewer_labels
from yieldloop.review.decisions import DecisionRequest, submit
from yieldloop.review.queue import (
    create_task,
    enqueue_confirmations,
    enqueue_escalations,
)
from yieldloop.review.reason_codes import REASON_CODES

pytestmark = pytest.mark.postgres

BANDS = RoutingBands(confidence_floor=0.55, auto_commit_threshold=0.95)
FILTER = InputFilter.from_settings(Settings())


@pytest.fixture(autouse=True)
def reason_codes(db_session: Session) -> None:
    """Assert the vocabulary is present rather than creating it.

    Seeded by a migration. A fixture that inserted it would pass even if that
    migration were deleted, which is how the missing seed went unnoticed until
    the browser hit a foreign key violation.
    """
    count = db_session.execute(select(func.count()).select_from(ReasonCode)).scalar_one()
    assert int(count) == len(REASON_CODES), "reason_codes is not seeded; run alembic upgrade head"


@pytest.fixture
def lot(db_session: Session) -> Lot:
    row = Lot(
        lot_name="lot00777",
        wafer_count=25,
        split=SplitName.TRAIN,
        lot_ordinal=777,
        derived_date=date(2021, 2, 2),
        derivation_rule="LOT_ORDINAL+LOT_DATE",
    )
    db_session.add(row)
    db_session.flush()
    return row


def _wafer(
    db_session: Session,
    lot: Lot,
    index: int,
    *,
    label: DefectPattern | None,
    split: SplitName = SplitName.TRAIN,
    die_fail: int = 100,
) -> Wafer:
    row = Wafer(
        lot_id=lot.id,
        wafer_id=f"{lot.lot_name}-{index}",
        wafer_index=index,
        die_size=1683.0,
        source_row=int(uuid.uuid4().int % 1_000_000_000),
        raw_height=45,
        raw_width=48,
        grid_height=64,
        grid_width=64,
        grid=bytes(64 * 64),
        die_total=1683,
        die_fail=die_fail,
        dataset_label=label,
        split=split,
    )
    db_session.add(row)
    db_session.flush()
    return row


def _artifact(db_session: Session) -> ModelArtifact:
    row = ModelArtifact(
        kind=ArtifactKind.CLASSIFIER,
        content_hash=uuid.uuid4().hex * 2,
        data_hash=uuid.uuid4().hex * 2,
        git_commit="0" * 40,
        git_dirty=False,
        seed=1,
        hyperparameters={},
        metrics={},
        label_count=10,
    )
    db_session.add(row)
    db_session.flush()
    return row


def _prediction(
    db_session: Session,
    wafer: Wafer,
    artifact: ModelArtifact,
    confidence: float,
    label: DefectPattern,
    entropy: float = 0.5,
) -> Prediction:
    others = [p for p in DefectPattern if p is not label]
    band = (
        RoutingBand.AUTO_COMMIT
        if confidence >= BANDS.auto_commit_threshold
        else RoutingBand.UNCERTAINTY_BAND
        if confidence >= BANDS.confidence_floor
        else RoutingBand.BELOW_FLOOR
    )
    row = Prediction(
        wafer_id=wafer.id,
        artifact_id=artifact.id,
        predicted_label=label,
        confidence=confidence,
        probabilities={
            label.value: confidence,
            **{p.value: (1 - confidence) / len(others) for p in others},
        },
        entropy=entropy,
        routing_band=band,
        auto_commit_threshold=BANDS.auto_commit_threshold,
        confidence_floor=BANDS.confidence_floor,
        inference_ms=1.0,
    )
    db_session.add(row)
    db_session.flush()
    return row


def _decide(db_session: Session, wafer: Wafer, label: DefectPattern) -> None:
    task = create_task(
        db_session,
        wafer=wafer,
        gate=TaskGate.LABEL,
        prediction=None,
        bands=BANDS,
        priority=1.0,
    )
    submit(
        db_session,
        DecisionRequest(
            task_id=task.id,
            reviewer_id="reviewer-a",
            action=DecisionAction.ACCEPT,
            chosen_label=label,
            reason_code=None,
            note=None,
            decision_ms=1500,
        ),
        input_filter=FILTER,
        request_id=uuid.uuid4().hex,
    )


# --- the loop closes ------------------------------------------------------


def test_a_reviewer_label_becomes_a_training_example(db_session: Session, lot: Lot) -> None:
    """The property the project is named after."""
    wafer = _wafer(db_session, lot, 1, label=None)
    assert load_samples(db_session, SplitName.TRAIN, labeled=True) == []

    _decide(db_session, wafer, DefectPattern.SCRATCH)

    samples = load_samples(db_session, SplitName.TRAIN, labeled=True)
    assert [s.wafer_id for s in samples] == [wafer.wafer_id]
    assert samples[0].label is DefectPattern.SCRATCH
    assert samples[0].source is LabelSource.REVIEWER


def test_a_reviewer_label_overrides_the_dataset_label(db_session: Session, lot: Lot) -> None:
    """The reviewer's is the more recent human judgement, made with context."""
    wafer = _wafer(db_session, lot, 2, label=DefectPattern.NONE)
    _decide(db_session, wafer, DefectPattern.EDGE_RING)

    samples = load_samples(db_session, SplitName.TRAIN, labeled=True)
    assert samples[0].label is DefectPattern.EDGE_RING
    assert samples[0].source is LabelSource.REVIEWER


def test_a_reviewed_wafer_leaves_the_unlabeled_pool(db_session: Session, lot: Lot) -> None:
    """Otherwise the sampler offers it for labeling again."""
    wafer = _wafer(db_session, lot, 3, label=None)
    assert [s.wafer_id for s in load_samples(db_session, SplitName.TRAIN, labeled=False)] == [
        wafer.wafer_id
    ]

    _decide(db_session, wafer, DefectPattern.LOC)
    assert load_samples(db_session, SplitName.TRAIN, labeled=False) == []


def test_a_holdout_decision_never_reaches_training(db_session: Session) -> None:
    """Split discipline. A reviewer decision on a holdout wafer would leak."""
    holdout_lot = Lot(
        lot_name="lot00888",
        wafer_count=25,
        split=SplitName.HOLDOUT,
        lot_ordinal=888,
        derived_date=date(2021, 3, 3),
        derivation_rule="LOT_ORDINAL+LOT_DATE",
    )
    db_session.add(holdout_lot)
    db_session.flush()
    wafer = _wafer(db_session, holdout_lot, 1, label=None, split=SplitName.HOLDOUT)
    _decide(db_session, wafer, DefectPattern.DONUT)

    assert reviewer_labels(db_session, SplitName.TRAIN) == {}
    assert load_samples(db_session, SplitName.TRAIN, labeled=True) == []
    assert wafer.wafer_id in reviewer_labels(db_session, SplitName.HOLDOUT)


def test_the_loop_can_be_opened_for_reproducibility(db_session: Session, lot: Lot) -> None:
    """Reproducing a pre-decision run must remain possible."""
    wafer = _wafer(db_session, lot, 4, label=DefectPattern.NONE)
    _decide(db_session, wafer, DefectPattern.CENTER)

    without = load_samples(db_session, SplitName.TRAIN, labeled=True, include_reviewer_labels=False)
    assert without[0].label is DefectPattern.NONE
    assert without[0].source is LabelSource.DATASET


# --- all three gates have producers ---------------------------------------


def test_the_confirm_gate_has_a_producer(db_session: Session, lot: Lot) -> None:
    artifact = _artifact(db_session)
    uncertain = _wafer(db_session, lot, 10, label=None)
    _prediction(db_session, uncertain, artifact, 0.70, DefectPattern.SCRATCH)

    outcome = enqueue_confirmations(db_session, bands=BANDS, limit=50)
    assert outcome.queued_for_confirmation == 1

    task = db_session.execute(
        select(ReviewTask).where(ReviewTask.gate == TaskGate.CONFIRM)
    ).scalar_one()
    assert task.state is TaskState.PENDING
    assert task.show_prediction


def test_confident_predictions_are_auto_committed_not_queued(db_session: Session, lot: Lot) -> None:
    """Auto-commit is a decision not to ask, expressed by producing no task."""
    artifact = _artifact(db_session)
    confident = _wafer(db_session, lot, 11, label=None)
    _prediction(db_session, confident, artifact, 0.99, DefectPattern.EDGE_RING)

    outcome = enqueue_confirmations(db_session, bands=BANDS, limit=50)
    assert outcome.auto_committed == 1
    assert outcome.queued_for_confirmation == 0
    assert (
        db_session.execute(
            select(ReviewTask).where(ReviewTask.gate == TaskGate.CONFIRM)
        ).scalar_one_or_none()
        is None
    )


def test_below_floor_confirmations_withhold_the_prediction(db_session: Session, lot: Lot) -> None:
    """The anchoring rule, applied by the producer rather than the console."""
    artifact = _artifact(db_session)
    weak = _wafer(db_session, lot, 12, label=None)
    _prediction(db_session, weak, artifact, 0.20, DefectPattern.LOC)

    enqueue_confirmations(db_session, bands=BANDS, limit=50)
    task = db_session.execute(
        select(ReviewTask).where(ReviewTask.gate == TaskGate.CONFIRM)
    ).scalar_one()
    assert not task.show_prediction


def test_confirmations_are_idempotent(db_session: Session, lot: Lot) -> None:
    """Queuing a wafer twice gets it confirmed by two reviewers independently."""
    artifact = _artifact(db_session)
    wafer = _wafer(db_session, lot, 13, label=None)
    _prediction(db_session, wafer, artifact, 0.70, DefectPattern.SCRATCH)

    first = enqueue_confirmations(db_session, bands=BANDS, limit=50)
    second = enqueue_confirmations(db_session, bands=BANDS, limit=50)
    assert first.queued_for_confirmation == 1
    assert second.queued_for_confirmation == 0
    assert second.already_queued == 1


def test_the_escalation_gate_has_a_producer(db_session: Session, lot: Lot) -> None:
    """A pattern across a lot is a lot-level cause; one odd wafer is noise."""
    artifact = _artifact(db_session)
    for index in (20, 21, 22):
        wafer = _wafer(db_session, lot, index, label=None, die_fail=500 + index)
        _prediction(db_session, wafer, artifact, 0.80, DefectPattern.EDGE_RING)

    assert enqueue_escalations(db_session, bands=BANDS, min_wafers=2) == 1
    task = db_session.execute(
        select(ReviewTask).where(ReviewTask.gate == TaskGate.ESCALATION)
    ).scalar_one()
    assert task.state is TaskState.PENDING


def test_a_single_flagged_wafer_does_not_escalate_a_lot(db_session: Session, lot: Lot) -> None:
    artifact = _artifact(db_session)
    wafer = _wafer(db_session, lot, 30, label=None)
    _prediction(db_session, wafer, artifact, 0.80, DefectPattern.SCRATCH)

    assert enqueue_escalations(db_session, bands=BANDS, min_wafers=2) == 0


def test_clean_lots_are_not_escalated(db_session: Session, lot: Lot) -> None:
    """`none` at high confidence is the system working, not an excursion."""
    artifact = _artifact(db_session)
    for index in (40, 41, 42):
        wafer = _wafer(db_session, lot, index, label=None)
        _prediction(db_session, wafer, artifact, 0.99, DefectPattern.NONE)

    assert enqueue_escalations(db_session, bands=BANDS, min_wafers=2) == 0


def test_low_confidence_flags_do_not_escalate(db_session: Session, lot: Lot) -> None:
    """Below the floor the model is not confident enough to raise a lot."""
    artifact = _artifact(db_session)
    for index in (50, 51, 52):
        wafer = _wafer(db_session, lot, index, label=None)
        _prediction(db_session, wafer, artifact, 0.30, DefectPattern.SCRATCH)

    assert enqueue_escalations(db_session, bands=BANDS, min_wafers=2) == 0
