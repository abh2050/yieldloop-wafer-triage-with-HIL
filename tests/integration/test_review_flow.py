"""The human decision loop, end to end against a real database.

What matters here is not that a decision is stored, but that it is stored with
enough context to be usable as training signal later: what the reviewer saw, what
the model had said, whether they disagreed, and why.
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
    AuditEventType,
    DecisionAction,
    DefectPattern,
    RoutingBand,
    SplitName,
    TaskGate,
    TaskState,
)
from yieldloop.db.models import (
    AuditRecord,
    Decision,
    Lot,
    ModelArtifact,
    Prediction,
    ReasonCode,
    ReviewTask,
    Wafer,
)
from yieldloop.guardrails.input_filter import InputFilter
from yieldloop.guardrails.thresholds import RoutingBands
from yieldloop.review.decisions import (
    DecisionError,
    DecisionRequest,
    override_rate,
    submit,
)
from yieldloop.review.queue import claim, create_task, next_items, queue_depth
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
def wafer(db_session: Session) -> Wafer:
    """One real-shaped wafer row.

    Schema scaffolding for the flow under test: it carries a grid of the correct
    size but is not dataset content and never reaches a metric.
    """
    lot = Lot(
        lot_name="lot00123",
        wafer_count=25,
        split=SplitName.TRAIN,
        lot_ordinal=123,
        derived_date=date(2021, 5, 4),
        derivation_rule="LOT_ORDINAL+LOT_DATE",
    )
    db_session.add(lot)
    db_session.flush()

    row = Wafer(
        lot_id=lot.id,
        wafer_id="lot00123-7",
        wafer_index=7,
        die_size=1683.0,
        source_row=int(uuid.uuid4().int % 1_000_000_000),
        raw_height=45,
        raw_width=48,
        grid_height=64,
        grid_width=64,
        grid=bytes(64 * 64),
        die_total=1683,
        die_fail=88,
        dataset_label=None,
        split=SplitName.TRAIN,
    )
    db_session.add(row)
    db_session.flush()
    return row


def _artifact(db_session: Session) -> ModelArtifact:
    artifact = ModelArtifact(
        kind=ArtifactKind.CLASSIFIER,
        content_hash=uuid.uuid4().hex + uuid.uuid4().hex,
        data_hash=uuid.uuid4().hex + uuid.uuid4().hex,
        git_commit="0" * 40,
        git_dirty=False,
        seed=1,
        hyperparameters={},
        metrics={},
        label_count=100,
    )
    db_session.add(artifact)
    db_session.flush()
    return artifact


def _prediction(
    db_session: Session, wafer: Wafer, confidence: float, label: DefectPattern
) -> Prediction:
    others = [p for p in DefectPattern if p is not label]
    band = (
        RoutingBand.AUTO_COMMIT
        if confidence >= BANDS.auto_commit_threshold
        else RoutingBand.UNCERTAINTY_BAND
        if confidence >= BANDS.confidence_floor
        else RoutingBand.BELOW_FLOOR
    )
    prediction = Prediction(
        wafer_id=wafer.id,
        artifact_id=_artifact(db_session).id,
        predicted_label=label,
        confidence=confidence,
        probabilities={
            label.value: confidence,
            **{p.value: (1 - confidence) / len(others) for p in others},
        },
        entropy=0.5,
        routing_band=band,
        auto_commit_threshold=BANDS.auto_commit_threshold,
        confidence_floor=BANDS.confidence_floor,
        inference_ms=1.0,
    )
    db_session.add(prediction)
    db_session.flush()
    return prediction


def test_uncertainty_band_task_shows_the_prediction(db_session: Session, wafer: Wafer) -> None:
    prediction = _prediction(db_session, wafer, 0.70, DefectPattern.SCRATCH)
    task = create_task(
        db_session,
        wafer=wafer,
        gate=TaskGate.CONFIRM,
        prediction=prediction,
        bands=BANDS,
        priority=1.0,
    )
    assert task.show_prediction

    items = next_items(db_session, gate=TaskGate.CONFIRM)
    assert items[0].predicted_label == DefectPattern.SCRATCH.value
    assert items[0].confidence == pytest.approx(0.70)


def test_below_floor_task_withholds_the_prediction(db_session: Session, wafer: Wafer) -> None:
    """The anchoring rule, enforced in the service rather than the UI."""
    prediction = _prediction(db_session, wafer, 0.20, DefectPattern.SCRATCH)
    task = create_task(
        db_session,
        wafer=wafer,
        gate=TaskGate.CONFIRM,
        prediction=prediction,
        bands=BANDS,
        priority=1.0,
    )
    assert not task.show_prediction

    items = next_items(db_session, gate=TaskGate.CONFIRM)
    assert items[0].predicted_label is None
    assert items[0].confidence is None
    assert items[0].routing_band is None


def test_label_gate_never_shows_a_prediction_however_confident(
    db_session: Session, wafer: Wafer
) -> None:
    """Even at 0.99. An independent label is the entire point of this gate."""
    prediction = _prediction(db_session, wafer, 0.99, DefectPattern.EDGE_RING)
    task = create_task(
        db_session,
        wafer=wafer,
        gate=TaskGate.LABEL,
        prediction=prediction,
        bands=BANDS,
        priority=1.0,
    )
    assert not task.show_prediction
    assert next_items(db_session, gate=TaskGate.LABEL)[0].predicted_label is None


def test_accepting_a_visible_prediction_records_agreement(
    db_session: Session, wafer: Wafer
) -> None:
    prediction = _prediction(db_session, wafer, 0.70, DefectPattern.SCRATCH)
    task = create_task(
        db_session,
        wafer=wafer,
        gate=TaskGate.CONFIRM,
        prediction=prediction,
        bands=BANDS,
        priority=1.0,
    )
    decision = submit(
        db_session,
        DecisionRequest(
            task_id=task.id,
            reviewer_id="reviewer-a",
            action=DecisionAction.ACCEPT,
            chosen_label=None,
            reason_code=None,
            note=None,
            decision_ms=1800,
        ),
        input_filter=FILTER,
        request_id="req-1",
    )
    assert decision.chosen_label is DefectPattern.SCRATCH
    assert not decision.is_override
    assert decision.prediction_was_shown
    reloaded = db_session.get(ReviewTask, task.id)
    assert reloaded is not None
    assert reloaded.state is TaskState.COMPLETED


def test_editing_records_an_override_with_its_reason(db_session: Session, wafer: Wafer) -> None:
    prediction = _prediction(db_session, wafer, 0.70, DefectPattern.SCRATCH)
    task = create_task(
        db_session,
        wafer=wafer,
        gate=TaskGate.CONFIRM,
        prediction=prediction,
        bands=BANDS,
        priority=1.0,
    )
    decision = submit(
        db_session,
        DecisionRequest(
            task_id=task.id,
            reviewer_id="reviewer-a",
            action=DecisionAction.EDIT,
            chosen_label=DefectPattern.EDGE_RING,
            reason_code="wrong_class",
            note="Ring is closed; this is not a scratch.",
            decision_ms=3100,
        ),
        input_filter=FILTER,
        request_id="req-2",
    )
    assert decision.is_override
    assert decision.chosen_label is DefectPattern.EDGE_RING
    assert decision.model_label is DefectPattern.SCRATCH
    assert decision.reason_code == "wrong_class"


def test_a_change_without_a_reason_is_refused(db_session: Session, wafer: Wafer) -> None:
    """Free-text-only disagreement is unusable as training signal."""
    prediction = _prediction(db_session, wafer, 0.70, DefectPattern.SCRATCH)
    task = create_task(
        db_session,
        wafer=wafer,
        gate=TaskGate.CONFIRM,
        prediction=prediction,
        bands=BANDS,
        priority=1.0,
    )
    with pytest.raises(DecisionError, match="reason code is required"):
        submit(
            db_session,
            DecisionRequest(
                task_id=task.id,
                reviewer_id="reviewer-a",
                action=DecisionAction.EDIT,
                chosen_label=DefectPattern.LOC,
                reason_code=None,
                note=None,
                decision_ms=900,
            ),
            input_filter=FILTER,
            request_id="req-3",
        )


def test_a_reason_code_from_another_gate_is_refused(db_session: Session, wafer: Wafer) -> None:
    prediction = _prediction(db_session, wafer, 0.70, DefectPattern.SCRATCH)
    task = create_task(
        db_session,
        wafer=wafer,
        gate=TaskGate.CONFIRM,
        prediction=prediction,
        bands=BANDS,
        priority=1.0,
    )
    with pytest.raises(DecisionError, match="not valid"):
        submit(
            db_session,
            DecisionRequest(
                task_id=task.id,
                reviewer_id="reviewer-a",
                action=DecisionAction.EDIT,
                chosen_label=DefectPattern.LOC,
                reason_code="hypothesis_unsupported",
                note=None,
                decision_ms=900,
            ),
            input_filter=FILTER,
            request_id="req-4",
        )


def test_accept_with_nothing_shown_must_carry_a_label(db_session: Session, wafer: Wafer) -> None:
    """Otherwise it records agreement with something invisible."""
    prediction = _prediction(db_session, wafer, 0.20, DefectPattern.SCRATCH)
    task = create_task(
        db_session,
        wafer=wafer,
        gate=TaskGate.CONFIRM,
        prediction=prediction,
        bands=BANDS,
        priority=1.0,
    )
    with pytest.raises(DecisionError, match="must still carry a label"):
        submit(
            db_session,
            DecisionRequest(
                task_id=task.id,
                reviewer_id="reviewer-a",
                action=DecisionAction.ACCEPT,
                chosen_label=None,
                reason_code=None,
                note=None,
                decision_ms=900,
            ),
            input_filter=FILTER,
            request_id="req-5",
        )


def test_a_task_cannot_be_decided_twice(db_session: Session, wafer: Wafer) -> None:
    prediction = _prediction(db_session, wafer, 0.70, DefectPattern.SCRATCH)
    task = create_task(
        db_session,
        wafer=wafer,
        gate=TaskGate.CONFIRM,
        prediction=prediction,
        bands=BANDS,
        priority=1.0,
    )
    request = DecisionRequest(
        task_id=task.id,
        reviewer_id="reviewer-a",
        action=DecisionAction.ACCEPT,
        chosen_label=None,
        reason_code=None,
        note=None,
        decision_ms=900,
    )
    submit(db_session, request, input_filter=FILTER, request_id="req-6")
    with pytest.raises(DecisionError, match="already has a decision"):
        submit(db_session, request, input_filter=FILTER, request_id="req-7")


def test_an_oversized_note_is_refused(db_session: Session, wafer: Wafer) -> None:
    prediction = _prediction(db_session, wafer, 0.70, DefectPattern.SCRATCH)
    task = create_task(
        db_session,
        wafer=wafer,
        gate=TaskGate.CONFIRM,
        prediction=prediction,
        bands=BANDS,
        priority=1.0,
    )
    with pytest.raises(Exception, match="over the"):
        submit(
            db_session,
            DecisionRequest(
                task_id=task.id,
                reviewer_id="reviewer-a",
                action=DecisionAction.EDIT,
                chosen_label=DefectPattern.LOC,
                reason_code="wrong_class",
                note="x" * 50_000,
                decision_ms=900,
            ),
            input_filter=FILTER,
            request_id="req-8",
        )


def test_every_decision_is_audited(db_session: Session, wafer: Wafer) -> None:
    prediction = _prediction(db_session, wafer, 0.70, DefectPattern.SCRATCH)
    task = create_task(
        db_session,
        wafer=wafer,
        gate=TaskGate.CONFIRM,
        prediction=prediction,
        bands=BANDS,
        priority=1.0,
    )
    submit(
        db_session,
        DecisionRequest(
            task_id=task.id,
            reviewer_id="reviewer-a",
            action=DecisionAction.EDIT,
            chosen_label=DefectPattern.LOC,
            reason_code="wrong_class",
            note=None,
            decision_ms=2400,
        ),
        input_filter=FILTER,
        request_id="req-9",
    )
    record = db_session.execute(
        select(AuditRecord).where(AuditRecord.event_type == AuditEventType.HUMAN_DECISION)
    ).scalar_one()
    assert record.actor == "reviewer-a"
    assert record.payload["is_override"] is True
    assert record.payload["prediction_was_shown"] is True
    assert record.payload["reason_code"] == "wrong_class"


def test_claiming_a_task_marks_it_assigned(db_session: Session, wafer: Wafer) -> None:
    prediction = _prediction(db_session, wafer, 0.70, DefectPattern.SCRATCH)
    task = create_task(
        db_session,
        wafer=wafer,
        gate=TaskGate.CONFIRM,
        prediction=prediction,
        bands=BANDS,
        priority=1.0,
    )
    assert queue_depth(db_session, gate=TaskGate.CONFIRM) == 1

    claimed = claim(db_session, task.id, "reviewer-a")
    assert claimed is not None
    assert claimed.state is TaskState.ASSIGNED
    assert claimed.assigned_to == "reviewer-a"
    # No longer pending, so a second reviewer cannot be handed the same wafer.
    assert queue_depth(db_session, gate=TaskGate.CONFIRM) == 0
    assert claim(db_session, task.id, "reviewer-b") is None


def test_override_rate_splits_by_what_the_reviewer_saw(db_session: Session, wafer: Wafer) -> None:
    """The gap between the two rates is the anchoring effect."""
    shown = _prediction(db_session, wafer, 0.70, DefectPattern.SCRATCH)
    shown_task = create_task(
        db_session,
        wafer=wafer,
        gate=TaskGate.CONFIRM,
        prediction=shown,
        bands=BANDS,
        priority=1.0,
    )
    submit(
        db_session,
        DecisionRequest(
            task_id=shown_task.id,
            reviewer_id="reviewer-a",
            action=DecisionAction.ACCEPT,
            chosen_label=None,
            reason_code=None,
            note=None,
            decision_ms=1000,
        ),
        input_filter=FILTER,
        request_id="req-10",
    )

    blind = _prediction(db_session, wafer, 0.20, DefectPattern.SCRATCH)
    blind_task = create_task(
        db_session,
        wafer=wafer,
        gate=TaskGate.ESCALATION,
        prediction=blind,
        bands=BANDS,
        priority=1.0,
    )
    submit(
        db_session,
        DecisionRequest(
            task_id=blind_task.id,
            reviewer_id="reviewer-a",
            action=DecisionAction.EDIT,
            chosen_label=DefectPattern.DONUT,
            reason_code="hypothesis_unsupported",
            note=None,
            decision_ms=2000,
        ),
        input_filter=FILTER,
        request_id="req-11",
    )

    assert override_rate(db_session) == pytest.approx(0.5)
    assert override_rate(db_session, shown_only=True) == pytest.approx(0.0)


def test_decisions_are_the_training_signal(db_session: Session, wafer: Wafer) -> None:
    """A decision must land in a form the next round can consume."""
    prediction = _prediction(db_session, wafer, 0.70, DefectPattern.SCRATCH)
    task = create_task(
        db_session,
        wafer=wafer,
        gate=TaskGate.CONFIRM,
        prediction=prediction,
        bands=BANDS,
        priority=1.0,
    )
    submit(
        db_session,
        DecisionRequest(
            task_id=task.id,
            reviewer_id="reviewer-a",
            action=DecisionAction.EDIT,
            chosen_label=DefectPattern.EDGE_RING,
            reason_code="wrong_class",
            note=None,
            decision_ms=2600,
        ),
        input_filter=FILTER,
        request_id="req-12",
    )
    stored = db_session.execute(select(Decision)).scalar_one()
    assert stored.chosen_label is DefectPattern.EDGE_RING
    assert stored.decision_ms == 2600
