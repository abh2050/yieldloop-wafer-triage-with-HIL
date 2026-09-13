"""Review queue state.

The queue is where the three gates become work. A task exists because the system
decided a human should look at something, and it carries *why* -- which gate
produced it, and whether the reviewer is allowed to see the model's prediction.

The ``show_prediction`` flag is set here, from the routing band, and is not a
display preference. Below the confidence floor the console must not render the
prediction at all, and putting that decision in the queue rather than in the
frontend means a UI change cannot quietly start anchoring reviewers.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from yieldloop.db.enums import DefectPattern, RoutingBand, TaskGate, TaskState
from yieldloop.db.models import Prediction, ReviewTask, Wafer
from yieldloop.guardrails.thresholds import RoutingBands, classify


@dataclass(frozen=True, slots=True)
class QueueItem:
    """One unit of review work, with only what the reviewer may see.

    The prediction fields are ``None`` whenever ``show_prediction`` is false.
    Withholding at construction rather than at render time means there is no path
    by which a template, a debug view, or an API consumer can reach a prediction
    the routing policy said to hide.
    """

    task_id: UUID
    wafer_id: str
    lot_name: str
    gate: TaskGate
    priority: float
    show_prediction: bool
    grid_height: int
    grid_width: int
    die_total: int
    die_fail: int
    predicted_label: str | None
    confidence: float | None
    routing_band: RoutingBand | None

    @property
    def failure_rate(self) -> float:
        return self.die_fail / self.die_total if self.die_total else 0.0


def pending_query(gate: TaskGate | None = None) -> Select[tuple[ReviewTask]]:
    """Pending tasks, most informative first."""
    statement = select(ReviewTask).where(ReviewTask.state == TaskState.PENDING)
    if gate is not None:
        statement = statement.where(ReviewTask.gate == gate)
    return statement.order_by(ReviewTask.priority.desc(), ReviewTask.created_at)


def queue_depth(session: Session, gate: TaskGate | None = None) -> int:
    statement = (
        select(func.count()).select_from(ReviewTask).where(ReviewTask.state == TaskState.PENDING)
    )
    if gate is not None:
        statement = statement.where(ReviewTask.gate == gate)
    return int(session.execute(statement).scalar_one())


def create_task(
    session: Session,
    *,
    wafer: Wafer,
    gate: TaskGate,
    prediction: Prediction | None,
    bands: RoutingBands,
    priority: float,
    round_id: UUID | None = None,
) -> ReviewTask:
    """Create one review task, deriving visibility from the routing band.

    A label-gate task never shows a prediction regardless of confidence: the
    whole point of that gate is an unanchored human label on a wafer the model
    finds informative.
    """
    if gate is TaskGate.LABEL or prediction is None:
        show_prediction = False
    else:
        show_prediction = classify(prediction.confidence, bands).show_prediction

    task = ReviewTask(
        wafer_id=wafer.id,
        round_id=round_id,
        prediction_id=prediction.id if prediction is not None else None,
        gate=gate,
        state=TaskState.PENDING,
        priority=priority,
        show_prediction=show_prediction,
    )
    session.add(task)
    session.flush()
    return task


def next_items(
    session: Session, *, gate: TaskGate | None = None, limit: int = 25
) -> list[QueueItem]:
    """Fetch the next tasks for a reviewer.

    One query with joins rather than per-task lookups: the reviewer-throughput
    claim is measured in seconds per decision, and an N+1 here would show up
    directly in it.
    """
    if limit < 1:
        raise ValueError(f"limit must be positive; got {limit}")

    statement = (
        select(ReviewTask, Wafer, Prediction)
        .join(Wafer, ReviewTask.wafer_id == Wafer.id)
        .outerjoin(Prediction, ReviewTask.prediction_id == Prediction.id)
        .where(ReviewTask.state == TaskState.PENDING)
        .order_by(ReviewTask.priority.desc(), ReviewTask.created_at)
        .limit(limit)
    )
    if gate is not None:
        statement = statement.where(ReviewTask.gate == gate)

    items: list[QueueItem] = []
    for task, wafer, prediction in session.execute(statement).all():
        visible = task.show_prediction and prediction is not None
        items.append(
            QueueItem(
                task_id=task.id,
                wafer_id=wafer.wafer_id,
                lot_name=wafer.wafer_id.rsplit("-", 1)[0],
                gate=task.gate,
                priority=task.priority,
                show_prediction=task.show_prediction,
                grid_height=wafer.grid_height,
                grid_width=wafer.grid_width,
                die_total=wafer.die_total,
                die_fail=wafer.die_fail,
                predicted_label=prediction.predicted_label.value if visible else None,
                confidence=prediction.confidence if visible else None,
                routing_band=prediction.routing_band if visible else None,
            )
        )
    return items


def claim(session: Session, task_id: UUID, reviewer_id: str) -> ReviewTask | None:
    """Assign a pending task to a reviewer.

    Uses ``SELECT ... FOR UPDATE SKIP LOCKED`` so two reviewers pulling at once
    cannot be handed the same wafer -- duplicate work is the cheapest way to
    destroy the throughput claim this system is built around.
    """
    task = session.execute(
        select(ReviewTask)
        .where(ReviewTask.id == task_id, ReviewTask.state == TaskState.PENDING)
        .with_for_update(skip_locked=True)
    ).scalar_one_or_none()
    if task is None:
        return None

    task.state = TaskState.ASSIGNED
    task.assigned_to = reviewer_id
    task.assigned_at = func.now()
    session.flush()
    return task


def claim_next(
    session: Session, reviewer_id: str, *, gate: TaskGate | None = None
) -> ReviewTask | None:
    """Claim the highest-priority pending task."""
    statement = (
        select(ReviewTask)
        .where(ReviewTask.state == TaskState.PENDING)
        .order_by(ReviewTask.priority.desc(), ReviewTask.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    if gate is not None:
        statement = statement.where(ReviewTask.gate == gate)

    task = session.execute(statement).scalar_one_or_none()
    if task is None:
        return None
    task.state = TaskState.ASSIGNED
    task.assigned_to = reviewer_id
    task.assigned_at = func.now()
    session.flush()
    return task


# ---------------------------------------------------------------------------
# Gate producers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoutingOutcome:
    """What a routing pass did with a batch of predictions."""

    considered: int
    auto_committed: int
    queued_for_confirmation: int
    already_queued: int

    @property
    def automation_rate(self) -> float:
        return self.auto_committed / self.considered if self.considered else 0.0


def enqueue_confirmations(
    session: Session,
    *,
    bands: RoutingBands,
    limit: int = 200,
    artifact_id: UUID | None = None,
) -> RoutingOutcome:
    """Create confirm-gate tasks for predictions that need a human.

    This is the producer for the second gate. Predictions at or above the
    auto-commit threshold are left alone -- that is what auto-commit means -- and
    everything below it becomes a review task whose ``show_prediction`` follows
    the band, so a prediction below the floor is withheld from the reviewer
    without the console having to decide that.

    Idempotent. A wafer already waiting at this gate is skipped rather than
    queued twice, since two reviewers confirming the same prediction is
    duplicated effort rather than a second opinion.
    """
    statement = (
        select(Prediction, Wafer)
        .join(Wafer, Prediction.wafer_id == Wafer.id)
        .order_by(Prediction.entropy.desc())
        .limit(limit)
    )
    if artifact_id is not None:
        statement = statement.where(Prediction.artifact_id == artifact_id)

    considered = 0
    committed = 0
    queued = 0
    skipped = 0

    for prediction, wafer in session.execute(statement).all():
        considered += 1
        decision = classify(prediction.confidence, bands)
        if decision.is_auto_committed:
            committed += 1
            continue

        existing = session.execute(
            select(ReviewTask.id).where(
                ReviewTask.wafer_id == wafer.id,
                ReviewTask.gate == TaskGate.CONFIRM,
                ReviewTask.state.in_((TaskState.PENDING, TaskState.ASSIGNED)),
            )
        ).scalar_one_or_none()
        if existing is not None:
            skipped += 1
            continue

        create_task(
            session,
            wafer=wafer,
            gate=TaskGate.CONFIRM,
            prediction=prediction,
            bands=bands,
            # Most uncertain first: the reviewer's time is worth most where the
            # model is least sure.
            priority=prediction.entropy,
        )
        queued += 1

    session.flush()
    return RoutingOutcome(
        considered=considered,
        auto_committed=committed,
        queued_for_confirmation=queued,
        already_queued=skipped,
    )


def enqueue_escalations(
    session: Session,
    *,
    bands: RoutingBands,
    limit: int = 25,
    min_wafers: int = 2,
) -> int:
    """Create escalation-gate tasks for lots that look like an excursion.

    The producer for the third gate. A lot is escalated when several of its
    wafers carry a non-``none`` prediction the model is confident enough to
    stand behind -- one odd wafer is noise, a pattern across a lot is a lot-level
    cause worth asking about.

    Returns the number of lots escalated. The task is attached to the lot's
    most-failed wafer, which is the one an engineer opens first.
    """
    counts = (
        select(
            Wafer.lot_id.label("lot_id"),
            func.count().label("flagged"),
            func.max(Prediction.confidence).label("top_confidence"),
        )
        .select_from(Prediction)
        .join(Wafer, Prediction.wafer_id == Wafer.id)
        .where(
            Prediction.predicted_label != DefectPattern.NONE,
            Prediction.confidence >= bands.confidence_floor,
        )
        .group_by(Wafer.lot_id)
        .having(func.count() >= min_wafers)
        .order_by(func.count().desc())
        .limit(limit)
        .subquery()
    )

    escalated = 0
    for row in session.execute(select(counts)).all():
        wafer = session.execute(
            select(Wafer).where(Wafer.lot_id == row.lot_id).order_by(Wafer.die_fail.desc()).limit(1)
        ).scalar_one_or_none()
        if wafer is None:
            continue

        existing = session.execute(
            select(ReviewTask.id).where(
                ReviewTask.wafer_id == wafer.id,
                ReviewTask.gate == TaskGate.ESCALATION,
                ReviewTask.state.in_((TaskState.PENDING, TaskState.ASSIGNED)),
            )
        ).scalar_one_or_none()
        if existing is not None:
            continue

        prediction = session.execute(
            select(Prediction).where(Prediction.wafer_id == wafer.id).limit(1)
        ).scalar_one_or_none()

        create_task(
            session,
            wafer=wafer,
            gate=TaskGate.ESCALATION,
            prediction=prediction,
            bands=bands,
            priority=float(row.flagged),
        )
        escalated += 1

    session.flush()
    return escalated
