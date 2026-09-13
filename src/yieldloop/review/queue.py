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

from yieldloop.db.enums import RoutingBand, TaskGate, TaskState
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
