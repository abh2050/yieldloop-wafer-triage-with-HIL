"""Writing human decisions.

Every decision is training signal for the next round, which sets the bar for what
has to be captured. A label alone is not enough: the same label means different
things depending on whether the reviewer could see the model's guess, how long
they took, and why they disagreed. All of that is recorded, because the
override-rate and anchoring analyses are impossible to reconstruct afterwards if
it is not.

Decisions are append-only in practice. A reviewer who changes their mind produces
a new task, not an edited record, so the audit trail reflects what was actually
decided at the time.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from yieldloop.db.enums import (
    DecisionAction,
    DefectPattern,
    LabelSource,
    TaskState,
)
from yieldloop.db.models import Decision, Prediction, ReviewTask
from yieldloop.guardrails import InputRejectedError
from yieldloop.guardrails.audit import AuditLog
from yieldloop.guardrails.input_filter import InputFilter, validate_reviewer_id
from yieldloop.review.reason_codes import is_valid


class DecisionError(InputRejectedError):
    """A decision could not be accepted as submitted."""

    reason = "decision_rejected"


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """What a reviewer submitted."""

    task_id: UUID
    reviewer_id: str
    action: DecisionAction
    chosen_label: DefectPattern | None
    reason_code: str | None
    note: str | None
    decision_ms: int


def submit(
    session: Session,
    request: DecisionRequest,
    *,
    input_filter: InputFilter,
    request_id: str,
) -> Decision:
    """Validate and record one human decision.

    Raises:
        DecisionError: if the task is missing, already decided, or the submission
            is internally inconsistent.
    """
    reviewer_id = validate_reviewer_id(request.reviewer_id)
    note = input_filter.check_note(request.note, field="note")

    if request.decision_ms < 0:
        raise DecisionError(
            f"decision_ms must be non-negative; got {request.decision_ms}",
            detail={"code": "negative_duration"},
        )

    task = session.get(ReviewTask, request.task_id)
    if task is None:
        raise DecisionError(
            f"no review task {request.task_id}", detail={"code": "unknown_task"}
        )
    if task.state is TaskState.COMPLETED:
        raise DecisionError(
            f"task {request.task_id} already has a decision",
            detail={"code": "already_decided"},
        )

    if request.action is not DecisionAction.ACCEPT and request.reason_code is None:
        raise DecisionError(
            f"a reason code is required for a {request.action.value} decision",
            detail={"code": "reason_code_required"},
        )
    if request.reason_code is not None and not is_valid(
        request.reason_code, task.gate, request.action
    ):
        raise DecisionError(
            f"reason code {request.reason_code!r} is not valid for a "
            f"{request.action.value} at the {task.gate.value} gate",
            detail={"code": "invalid_reason_code"},
        )

    prediction = (
        session.get(Prediction, task.prediction_id)
        if task.prediction_id is not None
        else None
    )

    # An accept at a gate that showed a prediction means the reviewer agreed with
    # it, so the model's label is the chosen one. An accept where nothing was
    # shown is meaningless and is refused rather than silently recorded as
    # agreement with something invisible.
    chosen = request.chosen_label
    if request.action is DecisionAction.ACCEPT:
        if chosen is None and prediction is not None and task.show_prediction:
            chosen = prediction.predicted_label
        elif chosen is None:
            raise DecisionError(
                "an accept with no visible prediction must still carry a label",
                detail={"code": "label_required"},
            )

    model_label = prediction.predicted_label if prediction is not None else None
    is_override = bool(
        model_label is not None and chosen is not None and chosen != model_label
    )

    decision = Decision(
        task_id=task.id,
        wafer_id=task.wafer_id,
        reviewer_id=reviewer_id,
        action=request.action,
        chosen_label=chosen,
        model_label=model_label,
        model_confidence=prediction.confidence if prediction is not None else None,
        is_override=is_override,
        prediction_was_shown=task.show_prediction,
        reason_code=request.reason_code,
        note=note,
        source=LabelSource.REVIEWER,
        decision_ms=request.decision_ms,
    )
    session.add(decision)

    task.state = TaskState.COMPLETED
    session.flush()

    AuditLog(session).record_human_decision(
        reviewer_id=reviewer_id,
        request_id=request_id,
        decision_id=str(decision.id),
        payload={
            "task_id": str(task.id),
            "gate": task.gate.value,
            "action": request.action.value,
            "chosen_label": chosen.value if chosen else None,
            "model_label": model_label.value if model_label else None,
            "is_override": is_override,
            "prediction_was_shown": task.show_prediction,
            "reason_code": request.reason_code,
            "decision_ms": request.decision_ms,
        },
    )
    session.flush()
    return decision


def override_rate(session: Session, *, shown_only: bool = False) -> float:
    """Fraction of decisions that disagreed with the model.

    ``shown_only`` restricts to decisions where the prediction was visible. The
    gap between the two figures is the anchoring effect, which is why
    ``prediction_was_shown`` is recorded on every row.
    """
    statement = select(Decision.is_override).where(Decision.model_label.is_not(None))
    if shown_only:
        statement = statement.where(Decision.prediction_was_shown.is_(True))
    values = session.execute(statement).scalars().all()
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def training_labels(session: Session) -> list[tuple[str, DefectPattern]]:
    """Reviewer-produced labels, in the form the next training round consumes."""
    rows = session.execute(
        select(Decision).where(Decision.chosen_label.is_not(None))
    ).scalars().all()
    return [(str(row.wafer_id), row.chosen_label) for row in rows if row.chosen_label]
