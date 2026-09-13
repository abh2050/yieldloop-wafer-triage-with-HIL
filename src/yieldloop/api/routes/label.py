"""The label gate: the most informative unlabeled wafers, and decisions on them.

This is the endpoint the throughput claim rests on, so the queue response carries
everything the grid needs to render a wafer without a second round trip.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select

from yieldloop.api.deps import InputFilterDep, RequestIdDep, ReviewerDep, SessionDep
from yieldloop.db.enums import DecisionAction, DefectPattern, TaskGate
from yieldloop.db.models import Wafer
from yieldloop.review import queue as queue_service
from yieldloop.review.decisions import DecisionRequest, submit
from yieldloop.review.reason_codes import REASON_CODES, codes_for

router = APIRouter(prefix="/label", tags=["label"])


class QueueItemResponse(BaseModel):
    """One wafer awaiting a label.

    The prediction fields are absent whenever the routing policy said to withhold
    them. They are not merely hidden by the client.
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
    failure_rate: float
    predicted_label: str | None = None
    confidence: float | None = None


class QueueResponse(BaseModel):
    items: list[QueueItemResponse]
    depth: int


class ReasonCodeResponse(BaseModel):
    code: str
    label: str
    description: str


class DecisionPayload(BaseModel):
    """A reviewer's decision. Validated at the boundary, enforced in the service."""

    task_id: UUID
    action: DecisionAction
    chosen_label: DefectPattern | None = None
    reason_code: str | None = None
    note: str | None = Field(default=None, max_length=10_000)
    decision_ms: int = Field(ge=0, description="Presentation to submit, in milliseconds")


class DecisionResponse(BaseModel):
    decision_id: UUID
    task_id: UUID
    chosen_label: str | None
    is_override: bool
    prediction_was_shown: bool


@router.get("/queue", response_model=QueueResponse)
def get_queue(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 25,
) -> QueueResponse:
    items = queue_service.next_items(session, gate=TaskGate.LABEL, limit=limit)
    return QueueResponse(
        items=[
            QueueItemResponse(
                task_id=item.task_id,
                wafer_id=item.wafer_id,
                lot_name=item.lot_name,
                gate=item.gate,
                priority=item.priority,
                show_prediction=item.show_prediction,
                grid_height=item.grid_height,
                grid_width=item.grid_width,
                die_total=item.die_total,
                die_fail=item.die_fail,
                failure_rate=item.failure_rate,
                predicted_label=item.predicted_label,
                confidence=item.confidence,
            )
            for item in items
        ],
        depth=queue_service.queue_depth(session, gate=TaskGate.LABEL),
    )


@router.get("/reason-codes", response_model=list[ReasonCodeResponse])
def get_reason_codes(
    gate: Annotated[TaskGate, Query()] = TaskGate.LABEL,
    action: Annotated[DecisionAction, Query()] = DecisionAction.EDIT,
) -> list[ReasonCodeResponse]:
    specs = codes_for(gate, action) or list(REASON_CODES)
    return [
        ReasonCodeResponse(code=s.code, label=s.label, description=s.description) for s in specs
    ]


@router.get("/wafer/{wafer_id}/grid", response_model=list[int])
def get_wafer_grid(wafer_id: str, session: SessionDep) -> list[int]:
    """The normalized wafer map as a flat array of 0/1/2.

    Served as the real array rather than a rendered image so the console draws it
    on canvas from the same bytes the classifier consumed.
    """
    wafer = session.execute(select(Wafer).where(Wafer.wafer_id == wafer_id)).scalar_one_or_none()
    if wafer is None:
        raise HTTPException(status_code=404, detail=f"no wafer {wafer_id}")
    return list(wafer.grid)


@router.post("/decision", response_model=DecisionResponse, status_code=201)
def post_decision(
    payload: DecisionPayload,
    session: SessionDep,
    reviewer_id: ReviewerDep,
    input_filter: InputFilterDep,
    request_id: RequestIdDep,
) -> DecisionResponse:
    decision = submit(
        session,
        DecisionRequest(
            task_id=payload.task_id,
            reviewer_id=reviewer_id,
            action=payload.action,
            chosen_label=payload.chosen_label,
            reason_code=payload.reason_code,
            note=payload.note,
            decision_ms=payload.decision_ms,
        ),
        input_filter=input_filter,
        request_id=request_id,
    )
    return DecisionResponse(
        decision_id=decision.id,
        task_id=decision.task_id,
        chosen_label=decision.chosen_label.value if decision.chosen_label else None,
        is_override=decision.is_override,
        prediction_was_shown=decision.prediction_was_shown,
    )
