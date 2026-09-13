"""The confirm gate: uncertainty-band predictions, and wafer detail."""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select

from yieldloop.api.deps import SessionDep, SettingsDep
from yieldloop.db.enums import RoutingBand, TaskGate
from yieldloop.db.models import Lot, Prediction, Wafer
from yieldloop.review import queue as queue_service

router = APIRouter(prefix="/triage", tags=["triage"])


class TriageItemResponse(BaseModel):
    task_id: UUID
    wafer_id: str
    lot_name: str
    priority: float
    show_prediction: bool
    die_total: int
    die_fail: int
    failure_rate: float
    predicted_label: str | None = None
    confidence: float | None = None
    routing_band: RoutingBand | None = None


class TriageQueueResponse(BaseModel):
    items: list[TriageItemResponse]
    depth: int
    confidence_floor: float
    auto_commit_threshold: float


class WaferDetailResponse(BaseModel):
    """Everything the detail screen renders for one wafer."""

    wafer_id: str
    lot_name: str
    split: str
    grid_height: int
    grid_width: int
    grid: list[int]
    die_total: int
    die_fail: int
    failure_rate: float
    dataset_label: str | None
    predicted_label: str | None
    confidence: float | None
    probabilities: dict[str, Any] | None
    routing_band: RoutingBand | None
    #: False below the confidence floor. The console must honour this.
    show_prediction: bool


@router.get("/queue", response_model=TriageQueueResponse)
def get_queue(
    session: SessionDep,
    settings: SettingsDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 25,
) -> TriageQueueResponse:
    items = queue_service.next_items(session, gate=TaskGate.CONFIRM, limit=limit)
    return TriageQueueResponse(
        items=[
            TriageItemResponse(
                task_id=item.task_id,
                wafer_id=item.wafer_id,
                lot_name=item.lot_name,
                priority=item.priority,
                show_prediction=item.show_prediction,
                die_total=item.die_total,
                die_fail=item.die_fail,
                failure_rate=item.failure_rate,
                predicted_label=item.predicted_label,
                confidence=item.confidence,
                routing_band=item.routing_band,
            )
            for item in items
        ],
        depth=queue_service.queue_depth(session, gate=TaskGate.CONFIRM),
        confidence_floor=settings.confidence_floor,
        auto_commit_threshold=settings.auto_commit_threshold,
    )


@router.get("/wafer/{wafer_id}", response_model=WaferDetailResponse)
def get_wafer(wafer_id: str, session: SessionDep, settings: SettingsDep) -> WaferDetailResponse:
    """One wafer in full.

    The prediction is withheld below the confidence floor here too. An endpoint
    that leaked it would let the detail view anchor a reviewer that the queue was
    careful not to.
    """
    row = session.execute(
        select(Wafer, Lot).join(Lot, Wafer.lot_id == Lot.id).where(Wafer.wafer_id == wafer_id)
    ).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no wafer {wafer_id}")
    wafer, lot = row

    prediction = session.execute(
        select(Prediction)
        .where(Prediction.wafer_id == wafer.id)
        .order_by(Prediction.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()

    show = prediction is not None and prediction.confidence >= settings.confidence_floor
    return WaferDetailResponse(
        wafer_id=wafer.wafer_id,
        lot_name=lot.lot_name,
        split=wafer.split.value,
        grid_height=wafer.grid_height,
        grid_width=wafer.grid_width,
        grid=list(wafer.grid),
        die_total=wafer.die_total,
        die_fail=wafer.die_fail,
        failure_rate=wafer.die_fail / wafer.die_total if wafer.die_total else 0.0,
        dataset_label=wafer.dataset_label.value if wafer.dataset_label else None,
        predicted_label=(
            prediction.predicted_label.value if prediction is not None and show else None
        ),
        confidence=prediction.confidence if prediction is not None and show else None,
        probabilities=(
            dict(prediction.probabilities) if prediction is not None and show else None
        ),
        routing_band=prediction.routing_band if prediction is not None else None,
        show_prediction=show,
    )
