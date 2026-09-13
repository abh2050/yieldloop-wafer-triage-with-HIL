"""Liveness, readiness, and model health."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import func, select

from yieldloop.api.deps import SessionDep, SettingsDep
from yieldloop.db.enums import ArtifactKind, TaskState
from yieldloop.db.models import Decision, DriftSnapshot, ModelArtifact, ReviewTask, Wafer
from yieldloop.db.session import check_connectivity
from yieldloop.review.decisions import override_rate

router = APIRouter(prefix="/health", tags=["health"])


class LivenessResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    database: bool
    wafers: int
    active_classifier: str | None


class ModelHealthResponse(BaseModel):
    """What the ModelHealth screen renders."""

    active_classifier: str | None
    label_count: int | None
    metrics: dict[str, Any]
    routing: dict[str, float]
    queue_depth: int
    decisions: int
    override_rate: float
    override_rate_when_shown: float
    latest_drift: dict[str, Any] | None


@router.get("/live", response_model=LivenessResponse)
def live() -> LivenessResponse:
    """Process is up. Deliberately touches nothing else."""
    return LivenessResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse)
def ready(session: SessionDep) -> ReadinessResponse:
    """Dependencies are reachable and the database holds real data."""
    database_ok = check_connectivity()
    wafers = int(session.execute(select(func.count()).select_from(Wafer)).scalar_one())
    active = session.execute(
        select(ModelArtifact.content_hash).where(
            ModelArtifact.kind == ArtifactKind.CLASSIFIER,
            ModelArtifact.is_active.is_(True),
        )
    ).scalar_one_or_none()
    return ReadinessResponse(
        status="ok" if database_ok and wafers > 0 else "degraded",
        database=database_ok,
        wafers=wafers,
        active_classifier=active,
    )


@router.get("/model", response_model=ModelHealthResponse)
def model_health(session: SessionDep, settings: SettingsDep) -> ModelHealthResponse:
    """Calibration, routing configuration, and reviewer agreement."""
    artifact = session.execute(
        select(ModelArtifact).where(
            ModelArtifact.kind == ArtifactKind.CLASSIFIER,
            ModelArtifact.is_active.is_(True),
        )
    ).scalar_one_or_none()

    snapshot = session.execute(
        select(DriftSnapshot).order_by(DriftSnapshot.window_end.desc()).limit(1)
    ).scalar_one_or_none()

    return ModelHealthResponse(
        active_classifier=artifact.content_hash if artifact else None,
        label_count=artifact.label_count if artifact else None,
        metrics=dict(artifact.metrics) if artifact else {},
        routing={
            "confidence_floor": settings.confidence_floor,
            "auto_commit_threshold": settings.auto_commit_threshold,
        },
        queue_depth=int(
            session.execute(
                select(func.count())
                .select_from(ReviewTask)
                .where(ReviewTask.state == TaskState.PENDING)
            ).scalar_one()
        ),
        decisions=int(session.execute(select(func.count()).select_from(Decision)).scalar_one()),
        override_rate=override_rate(session),
        override_rate_when_shown=override_rate(session, shown_only=True),
        latest_drift=(
            {
                "window_start": snapshot.window_start.isoformat(),
                "window_end": snapshot.window_end.isoformat(),
                "override_rate": snapshot.override_rate,
                "expected_calibration_error": snapshot.expected_calibration_error,
                "input_psi": snapshot.input_psi,
                "decision_count": snapshot.decision_count,
            }
            if snapshot
            else None
        ),
    )
