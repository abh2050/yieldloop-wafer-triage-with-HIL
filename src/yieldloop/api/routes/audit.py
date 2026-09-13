"""Read-only access to the audit trail.

There is no write endpoint and there never will be. Records are written by the
services that produce the events, and the table itself refuses mutation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import desc, select

from yieldloop.api.deps import SessionDep
from yieldloop.db.enums import AuditEventType, GuardrailStage
from yieldloop.db.models import AuditRecord, GuardrailAction
from yieldloop.guardrails.audit import AuditLog

router = APIRouter(prefix="/audit", tags=["audit"])


class AuditRecordResponse(BaseModel):
    sequence: int
    occurred_at: datetime
    event_type: AuditEventType
    actor: str
    subject_type: str
    subject_id: str
    request_id: str
    payload: dict[str, Any]
    digest: str


class AuditPageResponse(BaseModel):
    records: list[AuditRecordResponse]
    total: int


class GuardrailActionResponse(BaseModel):
    id: UUID
    created_at: datetime
    request_id: str
    stage: GuardrailStage
    outcome: str
    reason: str
    detail: dict[str, Any]


class ChainVerificationResponse(BaseModel):
    """Whether the hash chain is intact.

    Surfaced in the console because an audit log nobody checks is a filing
    cabinet, not a control.
    """

    checked: int
    intact: bool
    broken_at: list[int]


@router.get("", response_model=AuditPageResponse)
def list_records(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    event_type: Annotated[AuditEventType | None, Query()] = None,
) -> AuditPageResponse:
    statement = select(AuditRecord).order_by(desc(AuditRecord.sequence)).limit(limit)
    if event_type is not None:
        statement = statement.where(AuditRecord.event_type == event_type)
    records = session.execute(statement).scalars().all()
    return AuditPageResponse(
        records=[
            AuditRecordResponse(
                sequence=r.sequence,
                occurred_at=r.occurred_at,
                event_type=r.event_type,
                actor=r.actor,
                subject_type=r.subject_type,
                subject_id=r.subject_id,
                request_id=r.request_id,
                payload=dict(r.payload),
                digest=r.digest,
            )
            for r in records
        ],
        total=len(records),
    )


@router.get("/guardrails", response_model=list[GuardrailActionResponse])
def list_guardrail_actions(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    stage: Annotated[GuardrailStage | None, Query()] = None,
) -> list[GuardrailActionResponse]:
    statement = (
        select(GuardrailAction).order_by(desc(GuardrailAction.created_at)).limit(limit)
    )
    if stage is not None:
        statement = statement.where(GuardrailAction.stage == stage)
    return [
        GuardrailActionResponse(
            id=a.id,
            created_at=a.created_at,
            request_id=a.request_id,
            stage=a.stage,
            outcome=a.outcome.value,
            reason=a.reason,
            detail=dict(a.detail),
        )
        for a in session.execute(statement).scalars().all()
    ]


@router.get("/verify", response_model=ChainVerificationResponse)
def verify_chain(session: SessionDep) -> ChainVerificationResponse:
    result = AuditLog(session).verify_chain()
    return ChainVerificationResponse(
        checked=result.checked, intact=result.intact, broken_at=list(result.broken_at)
    )
