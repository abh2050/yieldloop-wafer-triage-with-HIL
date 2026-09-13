"""The escalation gate: ranked root cause hypotheses with inline evidence.

The handler does not call the model. It assembles a context bundle from retrieval
and hands it to the guarded service, which is the only path into the agent.
Nothing here can bypass a guardrail, because nothing here knows how.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from yieldloop.agent.context import build_bundle
from yieldloop.agent.hypothesis import HypothesisService
from yieldloop.api.deps import ReviewerDep, SessionDep, SettingsDep
from yieldloop.db.enums import CauseCategory
from yieldloop.db.models import Lot
from yieldloop.retrieval.context_builder import assemble_for_lot

router = APIRouter(prefix="/hypothesis", tags=["hypothesis"])


class CitationResponse(BaseModel):
    """An evidence reference the console renders as a clickable link.

    Every citation here resolved against the context bundle. The grounding gate
    dropped any that did not, so the frontend never has to handle a dead link.
    """

    evidence_id: str
    kind: str
    summary: str


class HypothesisResponseModel(BaseModel):
    rank: int
    cause_category: CauseCategory
    statement: str
    evidence_ids: list[str]
    supporting_signal: str
    contradicting_signal: str | None
    confidence: float
    confirming_query: str
    eliminating_query: str


class HypothesisRequestPayload(BaseModel):
    lot_name: str = Field(max_length=64)
    reviewer_note: str | None = Field(default=None, max_length=10_000)


class HypothesisEnvelope(BaseModel):
    """The complete guarded result.

    ``abstained`` is a first-class outcome, not an error. The console renders it
    as "insufficient evidence" with the reason shown.
    """

    request_id: str
    lot_id: str
    lot_name: str
    abstained: bool
    abstention_reason: str | None
    hypotheses: list[HypothesisResponseModel]
    citations: dict[str, CitationResponse]
    injection_suspected: bool
    context_gaps: list[str]
    degraded: bool
    returned_count: int
    grounded_count: int
    evidence_ids: list[str]


@router.post("", response_model=HypothesisEnvelope, status_code=201)
def generate(
    payload: HypothesisRequestPayload,
    session: SessionDep,
    settings: SettingsDep,
    reviewer_id: ReviewerDep,
) -> HypothesisEnvelope:
    lot = session.execute(select(Lot).where(Lot.lot_name == payload.lot_name)).scalar_one_or_none()
    if lot is None:
        raise HTTPException(status_code=404, detail=f"no lot {payload.lot_name}")

    evidence, descriptions = assemble_for_lot(session, settings, lot)
    bundle = build_bundle(lot=lot, **evidence)

    service = HypothesisService(session, settings)
    result = service.generate(
        bundle=bundle,
        requested_by=reviewer_id,
        session_id=reviewer_id,
        reviewer_note=payload.reviewer_note,
    )

    cited = {
        evidence_id
        for hypothesis in result.response.hypotheses
        for evidence_id in hypothesis.evidence_ids
    }
    return HypothesisEnvelope(
        request_id=result.request_id,
        lot_id=str(lot.id),
        lot_name=lot.lot_name,
        abstained=result.abstained,
        abstention_reason=result.response.abstention_reason,
        hypotheses=[
            HypothesisResponseModel(
                rank=h.rank,
                cause_category=h.cause_category,
                statement=h.statement,
                evidence_ids=list(h.evidence_ids),
                supporting_signal=h.supporting_signal,
                contradicting_signal=h.contradicting_signal,
                confidence=h.confidence,
                confirming_query=h.confirming_query,
                eliminating_query=h.eliminating_query,
            )
            for h in result.response.hypotheses
        ],
        citations={
            evidence_id: CitationResponse(**descriptions[evidence_id])
            for evidence_id in cited
            if evidence_id in descriptions
        },
        injection_suspected=result.injection_suspected,
        context_gaps=list(result.response.context_gaps),
        degraded=result.degraded,
        returned_count=result.returned_count,
        grounded_count=result.grounded_count,
        evidence_ids=sorted(bundle.evidence_ids),
    )


class LotSummaryResponse(BaseModel):
    lot_name: str
    lot_id: UUID
    wafer_count: int
    split: str
    evidence: dict[str, Any]


@router.get("/lot/{lot_name}", response_model=LotSummaryResponse)
def lot_summary(lot_name: str, session: SessionDep, settings: SettingsDep) -> LotSummaryResponse:
    """What evidence exists for a lot, before spending anything on the agent."""
    lot = session.execute(select(Lot).where(Lot.lot_name == lot_name)).scalar_one_or_none()
    if lot is None:
        raise HTTPException(status_code=404, detail=f"no lot {lot_name}")

    evidence, descriptions = assemble_for_lot(session, settings, lot)
    return LotSummaryResponse(
        lot_name=lot.lot_name,
        lot_id=lot.id,
        wafer_count=lot.wafer_count,
        split=lot.split.value,
        evidence={
            "classifier": len(evidence["classifier"]),
            "die_statistics": len(evidence["die_statistics"]),
            "similar_lots": len(evidence["similar_lots"]),
            "process_events": len(evidence["process_events"]),
            "evidence_ids": sorted(descriptions),
        },
    )
