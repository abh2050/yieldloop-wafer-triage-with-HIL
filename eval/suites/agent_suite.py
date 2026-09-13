"""Agent evaluation against real lots and real reviewer resolutions.

Precision at k is measured against causes a human actually accepted in this
console, never against a synthetic answer key. On a console with no decisions
yet, that metric is reported as unavailable rather than as zero: no data is not
the same as bad performance, and conflating them would let a fresh install look
like a broken agent.

Requires a live API key. Skipped, not stubbed, when one is absent -- a stubbed
agent would measure the stub.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from eval.datasets import load_reviewer_resolutions
from eval.metrics.grounding import GroundingObservation, precision_at_k, summarize
from yieldloop.agent.context import build_bundle
from yieldloop.agent.hypothesis import HypothesisService
from yieldloop.config import Settings
from yieldloop.db.enums import DefectPattern
from yieldloop.db.models import Lot
from yieldloop.ingest.normalize import PATTERN_TO_CAUSE
from yieldloop.logging import get_logger
from yieldloop.retrieval.context_builder import assemble_for_lot

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AgentSuiteResult:
    grounding: dict[str, Any]
    precision_at_1: float | None
    precision_at_3: float | None
    scored_lots: int
    lots_attempted: int
    total_cost_usd: float
    seconds: float
    #: Why precision is unavailable, when it is.
    precision_note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "grounding": self.grounding,
            "precision_at_1": self.precision_at_1,
            "precision_at_3": self.precision_at_3,
            "precision_note": self.precision_note,
            "scored_lots": self.scored_lots,
            "lots_attempted": self.lots_attempted,
            "total_cost_usd": self.total_cost_usd,
            "seconds": self.seconds,
        }


def run(session: Session, settings: Settings, *, max_lots: int = 10) -> AgentSuiteResult | None:
    """Run the agent over real lots. Returns None without an API key."""
    if not settings.openai_api_key.get_secret_value() and not os.environ.get("OPENAI_API_KEY"):
        logger.warning("agent_suite_skipped", reason="OPENAI_API_KEY is not set")
        return None

    started = time.monotonic()
    lots = session.execute(select(Lot).order_by(Lot.lot_ordinal).limit(max_lots)).scalars().all()
    if not lots:
        logger.warning("agent_suite_skipped", reason="no lots ingested")
        return None

    resolutions = load_reviewer_resolutions(session)
    accepted_by_lot: dict[str, str] = {}
    for resolution in resolutions:
        # The reviewer's chosen defect pattern maps to a cause under the same
        # documented convention the excursion corpus uses, so the agent is scored
        # against a human decision rather than against another model output.
        try:
            pattern = DefectPattern(resolution.chosen_label)
        except ValueError:
            continue
        accepted_by_lot[resolution.lot_name] = PATTERN_TO_CAUSE[pattern].value

    service = HypothesisService(session, settings)
    observations: list[GroundingObservation] = []
    ranked: list[list[str]] = []
    accepted: list[str | None] = []
    total_cost = 0.0

    for lot in lots:
        evidence, _ = assemble_for_lot(session, settings, lot)
        bundle = build_bundle(lot=lot, **evidence)
        result = service.generate(
            bundle=bundle, requested_by="eval-harness", session_id="eval-harness"
        )
        session.commit()

        total_cost += result.cost_usd
        observations.append(
            GroundingObservation(
                lot_name=lot.lot_name,
                returned=result.returned_count,
                grounded=result.grounded_count,
                abstained=result.abstained,
                injection_suspected=result.injection_suspected,
                latency_ms=result.latency_ms,
            )
        )
        ranked.append([h.cause_category.value for h in result.response.hypotheses])
        accepted.append(accepted_by_lot.get(lot.lot_name))

    scored = sum(1 for a in accepted if a is not None)
    note = (
        None
        if scored
        else (
            "No reviewer resolutions exist for the evaluated lots, so hypothesis "
            "precision cannot be measured. This is a data gap, not a score of zero."
        )
    )

    return AgentSuiteResult(
        grounding=summarize(observations).as_dict(),
        precision_at_1=precision_at_k(ranked, accepted, 1) if scored else None,
        precision_at_3=precision_at_k(ranked, accepted, 3) if scored else None,
        scored_lots=scored,
        lots_attempted=len(lots),
        total_cost_usd=total_cost,
        seconds=time.monotonic() - started,
        precision_note=note,
    )
