"""Reviewer agreement, from decisions actually captured by the console.

Separate from the classifier suite because it measures people, not the model.
Override rate alone is ambiguous -- a low rate can mean the model is good or that
reviewers are deferring to it -- so it is split by whether the prediction was
visible, and the gap between the two is the anchoring effect.

Reports a gap rather than a zero on a console nobody has used yet. No decisions
is not evidence of perfect agreement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from eval.datasets import load_reviewer_resolutions
from eval.metrics.agreement import AgreementReport, DecisionObservation, summarize
from yieldloop.logging import get_logger

logger = get_logger(__name__)

#: Below this, rates are reported but flagged. A rate from a handful of
#: decisions is noise, and acting on it is worse than having no number.
MIN_DECISIONS_FOR_CONFIDENCE = 30


@dataclass(frozen=True, slots=True)
class ReviewSuiteResult:
    report: AgreementReport
    thin: bool
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        payload = self.report.as_dict()
        payload["thin_sample"] = self.thin
        payload["min_decisions_for_confidence"] = MIN_DECISIONS_FOR_CONFIDENCE
        payload["seconds"] = self.seconds
        return payload


def run(session: Session) -> ReviewSuiteResult | None:
    """Summarize captured decisions. Returns None when there are none."""
    started = time.monotonic()
    resolutions = load_reviewer_resolutions(session)
    if not resolutions:
        logger.warning("review_suite_skipped", reason="no reviewer decisions captured")
        return None

    observations = [
        DecisionObservation(
            reviewer_id=r.wafer_id,
            model_label=r.model_label,
            chosen_label=r.chosen_label,
            prediction_was_shown=r.prediction_was_shown,
            is_override=r.is_override,
            decision_ms=r.decision_ms,
            reason_code=r.reason_code,
        )
        for r in resolutions
    ]
    report = summarize(observations)
    return ReviewSuiteResult(
        report=report,
        thin=report.decisions < MIN_DECISIONS_FOR_CONFIDENCE,
        seconds=time.monotonic() - started,
    )
