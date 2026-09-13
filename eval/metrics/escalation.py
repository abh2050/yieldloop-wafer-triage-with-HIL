"""Escalation and reviewer-load metrics.

How much human time the system asks for, and what it buys. A triage console that
routes everything to a human is safe and useless; one that routes nothing is fast
and dangerous. These numbers locate where a given configuration sits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EscalationReport:
    total_predictions: int
    auto_committed: int
    routed_to_human: int
    shown_prediction: int
    withheld_prediction: int
    #: Errors that were auto-committed and so never seen by a human.
    escaped_errors: int

    @property
    def automation_rate(self) -> float:
        return self.auto_committed / self.total_predictions if self.total_predictions else 0.0

    @property
    def escalation_rate(self) -> float:
        return self.routed_to_human / self.total_predictions if self.total_predictions else 0.0

    @property
    def blind_review_share(self) -> float:
        """Share of human work where the prediction was withheld.

        Not a cost to minimize: these are the cases where showing a weak
        prediction would anchor the reviewer, and the labels collected here are
        the most valuable ones the system gets.
        """
        return self.withheld_prediction / self.routed_to_human if self.routed_to_human else 0.0

    @property
    def escaped_error_rate(self) -> float:
        return self.escaped_errors / self.total_predictions if self.total_predictions else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_predictions": self.total_predictions,
            "auto_committed": self.auto_committed,
            "routed_to_human": self.routed_to_human,
            "shown_prediction": self.shown_prediction,
            "withheld_prediction": self.withheld_prediction,
            "automation_rate": self.automation_rate,
            "escalation_rate": self.escalation_rate,
            "blind_review_share": self.blind_review_share,
            "escaped_errors": self.escaped_errors,
            "escaped_error_rate": self.escaped_error_rate,
        }


def summarize(
    confidences: list[float],
    correct: list[bool],
    *,
    confidence_floor: float,
    auto_commit_threshold: float,
) -> EscalationReport:
    if len(confidences) != len(correct):
        raise ValueError(f"{len(confidences)} confidences against {len(correct)} outcomes")
    if not 0.0 < confidence_floor < auto_commit_threshold <= 1.0:
        raise ValueError("require 0 < confidence_floor < auto_commit_threshold <= 1")

    committed = 0
    shown = 0
    withheld = 0
    escaped = 0
    for confidence, is_correct in zip(confidences, correct, strict=True):
        if confidence >= auto_commit_threshold:
            committed += 1
            if not is_correct:
                escaped += 1
        elif confidence >= confidence_floor:
            shown += 1
        else:
            withheld += 1

    return EscalationReport(
        total_predictions=len(confidences),
        auto_committed=committed,
        routed_to_human=shown + withheld,
        shown_prediction=shown,
        withheld_prediction=withheld,
        escaped_errors=escaped,
    )
