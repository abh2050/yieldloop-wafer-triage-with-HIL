"""Reviewer agreement, and the anchoring effect.

Override rate alone is ambiguous: a low rate can mean the model is good or that
reviewers are deferring to it. Splitting by whether the prediction was visible
separates the two, which is why ``decisions.prediction_was_shown`` exists.

A large positive anchoring delta -- reviewers disagreeing far more often when
they could not see the prediction -- means the visible predictions are buying
agreement rather than earning it, and the confidence floor should rise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DecisionObservation:
    """One human decision, as recorded."""

    reviewer_id: str
    model_label: str | None
    chosen_label: str | None
    prediction_was_shown: bool
    is_override: bool
    decision_ms: int
    reason_code: str | None


@dataclass(frozen=True, slots=True)
class AgreementReport:
    decisions: int
    with_model_label: int
    overrides: int
    shown_decisions: int
    shown_overrides: int
    blind_decisions: int
    blind_overrides: int
    median_decision_ms: float
    p95_decision_ms: float
    reason_code_counts: dict[str, int]

    @property
    def override_rate(self) -> float:
        return self.overrides / self.with_model_label if self.with_model_label else 0.0

    @property
    def override_rate_shown(self) -> float:
        return self.shown_overrides / self.shown_decisions if self.shown_decisions else 0.0

    @property
    def override_rate_blind(self) -> float:
        return self.blind_overrides / self.blind_decisions if self.blind_decisions else 0.0

    @property
    def anchoring_delta(self) -> float:
        """Blind override rate minus shown override rate.

        Positive means reviewers disagree more when the prediction is hidden,
        which is the signature of anchoring. Only meaningful when both
        populations are non-empty.
        """
        if not self.shown_decisions or not self.blind_decisions:
            return 0.0
        return self.override_rate_blind - self.override_rate_shown

    @property
    def median_decision_seconds(self) -> float:
        return self.median_decision_ms / 1000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "with_model_label": self.with_model_label,
            "override_rate": self.override_rate,
            "override_rate_shown": self.override_rate_shown,
            "override_rate_blind": self.override_rate_blind,
            "anchoring_delta": self.anchoring_delta,
            "shown_decisions": self.shown_decisions,
            "blind_decisions": self.blind_decisions,
            "median_decision_ms": self.median_decision_ms,
            "median_decision_seconds": self.median_decision_seconds,
            "p95_decision_ms": self.p95_decision_ms,
            "reason_code_counts": self.reason_code_counts,
        }


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-len(ordered) * fraction // 1))))
    return float(ordered[rank - 1])


def summarize(observations: list[DecisionObservation]) -> AgreementReport:
    if not observations:
        return AgreementReport(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, {})

    with_model = [o for o in observations if o.model_label is not None]
    shown = [o for o in with_model if o.prediction_was_shown]
    blind = [o for o in with_model if not o.prediction_was_shown]

    reason_counts: dict[str, int] = {}
    for observation in observations:
        if observation.reason_code:
            reason_counts[observation.reason_code] = (
                reason_counts.get(observation.reason_code, 0) + 1
            )

    durations = [o.decision_ms for o in observations]
    return AgreementReport(
        decisions=len(observations),
        with_model_label=len(with_model),
        overrides=sum(1 for o in with_model if o.is_override),
        shown_decisions=len(shown),
        shown_overrides=sum(1 for o in shown if o.is_override),
        blind_decisions=len(blind),
        blind_overrides=sum(1 for o in blind if o.is_override),
        median_decision_ms=_percentile(durations, 0.5),
        p95_decision_ms=_percentile(durations, 0.95),
        reason_code_counts=reason_counts,
    )
