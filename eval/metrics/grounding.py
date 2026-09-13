"""Agent grounding metrics.

Grounding rate is the headline: the fraction of proposed hypotheses whose
citations all resolved to evidence actually in the context bundle. A rate below
1.0 does not mean the system failed -- the gate dropped those claims and they
never reached a reviewer -- it means the model attempted a fabrication, and the
trend in that number is worth watching.

Abstention rate is reported next to it, and deliberately not minimized.
Abstaining on a thin bundle is correct behaviour; an abstention rate of zero
against sparse evidence would mean the model is inventing support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GroundingObservation:
    """One guarded agent call."""

    lot_name: str
    returned: int
    grounded: int
    abstained: bool
    injection_suspected: bool
    latency_ms: float
    #: Evidence ids cited that were not in the bundle. Direct evidence of
    #: attempted fabrication.
    unresolvable_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GroundingReport:
    calls: int
    hypotheses_returned: int
    hypotheses_grounded: int
    abstentions: int
    injection_flags: int
    fabricated_citations: int
    median_latency_ms: float
    p95_latency_ms: float

    @property
    def grounding_rate(self) -> float:
        """Fraction of proposed hypotheses that survived the gate."""
        if self.hypotheses_returned == 0:
            return 1.0
        return self.hypotheses_grounded / self.hypotheses_returned

    @property
    def abstention_rate(self) -> float:
        return self.abstentions / self.calls if self.calls else 0.0

    @property
    def fabrication_rate(self) -> float:
        """Calls in which at least one citation did not resolve."""
        if self.hypotheses_returned == 0:
            return 0.0
        return self.fabricated_citations / self.hypotheses_returned

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "hypotheses_returned": self.hypotheses_returned,
            "hypotheses_grounded": self.hypotheses_grounded,
            "grounding_rate": self.grounding_rate,
            "abstentions": self.abstentions,
            "abstention_rate": self.abstention_rate,
            "injection_flags": self.injection_flags,
            "fabricated_citations": self.fabricated_citations,
            "fabrication_rate": self.fabrication_rate,
            "median_latency_ms": self.median_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
        }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-len(ordered) * fraction // 1))))
    return ordered[rank - 1]


def summarize(observations: list[GroundingObservation]) -> GroundingReport:
    if not observations:
        return GroundingReport(0, 0, 0, 0, 0, 0, 0.0, 0.0)

    latencies = [o.latency_ms for o in observations]
    return GroundingReport(
        calls=len(observations),
        hypotheses_returned=sum(o.returned for o in observations),
        hypotheses_grounded=sum(o.grounded for o in observations),
        abstentions=sum(1 for o in observations if o.abstained),
        injection_flags=sum(1 for o in observations if o.injection_suspected),
        fabricated_citations=sum(len(o.unresolvable_ids) for o in observations),
        median_latency_ms=_percentile(latencies, 0.5),
        p95_latency_ms=_percentile(latencies, 0.95),
    )


def precision_at_k(
    ranked_causes: list[list[str]], accepted_causes: list[str | None], k: int
) -> float:
    """Fraction of calls whose top-k contained the cause the human accepted.

    Measured against real reviewer resolutions, not against a synthetic answer
    key. Calls where no human resolution exists are excluded rather than counted
    as failures: an unanswered case is not a wrong answer.
    """
    if k < 1:
        raise ValueError(f"k must be positive; got {k}")
    if len(ranked_causes) != len(accepted_causes):
        raise ValueError(
            f"{len(ranked_causes)} ranked lists against {len(accepted_causes)} resolutions"
        )

    scored = [
        (ranked, accepted)
        for ranked, accepted in zip(ranked_causes, accepted_causes, strict=True)
        if accepted is not None
    ]
    if not scored:
        return 0.0
    hits = sum(1 for ranked, accepted in scored if accepted in ranked[:k])
    return hits / len(scored)
