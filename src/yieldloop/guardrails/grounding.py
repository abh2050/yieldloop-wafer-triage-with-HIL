"""The grounding gate.

This is the guardrail the project exists to demonstrate. A language model asked
for root cause hypotheses will produce fluent, plausible, well-structured
hypotheses whether or not it has any evidence, and a process engineer cannot tell
the two apart by reading them. The only defence that survives contact with a real
fab is refusing to render a claim that does not resolve to evidence the system
itself retrieved.

The rule is mechanical. Every hypothesis cites one or more ``evidence_id``
values. Every cited ID must be present in the context bundle assembled for *that*
request. An ID that does not resolve is not a formatting problem to be repaired;
it is a fabrication, and the hypothesis carrying it is dropped whole. Partial
repair -- keeping a hypothesis after discarding its bad citations -- is
explicitly not done, because the remaining citations were selected to support a
claim that was partly built on something invented.

If nothing survives, the result is an explicit abstention. The console renders
that as "insufficient evidence", which is a useful answer. An empty card is not.
"""

from __future__ import annotations

from dataclasses import dataclass

from yieldloop.agent.schemas import ContextBundle, Hypothesis, HypothesisResponse

#: Reason codes recorded on the audit trail for each drop.
REASON_UNRESOLVABLE_EVIDENCE = "unresolvable_evidence_id"
REASON_EMPTY_BUNDLE = "empty_context_bundle"
REASON_ALL_HYPOTHESES_DROPPED = "all_hypotheses_ungrounded"


@dataclass(frozen=True, slots=True)
class DroppedHypothesis:
    """A hypothesis the gate refused, kept for the audit log."""

    rank: int
    cause_category: str
    reason: str
    #: The cited IDs that were not in the bundle. This is the evidence of
    #: fabrication, and it is what the guardrail dashboard counts.
    unresolvable_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroundingResult:
    """The outcome of the gate, including what it removed and why."""

    response: HypothesisResponse
    dropped: tuple[DroppedHypothesis, ...]
    #: Hypotheses the model returned, before the gate.
    returned_count: int
    #: Hypotheses that survived it.
    grounded_count: int

    @property
    def rejection_rate(self) -> float:
        """Fraction of hypotheses dropped. Watched by the circuit breaker."""
        if self.returned_count == 0:
            return 0.0
        return len(self.dropped) / self.returned_count

    @property
    def abstained(self) -> bool:
        return self.response.abstained


def unresolvable_ids(hypothesis: Hypothesis, bundle: ContextBundle) -> tuple[str, ...]:
    """Cited IDs that are not present in the bundle."""
    available = bundle.evidence_ids
    return tuple(eid for eid in hypothesis.evidence_ids if eid not in available)


def enforce(response: HypothesisResponse, bundle: ContextBundle) -> GroundingResult:
    """Apply the grounding gate to a parsed response.

    An already-abstained response passes through untouched: there is nothing to
    ground, and the model declining to answer is the outcome the gate exists to
    produce anyway.
    """
    if response.abstained:
        return GroundingResult(response=response, dropped=(), returned_count=0, grounded_count=0)

    returned = len(response.hypotheses)

    if bundle.is_empty:
        # Every citation is unresolvable by construction. Reported as one clear
        # reason rather than as N identical per-hypothesis failures.
        return GroundingResult(
            response=HypothesisResponse.abstention(
                lot_id=response.lot_id,
                reason=(
                    "No evidence was retrieved for this lot, so no hypothesis can be "
                    "grounded. Additional retrieval or a wider lot window is required."
                ),
                injection_suspected=response.injection_suspected,
                context_gaps=response.context_gaps,
            ),
            dropped=tuple(
                DroppedHypothesis(
                    rank=h.rank,
                    cause_category=h.cause_category.value,
                    reason=REASON_EMPTY_BUNDLE,
                    unresolvable_ids=tuple(h.evidence_ids),
                )
                for h in response.hypotheses
            ),
            returned_count=returned,
            grounded_count=0,
        )

    survivors: list[Hypothesis] = []
    dropped: list[DroppedHypothesis] = []

    for hypothesis in response.hypotheses:
        missing = unresolvable_ids(hypothesis, bundle)
        if missing:
            dropped.append(
                DroppedHypothesis(
                    rank=hypothesis.rank,
                    cause_category=hypothesis.cause_category.value,
                    reason=REASON_UNRESOLVABLE_EVIDENCE,
                    unresolvable_ids=missing,
                )
            )
            continue
        survivors.append(hypothesis)

    if not survivors:
        return GroundingResult(
            response=HypothesisResponse.abstention(
                lot_id=response.lot_id,
                reason=(
                    f"All {returned} proposed hypotheses cited evidence that is not in the "
                    "retrieved context and were rejected. No grounded root cause can be "
                    "offered for this lot."
                ),
                injection_suspected=response.injection_suspected,
                context_gaps=response.context_gaps,
            ),
            dropped=tuple(dropped),
            returned_count=returned,
            grounded_count=0,
        )

    # Dropping a middle hypothesis leaves a gap in the rank sequence, which the
    # response schema forbids and which would corrupt precision-at-k. Re-rank
    # densely, preserving the model's relative ordering.
    regraded = tuple(
        hypothesis.model_copy(update={"rank": position})
        for position, hypothesis in enumerate(sorted(survivors, key=lambda h: h.rank), start=1)
    )

    return GroundingResult(
        response=response.model_copy(update={"hypotheses": regraded}),
        dropped=tuple(dropped),
        returned_count=returned,
        grounded_count=len(regraded),
    )
