"""Boundary schemas for the hypothesis agent.

Two families live here and the split matters.

The *context bundle* is everything the agent is permitted to know. It is
assembled from retrieval results only, and every element in it carries an
``evidence_id``. The set of IDs in the bundle is exactly the set the agent may
cite; the grounding gate resolves citations against that set and nothing else.

The *response* is what the agent returns. It is parsed strictly. A response that
does not validate never reaches the frontend, and there is no lenient fallback
path, because a partially-parsed root cause claim is worse than none.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yieldloop.db.enums import CauseCategory, DefectPattern, EvidenceKind

#: Evidence identifiers are structured (``kind:key[:ordinal]``) and bounded.
#: The pattern is enforced at parse time so a citation cannot smuggle prose,
#: whitespace, or markup into a field that is later rendered as a link.
EVIDENCE_ID_PATTERN = r"^(cp|ds|hx|pe):[A-Za-z0-9_.\-]{1,64}(:[0-9]{1,4})?$"

EvidenceId = Annotated[str, Field(pattern=EVIDENCE_ID_PATTERN, max_length=96)]


class StrictModel(BaseModel):
    """Base for every boundary payload. Unknown fields are an error.

    ``extra="forbid"`` is deliberate: a model that returns an unexpected key is
    not following the contract, and silently dropping the key would hide that.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Context bundle
# ---------------------------------------------------------------------------


class ClassifierEvidence(StrictModel):
    """The calibrated classifier prediction for the wafer under review."""

    evidence_id: EvidenceId
    kind: Literal[EvidenceKind.CLASSIFIER_PREDICTION] = EvidenceKind.CLASSIFIER_PREDICTION
    wafer_id: str = Field(max_length=80)
    predicted_pattern: DefectPattern
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]

    @model_validator(mode="after")
    def _check_distribution(self) -> Self:
        unknown = set(self.probabilities) - {p.value for p in DefectPattern}
        if unknown:
            raise ValueError(f"probabilities contain unknown classes: {sorted(unknown)}")
        total = sum(self.probabilities.values())
        if not 0.999 <= total <= 1.001:
            raise ValueError(f"probabilities must sum to 1.0; got {total}")
        return self


class DieStatisticsEvidence(StrictModel):
    """Measured die-level statistics. Every value is counted, not modelled."""

    evidence_id: EvidenceId
    kind: Literal[EvidenceKind.DIE_STATISTICS] = EvidenceKind.DIE_STATISTICS
    wafer_id: str = Field(max_length=80)
    die_total: int = Field(gt=0)
    die_fail: int = Field(ge=0)
    failure_rate: float = Field(ge=0.0, le=1.0)
    #: Fail rate per concentric ring, innermost first.
    radial_fail_rates: tuple[float, ...]
    edge_concentration: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        if self.die_fail > self.die_total:
            raise ValueError(f"die_fail {self.die_fail} exceeds die_total {self.die_total}")
        return self


class SimilarLotEvidence(StrictModel):
    """A retrieved historical lot with a recorded cause.

    ``resolution_text`` is untrusted: it is rendered from real measurements, but
    it is free text and is structurally isolated before it enters a prompt.
    """

    evidence_id: EvidenceId
    kind: Literal[EvidenceKind.SIMILAR_LOT] = EvidenceKind.SIMILAR_LOT
    lot_name: str = Field(max_length=64)
    observed_pattern: DefectPattern
    pattern_share: float = Field(gt=0.0, le=1.0)
    resolved_cause: CauseCategory
    resolution_text: str = Field(max_length=2000)
    similarity: float = Field(ge=0.0, le=1.0)
    #: Always true. These rows are derived by a documented rule, never observed.
    is_derived: Literal[True] = True


class ProcessEventEvidence(StrictModel):
    """A derived process event in the window around the lot.

    Carries no tool, chamber, recipe, operator, or production timestamp, because
    WM811K contains none. The agent is therefore structurally unable to cite one.
    """

    evidence_id: EvidenceId
    kind: Literal[EvidenceKind.PROCESS_EVENT] = EvidenceKind.PROCESS_EVENT
    lot_name: str = Field(max_length=64)
    event_date: date
    category: CauseCategory
    summary: str = Field(max_length=1000)
    attributes: dict[str, float | int | str]
    is_derived: Literal[True] = True
    derivation_rule: str = Field(max_length=64)


Evidence = ClassifierEvidence | DieStatisticsEvidence | SimilarLotEvidence | ProcessEventEvidence


class ContextBundle(StrictModel):
    """Everything the agent is allowed to know for one request.

    The bundle defines the universe of citable evidence. :attr:`evidence_ids` is
    what the grounding gate checks against, and :meth:`content_hash` is recorded
    on the request so an audit can later prove exactly what was available.
    """

    lot_id: str = Field(max_length=64)
    lot_name: str = Field(max_length=64)
    classifier: tuple[ClassifierEvidence, ...]
    die_statistics: tuple[DieStatisticsEvidence, ...]
    similar_lots: tuple[SimilarLotEvidence, ...]
    process_events: tuple[ProcessEventEvidence, ...]

    @property
    def items(self) -> tuple[Evidence, ...]:
        return (
            *self.classifier,
            *self.die_statistics,
            *self.similar_lots,
            *self.process_events,
        )

    @property
    def evidence_ids(self) -> frozenset[str]:
        """Exactly the set of IDs a hypothesis may cite."""
        return frozenset(item.evidence_id for item in self.items)

    @property
    def is_empty(self) -> bool:
        """True when there is nothing to reason from, so abstention is the only
        honest outcome."""
        return not self.items

    @model_validator(mode="after")
    def _check_ids_are_unique(self) -> Self:
        seen: set[str] = set()
        for item in self.items:
            if item.evidence_id in seen:
                raise ValueError(f"duplicate evidence_id in bundle: {item.evidence_id}")
            seen.add(item.evidence_id)
        return self

    def content_hash(self) -> str:
        """Stable sha256 over the bundle, independent of key ordering."""
        payload = self.model_dump(mode="json")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class Hypothesis(StrictModel):
    """One ranked root cause hypothesis.

    ``evidence_ids`` must be non-empty. An uncited claim is invalid output by
    definition, so it is rejected at parse time rather than surviving to the
    grounding gate as a claim with nothing to resolve.
    """

    rank: int = Field(ge=1, le=10)
    cause_category: CauseCategory
    statement: str = Field(min_length=1, max_length=1000)
    evidence_ids: tuple[EvidenceId, ...] = Field(min_length=1, max_length=16)
    supporting_signal: str = Field(min_length=1, max_length=1000)
    contradicting_signal: str | None = Field(default=None, max_length=1000)
    confidence: float = Field(ge=0.0, le=1.0)
    confirming_query: str = Field(min_length=1, max_length=500)
    eliminating_query: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def _check_citations_are_unique(self) -> Self:
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError(f"duplicate evidence_ids on hypothesis {self.rank}")
        return self


class HypothesisResponse(StrictModel):
    """The agent's complete output for one request."""

    lot_id: str = Field(max_length=64)
    abstained: bool
    abstention_reason: str | None = Field(default=None, max_length=1000)
    hypotheses: tuple[Hypothesis, ...] = Field(max_length=10)
    injection_suspected: bool
    context_gaps: tuple[str, ...] = Field(default=(), max_length=20)

    @model_validator(mode="after")
    def _check_abstention_is_coherent(self) -> Self:
        """Abstention and hypotheses are mutually exclusive.

        A response claiming both would let a caller render a hypothesis from a
        request the model declined to answer.
        """
        if self.abstained:
            if self.hypotheses:
                raise ValueError("an abstained response must not carry hypotheses")
            if not self.abstention_reason:
                raise ValueError("an abstained response must state a reason")
        elif not self.hypotheses:
            raise ValueError(
                "a non-abstained response must carry at least one hypothesis; "
                "use abstained=true with a reason instead of returning nothing"
            )
        return self

    @model_validator(mode="after")
    def _check_ranks_are_dense_and_ordered(self) -> Self:
        """Ranks must be 1..n with no gaps and no ties.

        The console renders these in order and the eval harness measures
        precision at 1 and at 3, both of which are meaningless if rank is not a
        total order.
        """
        ranks = [h.rank for h in self.hypotheses]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"ranks must be 1..{len(ranks)} in order; got {ranks}")
        return self

    @classmethod
    def abstention(
        cls,
        lot_id: str,
        reason: str,
        *,
        injection_suspected: bool = False,
        context_gaps: tuple[str, ...] = (),
    ) -> HypothesisResponse:
        """Build the explicit abstention object.

        Abstaining is a correct outcome. The console renders this as
        "insufficient evidence" rather than as an empty card.
        """
        return cls(
            lot_id=lot_id,
            abstained=True,
            abstention_reason=reason,
            hypotheses=(),
            injection_suspected=injection_suspected,
            context_gaps=context_gaps,
        )
