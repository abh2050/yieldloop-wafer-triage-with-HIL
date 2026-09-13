"""The grounding gate must fail closed on every evidence-starved input.

The threat model is not a model that returns garbage -- garbage is caught by the
schema. It is a model that returns a fluent, well-formed, entirely plausible
hypothesis citing an evidence ID that does not exist. A process engineer reading
the console cannot distinguish that from a real one, so the gate must.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from yieldloop.agent.schemas import (
    ClassifierEvidence,
    ContextBundle,
    DieStatisticsEvidence,
    Hypothesis,
    HypothesisResponse,
    ProcessEventEvidence,
    SimilarLotEvidence,
)
from yieldloop.db.enums import CauseCategory, DefectPattern
from yieldloop.guardrails.grounding import (
    REASON_EMPTY_BUNDLE,
    REASON_UNRESOLVABLE_EVIDENCE,
    enforce,
    unresolvable_ids,
)

LOT_ID = "lot00891"


def _uniform_probabilities(top: DefectPattern, confidence: float) -> dict[str, float]:
    others = [p for p in DefectPattern if p is not top]
    share = (1.0 - confidence) / len(others)
    return {top.value: confidence, **{p.value: share for p in others}}


def _bundle() -> ContextBundle:
    """A bundle with one of each evidence kind, so citations can be checked
    against a realistic set rather than a single ID."""
    return ContextBundle(
        lot_id=LOT_ID,
        lot_name="lot00891",
        classifier=(
            ClassifierEvidence(
                evidence_id="cp:lot00891-3",
                wafer_id="lot00891-3",
                predicted_pattern=DefectPattern.EDGE_RING,
                confidence=0.72,
                probabilities=_uniform_probabilities(DefectPattern.EDGE_RING, 0.72),
            ),
        ),
        die_statistics=(
            DieStatisticsEvidence(
                evidence_id="ds:lot00891-3",
                wafer_id="lot00891-3",
                die_total=1124,
                die_fail=408,
                failure_rate=0.363,
                radial_fail_rates=(0.0, 0.0, 0.02, 0.31, 0.97),
                edge_concentration=2.67,
            ),
        ),
        similar_lots=(
            SimilarLotEvidence(
                evidence_id="hx:lot00412",
                lot_name="lot00412",
                observed_pattern=DefectPattern.EDGE_RING,
                pattern_share=0.81,
                resolved_cause=CauseCategory.UPSTREAM_PROCESS,
                resolution_text="Lot lot00412: edge ring in 81% of labeled wafers.",
                similarity=0.91,
            ),
        ),
        process_events=(
            ProcessEventEvidence(
                evidence_id="pe:lot00891:0",
                lot_name="lot00891",
                event_date="2021-04-02",
                category=CauseCategory.HANDLING_MECHANICAL,
                summary="waferIndex values are non-contiguous; 3 indices missing.",
                attributes={"missing_count": 3, "wafer_count": 22},
                derivation_rule="PROCESS_EVENT_DERIVE",
            ),
        ),
    )


def _empty_bundle() -> ContextBundle:
    return ContextBundle(
        lot_id=LOT_ID,
        lot_name="lot00891",
        classifier=(),
        die_statistics=(),
        similar_lots=(),
        process_events=(),
    )


def _hypothesis(rank: int, evidence_ids: tuple[str, ...]) -> Hypothesis:
    return Hypothesis(
        rank=rank,
        cause_category=CauseCategory.UPSTREAM_PROCESS,
        statement=f"Edge-concentrated failure consistent with upstream handling ({rank}).",
        evidence_ids=evidence_ids,
        supporting_signal="Outer-ring fail rate is 0.97 against 0.363 whole-wafer.",
        confidence=0.6,
        confirming_query="Compare edge-exclusion settings across the lot window.",
        eliminating_query="Check whether the inner rings show any elevated failure.",
    )


def _response(*hypotheses: Hypothesis) -> HypothesisResponse:
    return HypothesisResponse(
        lot_id=LOT_ID,
        abstained=False,
        hypotheses=hypotheses,
        injection_suspected=False,
    )


# --- fabricated evidence ---------------------------------------------------


@pytest.mark.parametrize(
    "fabricated",
    [
        "cp:lot99999-1",
        "ds:lot00891-9",
        "hx:lot00000",
        "pe:lot00891:7",
        "pe:lot00412:0",
    ],
)
def test_a_single_fabricated_citation_drops_the_hypothesis(fabricated: str) -> None:
    result = enforce(_response(_hypothesis(1, (fabricated,))), _bundle())
    assert result.abstained
    assert result.grounded_count == 0
    assert result.dropped[0].reason == REASON_UNRESOLVABLE_EVIDENCE
    assert result.dropped[0].unresolvable_ids == (fabricated,)


@pytest.mark.parametrize(
    "malformed",
    [
        "pe:lot00891:0x",
        "xx:lot00891",
        "see the edge ring on wafer 3",
        "<a href='#'>cp:lot00891-3</a>",
        "cp:lot00891-3 and also ds:lot00891-3",
        "",
    ],
)
def test_malformed_citations_are_rejected_before_the_grounding_gate(
    malformed: str,
) -> None:
    """Defence in depth: an ID that is not even shaped like one never reaches
    grounding, because the response schema refuses to parse it."""
    with pytest.raises(ValidationError):
        _hypothesis(1, (malformed,))


def test_a_partly_fabricated_hypothesis_is_dropped_whole() -> None:
    """No partial repair.

    Keeping the claim after discarding its bad citation would leave a statement
    that was partly built on something invented, with the surviving citations
    lending it unearned credibility.
    """
    result = enforce(
        _response(_hypothesis(1, ("cp:lot00891-3", "pe:ghost:1", "ds:lot00891-3"))),
        _bundle(),
    )
    assert result.abstained
    assert result.grounded_count == 0
    assert result.dropped[0].unresolvable_ids == ("pe:ghost:1",)


def test_grounded_hypotheses_survive_untouched() -> None:
    response = _response(
        _hypothesis(1, ("cp:lot00891-3", "ds:lot00891-3")),
        _hypothesis(2, ("hx:lot00412",)),
    )
    result = enforce(response, _bundle())
    assert not result.abstained
    assert result.grounded_count == 2
    assert result.dropped == ()
    assert result.rejection_rate == 0.0


def test_surviving_ranks_are_re_densified() -> None:
    """Dropping a middle hypothesis must not leave a gap.

    A gap would violate the response schema and would corrupt precision at k.
    """
    response = _response(
        _hypothesis(1, ("cp:lot00891-3",)),
        _hypothesis(2, ("pe:ghost:2",)),
        _hypothesis(3, ("hx:lot00412",)),
    )
    result = enforce(response, _bundle())
    assert result.grounded_count == 2
    assert [h.rank for h in result.response.hypotheses] == [1, 2]
    assert result.rejection_rate == pytest.approx(1 / 3)


def test_relative_order_is_preserved_after_re_ranking() -> None:
    response = _response(
        _hypothesis(1, ("pe:ghost:1",)),
        _hypothesis(2, ("cp:lot00891-3",)),
        _hypothesis(3, ("hx:lot00412",)),
    )
    result = enforce(response, _bundle())
    statements = [h.statement for h in result.response.hypotheses]
    assert statements[0].endswith("(2).")
    assert statements[1].endswith("(3).")


# --- abstention ------------------------------------------------------------


def test_all_fabricated_yields_an_explicit_abstention_not_an_empty_card() -> None:
    result = enforce(
        _response(_hypothesis(1, ("pe:ghost:1",)), _hypothesis(2, ("hx:ghost",))),
        _bundle(),
    )
    assert result.abstained
    assert result.response.hypotheses == ()
    assert result.response.abstention_reason
    assert "not in the retrieved context" in result.response.abstention_reason


def test_empty_bundle_abstains_with_its_own_reason_code() -> None:
    """An evidence-starved context is reported as one clear cause, not as N
    identical per-hypothesis failures."""
    result = enforce(_response(_hypothesis(1, ("cp:lot00891-3",))), _empty_bundle())
    assert result.abstained
    assert all(d.reason == REASON_EMPTY_BUNDLE for d in result.dropped)
    assert result.response.abstention_reason
    assert "No evidence was retrieved" in result.response.abstention_reason


def test_an_already_abstained_response_passes_through_unchanged() -> None:
    response = HypothesisResponse.abstention(LOT_ID, "model declined")
    result = enforce(response, _bundle())
    assert result.abstained
    assert result.dropped == ()
    assert result.response.abstention_reason == "model declined"


def test_injection_flag_survives_abstention() -> None:
    """Losing the flag would hide an attack behind a routine abstention."""
    response = HypothesisResponse(
        lot_id=LOT_ID,
        abstained=False,
        hypotheses=(_hypothesis(1, ("pe:ghost:1",)),),
        injection_suspected=True,
        context_gaps=("no similar lots retrieved",),
    )
    result = enforce(response, _bundle())
    assert result.abstained
    assert result.response.injection_suspected
    assert result.response.context_gaps == ("no similar lots retrieved",)


# --- helpers ---------------------------------------------------------------


def test_unresolvable_ids_reports_only_the_missing_ones() -> None:
    hypothesis = _hypothesis(1, ("cp:lot00891-3", "pe:ghost:1", "hx:ghost"))
    assert unresolvable_ids(hypothesis, _bundle()) == ("pe:ghost:1", "hx:ghost")


def test_every_bundle_id_is_citable() -> None:
    bundle = _bundle()
    assert bundle.evidence_ids == {
        "cp:lot00891-3",
        "ds:lot00891-3",
        "hx:lot00412",
        "pe:lot00891:0",
    }
    for evidence_id in bundle.evidence_ids:
        result = enforce(_response(_hypothesis(1, (evidence_id,))), bundle)
        assert not result.abstained, f"{evidence_id} should be citable"


def test_rejection_rate_is_reported_for_the_circuit_breaker() -> None:
    response = _response(
        _hypothesis(1, ("cp:lot00891-3",)),
        _hypothesis(2, ("pe:ghost:1",)),
        _hypothesis(3, ("pe:ghost:2",)),
        _hypothesis(4, ("hx:lot00412",)),
    )
    result = enforce(response, _bundle())
    assert result.returned_count == 4
    assert result.grounded_count == 2
    assert result.rejection_rate == pytest.approx(0.5)
