"""Assembling a context bundle from retrieval results.

The bundle is the agent's entire universe. Anything not in it cannot be cited,
so this module is what decides what the agent is allowed to know -- which makes
it a guardrail surface in its own right, even though it lives under ``agent``.

Evidence IDs are constructed here by the same functions the ingest layer uses, so
the identifiers in a prompt and the identifiers in the database cannot drift.
"""

from __future__ import annotations

from collections.abc import Sequence

from yieldloop.agent.schemas import (
    ClassifierEvidence,
    ContextBundle,
    DieStatisticsEvidence,
    ProcessEventEvidence,
    SimilarLotEvidence,
)
from yieldloop.db.models import HistoricalExcursion, Lot, Prediction, ProcessEvent, Wafer
from yieldloop.ingest.normalize import (
    DieStatistics,
    evidence_id_for_die_statistics,
    evidence_id_for_prediction,
)


def classifier_evidence(wafer: Wafer, prediction: Prediction) -> ClassifierEvidence:
    """Render a stored prediction as citable evidence."""
    return ClassifierEvidence(
        evidence_id=evidence_id_for_prediction(wafer.wafer_id),
        wafer_id=wafer.wafer_id,
        predicted_pattern=prediction.predicted_label,
        confidence=prediction.confidence,
        probabilities={k: float(v) for k, v in prediction.probabilities.items()},
    )


def die_statistics_evidence(wafer: Wafer, statistics: DieStatistics) -> DieStatisticsEvidence:
    """Render measured die statistics as citable evidence."""
    return DieStatisticsEvidence(
        evidence_id=evidence_id_for_die_statistics(wafer.wafer_id),
        wafer_id=wafer.wafer_id,
        die_total=statistics.die_total,
        die_fail=statistics.die_fail,
        failure_rate=statistics.failure_rate,
        radial_fail_rates=statistics.radial_fail_rates,
        edge_concentration=statistics.edge_concentration,
    )


def similar_lot_evidence(
    excursion: HistoricalExcursion, lot_name: str, similarity: float
) -> SimilarLotEvidence:
    """Render a retrieved historical lot as citable evidence."""
    return SimilarLotEvidence(
        evidence_id=excursion.evidence_id,
        lot_name=lot_name,
        observed_pattern=excursion.observed_pattern,
        pattern_share=excursion.pattern_share,
        resolved_cause=excursion.resolved_cause,
        resolution_text=excursion.resolution_text,
        similarity=similarity,
    )


def process_event_evidence(event: ProcessEvent, lot_name: str) -> ProcessEventEvidence:
    """Render a derived process event as citable evidence."""
    return ProcessEventEvidence(
        evidence_id=event.evidence_id,
        lot_name=lot_name,
        event_date=event.event_date,
        category=event.category,
        summary=event.summary,
        attributes={k: v for k, v in event.attributes.items() if isinstance(v, int | float | str)},
        derivation_rule=event.derivation_rule,
    )


def build_bundle(
    *,
    lot: Lot,
    classifier: Sequence[ClassifierEvidence] = (),
    die_statistics: Sequence[DieStatisticsEvidence] = (),
    similar_lots: Sequence[SimilarLotEvidence] = (),
    process_events: Sequence[ProcessEventEvidence] = (),
) -> ContextBundle:
    """Assemble the bundle for one lot.

    An empty bundle is a legitimate outcome, not an error: it means retrieval
    found nothing, and the correct response to that is an abstention, which the
    guarded entry point produces without paying for a model call.
    """
    return ContextBundle(
        lot_id=str(lot.id),
        lot_name=lot.lot_name,
        classifier=tuple(classifier),
        die_statistics=tuple(die_statistics),
        similar_lots=tuple(similar_lots),
        process_events=tuple(process_events),
    )
