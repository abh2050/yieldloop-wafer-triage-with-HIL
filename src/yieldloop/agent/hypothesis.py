"""The guarded entry point for hypothesis generation.

This is the only way into the model. The guardrails are not parameters of this
function and there is no flag that disables one: the sequence below is the
implementation, so a caller cannot skip a stage without editing this file, and
editing it shows up in review.

The ordering is deliberate. Input validation and the budget pre-check happen
before any tokens are spent. Isolation happens while the prompt is built, so no
untrusted text ever reaches the instruction layer. Parsing and grounding happen
before anything is persisted or returned, so an ungrounded claim never becomes a
database row. Everything, including every refusal, is audited.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime

from sqlalchemy.orm import Session

from yieldloop.agent.client import AgentClient, AgentTransportError, strict_schema
from yieldloop.agent.prompts import SYSTEM_PROMPT, estimate_prompt_tokens, render_context
from yieldloop.agent.schemas import ContextBundle, HypothesisResponse
from yieldloop.config import Settings
from yieldloop.db.enums import (
    AuditEventType,
    EvidenceKind,
    GuardrailOutcome,
    GuardrailStage,
)
from yieldloop.db.models import BreakerEvent, HypothesisCitation, HypothesisRequest
from yieldloop.db.models import Hypothesis as HypothesisRow
from yieldloop.guardrails import BudgetExceededError, GuardrailError, SchemaViolationError
from yieldloop.guardrails.audit import AuditLog
from yieldloop.guardrails.budget import BudgetGuard, BudgetLimits, Pricing
from yieldloop.guardrails.circuit_breaker import (
    BreakerThresholds,
    CircuitBreaker,
    Outcome,
    Transition,
)
from yieldloop.guardrails.grounding import enforce
from yieldloop.guardrails.input_filter import InputFilter, validate_lot_name
from yieldloop.guardrails.schema_validator import parse_response, response_json_schema
from yieldloop.logging import get_logger

logger = get_logger(__name__)

#: Maps an evidence id prefix to its kind, for persisting citations.
_PREFIX_TO_KIND: dict[str, EvidenceKind] = {
    "cp": EvidenceKind.CLASSIFIER_PREDICTION,
    "ds": EvidenceKind.DIE_STATISTICS,
    "hx": EvidenceKind.SIMILAR_LOT,
    "pe": EvidenceKind.PROCESS_EVENT,
}


@dataclass(frozen=True, slots=True)
class GuardedResult:
    """What the API returns, after every gate has run."""

    response: HypothesisResponse
    request_id: str
    #: True when the breaker was open and no model call was made.
    degraded: bool
    #: Hypotheses the model returned, before grounding.
    returned_count: int
    #: Hypotheses that survived it.
    grounded_count: int
    injection_suspected: bool
    cost_usd: float
    latency_ms: float

    @property
    def abstained(self) -> bool:
        return self.response.abstained


class HypothesisService:
    """Composes the guardrails around one agent call.

    The circuit breaker is held on the instance rather than per call, because its
    window has to span requests to mean anything.
    """

    def __init__(
        self,
        session: Session,
        settings: Settings,
        *,
        client: AgentClient | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._session = session
        self._settings = settings
        self._client = client
        self._audit = AuditLog(session)
        self._input_filter = InputFilter.from_settings(settings)
        self._budget = BudgetGuard(
            session, BudgetLimits.from_settings(settings), Pricing.from_settings(settings)
        )
        self._breaker = breaker or CircuitBreaker(
            thresholds=BreakerThresholds.from_settings(settings)
        )

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    def generate(
        self,
        *,
        bundle: ContextBundle,
        requested_by: str,
        session_id: str,
        reviewer_note: str | None = None,
        now: datetime | None = None,
    ) -> GuardedResult:
        """Produce grounded hypotheses for one lot, or abstain.

        Never raises for an expected guardrail outcome. A refusal is returned as
        an abstention with the reason stated, because the console has to render
        something useful either way and an exception would become a 500 that
        tells the reviewer nothing.
        """
        request_id = uuid.uuid4().hex
        moment = now or datetime.now(UTC)
        started = time.monotonic()

        # 1. Validate before spending anything.
        try:
            validate_lot_name(bundle.lot_name)
            note = self._input_filter.check_note(reviewer_note, field="reviewer_note")
        except GuardrailError as exc:
            return self._refuse(
                bundle=bundle,
                request_id=request_id,
                requested_by=requested_by,
                stage=GuardrailStage.INPUT_FILTER,
                reason=str(exc.detail.get("code", exc.reason)),
                message=exc.message,
                started=started,
            )

        # 2. An open breaker means classifier-only mode, not a failed page.
        self._poll_breaker(moment=moment, requested_by=requested_by, request_id=request_id)
        if self._breaker.is_open:
            return self._refuse(
                bundle=bundle,
                request_id=request_id,
                requested_by=requested_by,
                stage=GuardrailStage.CIRCUIT_BREAKER,
                reason="circuit_open",
                message=(
                    "Hypothesis generation is temporarily unavailable and the console is "
                    "running in classifier-only mode. The wafer map, prediction, and review "
                    "queue are unaffected."
                ),
                started=started,
                degraded=True,
            )

        # 3. An empty bundle cannot support any claim; do not pay to find out.
        if bundle.is_empty:
            return self._refuse(
                bundle=bundle,
                request_id=request_id,
                requested_by=requested_by,
                stage=GuardrailStage.GROUNDING,
                reason="empty_context_bundle",
                message=(
                    "No evidence was retrieved for this lot, so no hypothesis can be "
                    "grounded. Additional retrieval or a wider lot window is required."
                ),
                started=started,
            )

        # 4. Build the prompt. Every untrusted field is isolated in here.
        rendered, injection_signals = render_context(bundle, reviewer_note=note)
        injection_suspected = bool(injection_signals)
        if injection_suspected:
            self._audit.record_guardrail_action(
                actor=requested_by,
                request_id=request_id,
                stage=GuardrailStage.INJECTION,
                outcome=GuardrailOutcome.MODIFIED,
                reason="instruction_shaped_content",
                detail={"signals": list(injection_signals)},
            )

        # 5. Budget, checked before the call.
        estimated_prompt = estimate_prompt_tokens(SYSTEM_PROMPT, rendered)
        try:
            self._budget.check_tokens(
                prompt_tokens=estimated_prompt,
                completion_tokens=self._settings.max_completion_tokens,
            )
            self._budget.check_spend(
                session_id=session_id,
                spend_date=moment.date(),
                estimated_prompt_tokens=estimated_prompt,
                estimated_completion_tokens=self._settings.max_completion_tokens,
            )
        except BudgetExceededError as exc:
            return self._refuse(
                bundle=bundle,
                request_id=request_id,
                requested_by=requested_by,
                stage=GuardrailStage.BUDGET,
                reason=str(exc.detail.get("code", exc.reason)),
                message=exc.message,
                started=started,
                injection_suspected=injection_suspected,
            )

        # 6. The call.
        try:
            client = self._client or AgentClient(self._settings)
            completion = client.complete(
                system_prompt=SYSTEM_PROMPT,
                user_message=rendered,
                json_schema=strict_schema(response_json_schema()),
                max_completion_tokens=self._settings.max_completion_tokens,
            )
        except AgentTransportError as exc:
            self._record_outcome(schema_ok=False, latency=0.0, rejection_rate=0.0, moment=moment,
                                 requested_by=requested_by, request_id=request_id)
            return self._refuse(
                bundle=bundle,
                request_id=request_id,
                requested_by=requested_by,
                stage=GuardrailStage.CIRCUIT_BREAKER,
                reason="transport_error",
                message=f"The hypothesis agent could not be reached: {exc}",
                started=started,
                degraded=True,
                injection_suspected=injection_suspected,
            )

        # 7. Record what was actually spent, whatever happens next.
        self._budget.record(
            session_id=session_id,
            request_id=request_id,
            spend_date=moment.date(),
            model=completion.model,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
        )
        cost = self._budget.pricing.cost(
            completion.prompt_tokens, completion.completion_tokens
        )

        # 8. Strict parse. Fails closed.
        try:
            parsed = parse_response(completion.content, expected_lot_id=bundle.lot_id)
        except SchemaViolationError as exc:
            self._audit.record_guardrail_action(
                actor=requested_by,
                request_id=request_id,
                stage=GuardrailStage.SCHEMA_VALIDATOR,
                outcome=GuardrailOutcome.BLOCKED,
                reason=str(exc.detail.get("stage", "schema")),
                detail={"truncated": completion.truncated},
            )
            self._record_outcome(
                schema_ok=False,
                latency=completion.latency_seconds,
                rejection_rate=0.0,
                moment=moment,
                requested_by=requested_by,
                request_id=request_id,
            )
            return self._refuse(
                bundle=bundle,
                request_id=request_id,
                requested_by=requested_by,
                stage=GuardrailStage.SCHEMA_VALIDATOR,
                reason="schema_violation",
                message=(
                    "The hypothesis agent returned output that did not match the required "
                    "schema, so nothing is shown. This is recorded and counted toward the "
                    "circuit breaker."
                ),
                started=started,
                cost=cost,
                injection_suspected=injection_suspected,
            )

        # 9. The grounding gate.
        grounded = enforce(parsed, bundle)
        for dropped in grounded.dropped:
            self._audit.record_guardrail_action(
                actor=requested_by,
                request_id=request_id,
                stage=GuardrailStage.GROUNDING,
                outcome=GuardrailOutcome.BLOCKED,
                reason=dropped.reason,
                detail={
                    "rank": dropped.rank,
                    "cause_category": dropped.cause_category,
                    "unresolvable_ids": list(dropped.unresolvable_ids),
                },
            )

        self._record_outcome(
            schema_ok=True,
            latency=completion.latency_seconds,
            rejection_rate=grounded.rejection_rate,
            moment=moment,
            requested_by=requested_by,
            request_id=request_id,
        )

        # The model's own injection flag is combined with what the scanner saw.
        # Either alone is a reason to raise it; neither is authoritative.
        suspected = injection_suspected or grounded.response.injection_suspected
        response = grounded.response.model_copy(update={"injection_suspected": suspected})

        latency_ms = (time.monotonic() - started) * 1000.0
        self._persist(
            bundle=bundle,
            response=response,
            request_id=request_id,
            requested_by=requested_by,
            model=completion.model,
            returned=grounded.returned_count,
            grounded_count=grounded.grounded_count,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            cost=cost,
            latency_ms=latency_ms,
            degraded=False,
        )

        return GuardedResult(
            response=response,
            request_id=request_id,
            degraded=False,
            returned_count=grounded.returned_count,
            grounded_count=grounded.grounded_count,
            injection_suspected=suspected,
            cost_usd=cost,
            latency_ms=latency_ms,
        )

    # -- internals ----------------------------------------------------------

    def _poll_breaker(self, *, moment: datetime, requested_by: str, request_id: str) -> None:
        transition = self._breaker.poll(now=moment.timestamp())
        if transition is not None:
            self._persist_transition(transition, requested_by=requested_by, request_id=request_id)

    def _record_outcome(
        self,
        *,
        schema_ok: bool,
        latency: float,
        rejection_rate: float,
        moment: datetime,
        requested_by: str,
        request_id: str,
    ) -> None:
        transition = self._breaker.record(
            Outcome(
                schema_ok=schema_ok,
                latency_seconds=latency,
                grounding_rejection_rate=rejection_rate,
            ),
            now=moment.timestamp(),
        )
        if transition is not None:
            self._persist_transition(transition, requested_by=requested_by, request_id=request_id)

    def _persist_transition(
        self, transition: Transition, *, requested_by: str, request_id: str
    ) -> None:
        self._session.add(
            BreakerEvent(
                from_state=transition.from_state,
                to_state=transition.to_state,
                trigger=transition.trigger,
                observed=dict(transition.observed),
            )
        )
        self._audit.record_guardrail_action(
            actor=requested_by,
            request_id=request_id,
            stage=GuardrailStage.CIRCUIT_BREAKER,
            outcome=GuardrailOutcome.BLOCKED
            if transition.to_state.value == "open"
            else GuardrailOutcome.PASSED,
            reason=transition.trigger,
            detail={
                "from": transition.from_state.value,
                "to": transition.to_state.value,
                **transition.observed,
            },
        )

    def _refuse(
        self,
        *,
        bundle: ContextBundle,
        request_id: str,
        requested_by: str,
        stage: GuardrailStage,
        reason: str,
        message: str,
        started: float,
        degraded: bool = False,
        cost: float = 0.0,
        injection_suspected: bool = False,
    ) -> GuardedResult:
        """Turn a guardrail refusal into an audited abstention."""
        self._audit.record_guardrail_action(
            actor=requested_by,
            request_id=request_id,
            stage=stage,
            outcome=GuardrailOutcome.BLOCKED,
            reason=reason,
            detail={"lot_id": bundle.lot_id},
        )
        response = HypothesisResponse.abstention(
            lot_id=bundle.lot_id, reason=message, injection_suspected=injection_suspected
        )
        latency_ms = (time.monotonic() - started) * 1000.0
        self._persist(
            bundle=bundle,
            response=response,
            request_id=request_id,
            requested_by=requested_by,
            model=self._settings.openai_model,
            returned=0,
            grounded_count=0,
            prompt_tokens=0,
            completion_tokens=0,
            cost=cost,
            latency_ms=latency_ms,
            degraded=degraded,
        )
        logger.info(
            "hypothesis_refused",
            request_id=request_id,
            stage=stage.value,
            reason=reason,
            degraded=degraded,
        )
        return GuardedResult(
            response=response,
            request_id=request_id,
            degraded=degraded,
            returned_count=0,
            grounded_count=0,
            injection_suspected=injection_suspected,
            cost_usd=cost,
            latency_ms=latency_ms,
        )

    def _persist(
        self,
        *,
        bundle: ContextBundle,
        response: HypothesisResponse,
        request_id: str,
        requested_by: str,
        model: str,
        returned: int,
        grounded_count: int,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        latency_ms: float,
        degraded: bool,
    ) -> HypothesisRequest:
        """Write the request, its surviving hypotheses, and their citations.

        Only grounded hypotheses reach this point, so every citation row here is
        resolvable by construction.
        """
        row = HypothesisRequest(
            lot_id=bundle.lot_id,
            requested_by=requested_by,
            model=model,
            context_hash=bundle.content_hash(),
            evidence_ids=sorted(bundle.evidence_ids),
            abstained=response.abstained,
            abstention_reason=response.abstention_reason,
            injection_suspected=response.injection_suspected,
            context_gaps=list(response.context_gaps),
            hypotheses_returned=returned,
            hypotheses_grounded=grounded_count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            degraded=degraded,
        )
        self._session.add(row)
        self._session.flush()

        for hypothesis in response.hypotheses:
            hypothesis_row = HypothesisRow(
                request_id=row.id,
                rank=hypothesis.rank,
                cause_category=hypothesis.cause_category,
                statement=hypothesis.statement,
                supporting_signal=hypothesis.supporting_signal,
                contradicting_signal=hypothesis.contradicting_signal,
                confidence=hypothesis.confidence,
                confirming_query=hypothesis.confirming_query,
                eliminating_query=hypothesis.eliminating_query,
            )
            self._session.add(hypothesis_row)
            self._session.flush()
            for evidence_id in hypothesis.evidence_ids:
                self._session.add(
                    HypothesisCitation(
                        hypothesis_id=hypothesis_row.id,
                        evidence_id=evidence_id,
                        evidence_kind=_PREFIX_TO_KIND[evidence_id.split(":", 1)[0]],
                    )
                )

        self._audit.append(
            event_type=AuditEventType.MODEL_OUTPUT,
            actor=requested_by,
            subject_type="hypothesis_request",
            subject_id=str(row.id),
            request_id=request_id,
            payload={
                "lot_id": bundle.lot_id,
                "context_hash": row.context_hash,
                "abstained": response.abstained,
                "hypotheses_returned": returned,
                "hypotheses_grounded": grounded_count,
                "injection_suspected": response.injection_suspected,
                "degraded": degraded,
                "cost_usd": cost,
            },
        )
        self._session.flush()
        return row


def spend_date_for(moment: datetime) -> date:
    """The ledger day a call belongs to, in UTC."""
    return moment.astimezone(UTC).date()
