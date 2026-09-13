"""The append-only audit trail.

Four categories of event are required to be recorded: every model output, every
human decision, every guardrail action, and every threshold change. This module
is the only sanctioned writer.

Immutability is enforced by the database, not here -- ``audit_records`` has
UPDATE, DELETE, and TRUNCATE revoked and a trigger that raises regardless of
privilege. What this module adds is *tamper evidence*: each record's digest
covers its own payload and the previous record's digest, so the log forms a
hash chain. Deleting or altering a record is already impossible; altering one and
recomputing the chain requires rewriting every record after it, and the break
shows up in :func:`verify_chain`.

The digest covers a canonical JSON serialization with sorted keys, so two
semantically identical payloads hash identically regardless of dict ordering.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from yieldloop.db.enums import AuditEventType, GuardrailOutcome, GuardrailStage
from yieldloop.db.models import AuditRecord, GuardrailAction

#: Digest value used as the predecessor of the very first record.
GENESIS_DIGEST = "0" * 64


def canonical_payload(payload: dict[str, Any]) -> str:
    """Serialize a payload deterministically for hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def compute_digest(
    *,
    prev_digest: str | None,
    event_type: AuditEventType,
    actor: str,
    subject_type: str,
    subject_id: str,
    request_id: str,
    payload: dict[str, Any],
) -> str:
    """Digest over the predecessor and this record's content.

    Every field that carries meaning is covered. Omitting any of them would let
    that field be altered without breaking the chain.
    """
    parts = [
        prev_digest or GENESIS_DIGEST,
        event_type.value,
        actor,
        subject_type,
        subject_id,
        request_id,
        canonical_payload(payload),
    ]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ChainVerification:
    """The result of walking the chain."""

    checked: int
    #: Sequence numbers where the recomputed digest did not match the stored one.
    broken_at: tuple[int, ...]

    @property
    def intact(self) -> bool:
        return not self.broken_at


class AuditLog:
    """The sanctioned writer for ``audit_records``."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def _latest_digest(self) -> str | None:
        return self._session.execute(
            select(AuditRecord.digest).order_by(desc(AuditRecord.sequence)).limit(1)
        ).scalar_one_or_none()

    def append(
        self,
        *,
        event_type: AuditEventType,
        actor: str,
        subject_type: str,
        subject_id: str,
        request_id: str,
        payload: dict[str, Any],
    ) -> AuditRecord:
        """Append one record, linking it to the current head of the chain."""
        prev_digest = self._latest_digest()
        digest = compute_digest(
            prev_digest=prev_digest,
            event_type=event_type,
            actor=actor,
            subject_type=subject_type,
            subject_id=subject_id,
            request_id=request_id,
            payload=payload,
        )
        record = AuditRecord(
            event_type=event_type,
            actor=actor,
            subject_type=subject_type,
            subject_id=subject_id,
            request_id=request_id,
            payload=payload,
            prev_digest=prev_digest,
            digest=digest,
        )
        self._session.add(record)
        self._session.flush()
        return record

    def record_model_output(
        self, *, actor: str, request_id: str, subject_id: str, payload: dict[str, Any]
    ) -> AuditRecord:
        return self.append(
            event_type=AuditEventType.MODEL_OUTPUT,
            actor=actor,
            subject_type="hypothesis_request",
            subject_id=subject_id,
            request_id=request_id,
            payload=payload,
        )

    def record_human_decision(
        self, *, reviewer_id: str, request_id: str, decision_id: str, payload: dict[str, Any]
    ) -> AuditRecord:
        return self.append(
            event_type=AuditEventType.HUMAN_DECISION,
            actor=reviewer_id,
            subject_type="decision",
            subject_id=decision_id,
            request_id=request_id,
            payload=payload,
        )

    def record_guardrail_action(
        self,
        *,
        actor: str,
        request_id: str,
        stage: GuardrailStage,
        outcome: GuardrailOutcome,
        reason: str,
        detail: dict[str, Any] | None = None,
    ) -> tuple[GuardrailAction, AuditRecord]:
        """Record a guardrail intervention in both its own table and the chain.

        The dedicated table drives the guardrail dashboards and the adversarial
        suite; the audit record is what makes the action non-repudiable.
        """
        payload: dict[str, Any] = {
            "stage": stage.value,
            "outcome": outcome.value,
            "reason": reason,
            "detail": detail or {},
        }
        action = GuardrailAction(
            request_id=request_id,
            stage=stage,
            outcome=outcome,
            reason=reason,
            detail=detail or {},
        )
        self._session.add(action)
        self._session.flush()

        record = self.append(
            event_type=AuditEventType.GUARDRAIL_ACTION,
            actor=actor,
            subject_type="guardrail_action",
            subject_id=str(action.id),
            request_id=request_id,
            payload=payload,
        )
        return action, record

    def record_threshold_change(
        self,
        *,
        changed_by: str,
        request_id: str,
        name: str,
        old_value: float,
        new_value: float,
        rationale: str,
    ) -> AuditRecord:
        return self.append(
            event_type=AuditEventType.THRESHOLD_CHANGE,
            actor=changed_by,
            subject_type="threshold",
            subject_id=name,
            request_id=request_id,
            payload={
                "name": name,
                "old_value": old_value,
                "new_value": new_value,
                "rationale": rationale,
            },
        )

    def verify_chain(self) -> ChainVerification:
        """Recompute every digest and report where the chain breaks.

        Walks in sequence order. A mismatch means a record's content no longer
        matches what was hashed when it was written, or that a record's
        predecessor link does not match the record actually before it.
        """
        records = (
            self._session.execute(select(AuditRecord).order_by(AuditRecord.sequence))
            .scalars()
            .all()
        )
        broken: list[int] = []
        expected_prev: str | None = None

        for record in records:
            recomputed = compute_digest(
                prev_digest=record.prev_digest,
                event_type=record.event_type,
                actor=record.actor,
                subject_type=record.subject_type,
                subject_id=record.subject_id,
                request_id=record.request_id,
                payload=record.payload,
            )
            if recomputed != record.digest or record.prev_digest != expected_prev:
                broken.append(record.sequence)
            expected_prev = record.digest

        return ChainVerification(checked=len(records), broken_at=tuple(broken))
