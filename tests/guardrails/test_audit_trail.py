"""The audit writer: coverage of all four event categories, and tamper evidence.

Immutability is proven in ``tests/integration/test_audit_immutability.py`` -- the
database refuses to alter a row at all. What is tested here is the property that
sits on top of it: the hash chain, which makes a break detectable even by
something that bypasses the application entirely.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from yieldloop.db.enums import (
    AuditEventType,
    GuardrailOutcome,
    GuardrailStage,
)
from yieldloop.db.models import AuditRecord, GuardrailAction
from yieldloop.guardrails.audit import GENESIS_DIGEST, AuditLog, compute_digest

pytestmark = pytest.mark.postgres


@pytest.fixture
def audit(db_session: Session) -> AuditLog:
    return AuditLog(db_session)


def test_every_required_event_category_is_writable(
    audit: AuditLog, db_session: Session
) -> None:
    """All four categories the design requires must be recordable."""
    audit.record_model_output(
        actor="agent", request_id="r1", subject_id="hr-1", payload={"abstained": False}
    )
    audit.record_human_decision(
        reviewer_id="reviewer-a", request_id="r1", decision_id="d-1", payload={"action": "edit"}
    )
    audit.record_guardrail_action(
        actor="agent",
        request_id="r1",
        stage=GuardrailStage.GROUNDING,
        outcome=GuardrailOutcome.BLOCKED,
        reason="unresolvable_evidence_id",
        detail={"ids": ["pe:ghost:1"]},
    )
    audit.record_threshold_change(
        changed_by="operator-a",
        request_id="r1",
        name="confidence_floor",
        old_value=0.55,
        new_value=0.60,
        rationale="override rate rose in the lowest band",
    )
    db_session.flush()

    recorded = set(
        db_session.execute(select(AuditRecord.event_type)).scalars().all()
    )
    assert recorded == set(AuditEventType)


def test_first_record_links_to_genesis(audit: AuditLog, db_session: Session) -> None:
    record = audit.record_model_output(
        actor="agent", request_id="r1", subject_id="hr-1", payload={"k": 1}
    )
    db_session.flush()
    assert record.prev_digest is None
    assert record.digest == compute_digest(
        prev_digest=None,
        event_type=AuditEventType.MODEL_OUTPUT,
        actor="agent",
        subject_type="hypothesis_request",
        subject_id="hr-1",
        request_id="r1",
        payload={"k": 1},
    )


def test_each_record_links_to_its_predecessor(audit: AuditLog, db_session: Session) -> None:
    first = audit.record_model_output(
        actor="agent", request_id="r1", subject_id="hr-1", payload={"n": 1}
    )
    second = audit.record_model_output(
        actor="agent", request_id="r2", subject_id="hr-2", payload={"n": 2}
    )
    db_session.flush()
    assert second.prev_digest == first.digest
    assert second.sequence > first.sequence


def test_chain_verifies_clean(audit: AuditLog, db_session: Session) -> None:
    for index in range(8):
        audit.record_model_output(
            actor="agent",
            request_id=f"r{index}",
            subject_id=f"hr-{index}",
            payload={"n": index},
        )
    db_session.flush()
    verification = audit.verify_chain()
    assert verification.checked == 8
    assert verification.intact
    assert verification.broken_at == ()


def test_chain_detects_a_payload_altered_outside_the_application(
    audit: AuditLog, db_session: Session, migrated_engine: object
) -> None:
    """Tamper evidence.

    The application cannot alter a record -- the trigger forbids it. This
    simulates an actor with enough privilege to disable the trigger, which is the
    only scenario where the hash chain is the remaining line of defence.
    """
    for index in range(5):
        audit.record_model_output(
            actor="agent",
            request_id=f"r{index}",
            subject_id=f"hr-{index}",
            payload={"n": index},
        )
    db_session.flush()
    assert audit.verify_chain().intact

    target = db_session.execute(
        select(AuditRecord).order_by(AuditRecord.sequence).offset(2).limit(1)
    ).scalar_one()
    tampered_sequence = target.sequence

    db_session.execute(text("ALTER TABLE audit_records DISABLE TRIGGER audit_records_no_update"))
    db_session.execute(
        text("UPDATE audit_records SET payload = '{\"n\": 999}'::jsonb WHERE id = :id"),
        {"id": target.id},
    )
    db_session.execute(text("ALTER TABLE audit_records ENABLE TRIGGER audit_records_no_update"))
    db_session.expire_all()

    verification = audit.verify_chain()
    assert not verification.intact
    assert tampered_sequence in verification.broken_at


def test_guardrail_action_is_written_to_both_places(
    audit: AuditLog, db_session: Session
) -> None:
    """The dedicated table drives dashboards; the audit record makes it
    non-repudiable. Losing either would leave a gap."""
    action, record = audit.record_guardrail_action(
        actor="agent",
        request_id="r1",
        stage=GuardrailStage.INJECTION,
        outcome=GuardrailOutcome.MODIFIED,
        reason="instruction_shaped_content",
        detail={"signals": ["override_instruction"]},
    )
    db_session.flush()

    stored = db_session.execute(select(GuardrailAction)).scalar_one()
    assert stored.id == action.id
    assert stored.stage is GuardrailStage.INJECTION
    assert stored.outcome is GuardrailOutcome.MODIFIED
    assert record.subject_id == str(action.id)
    assert record.payload["reason"] == "instruction_shaped_content"


def test_digest_is_independent_of_key_ordering() -> None:
    """Two semantically identical payloads must hash identically."""
    common = {
        "prev_digest": GENESIS_DIGEST,
        "event_type": AuditEventType.MODEL_OUTPUT,
        "actor": "agent",
        "subject_type": "hypothesis_request",
        "subject_id": "hr-1",
        "request_id": "r1",
    }
    assert compute_digest(**common, payload={"a": 1, "b": 2}) == compute_digest(  # type: ignore[arg-type]
        **common, payload={"b": 2, "a": 1}  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    "field",
    ["actor", "subject_type", "subject_id", "request_id"],
)
def test_every_meaningful_field_is_covered_by_the_digest(field: str) -> None:
    """A field left out of the digest could be altered without breaking the chain."""
    base: dict[str, object] = {
        "prev_digest": GENESIS_DIGEST,
        "event_type": AuditEventType.MODEL_OUTPUT,
        "actor": "agent",
        "subject_type": "hypothesis_request",
        "subject_id": "hr-1",
        "request_id": "r1",
        "payload": {"n": 1},
    }
    altered = {**base, field: "tampered"}
    assert compute_digest(**base) != compute_digest(**altered)  # type: ignore[arg-type]


def test_payload_change_breaks_the_digest() -> None:
    base: dict[str, object] = {
        "prev_digest": GENESIS_DIGEST,
        "event_type": AuditEventType.MODEL_OUTPUT,
        "actor": "agent",
        "subject_type": "hypothesis_request",
        "subject_id": "hr-1",
        "request_id": "r1",
    }
    assert compute_digest(**base, payload={"n": 1}) != compute_digest(  # type: ignore[arg-type]
        **base, payload={"n": 2}  # type: ignore[arg-type]
    )
