"""The audit log must be immutable as a property of the database.

These run against a real Postgres started by the test session and migrated with
the real Alembic chain. Nothing is mocked, because the guarantee under test *is*
the database's behaviour -- a fake session could be made to pass while the
deployed system silently allowed an UPDATE.
"""

from __future__ import annotations

import hashlib
import uuid

import pytest
from sqlalchemy import Engine, func, select, text
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import Session

from yieldloop.db.enums import AuditEventType
from yieldloop.db.models import AuditRecord

pytestmark = pytest.mark.postgres


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()


def _record(marker: str, prev: str | None = None) -> AuditRecord:
    return AuditRecord(
        event_type=AuditEventType.MODEL_OUTPUT,
        actor="test-harness",
        subject_type="wafer",
        subject_id=f"wafer-{marker}",
        request_id=f"req-{marker}",
        payload={"marker": marker},
        prev_digest=prev,
        digest=_digest(f"{prev}:{marker}"),
    )


def test_insert_is_permitted_and_assigns_a_sequence(db_session: Session) -> None:
    record = _record(uuid.uuid4().hex)
    db_session.add(record)
    db_session.commit()
    db_session.refresh(record)
    assert record.sequence >= 1
    assert record.occurred_at is not None


def test_update_is_rejected_by_the_database(db_session: Session) -> None:
    record = _record(uuid.uuid4().hex)
    db_session.add(record)
    db_session.commit()

    with pytest.raises(DatabaseError, match="append only"):
        db_session.execute(
            text("UPDATE audit_records SET actor = :actor WHERE id = :id"),
            {"actor": "tampered", "id": record.id},
        )
    db_session.rollback()

    assert db_session.get(AuditRecord, record.id) is not None
    stored = db_session.execute(
        select(AuditRecord.actor).where(AuditRecord.id == record.id)
    ).scalar_one()
    assert stored == "test-harness"


def test_delete_is_rejected_by_the_database(db_session: Session) -> None:
    record = _record(uuid.uuid4().hex)
    db_session.add(record)
    db_session.commit()

    with pytest.raises(DatabaseError, match="append only"):
        db_session.execute(text("DELETE FROM audit_records WHERE id = :id"), {"id": record.id})
    db_session.rollback()

    assert db_session.get(AuditRecord, record.id) is not None


def test_truncate_is_rejected_by_the_database(db_session: Session) -> None:
    db_session.add(_record(uuid.uuid4().hex))
    db_session.commit()
    before = db_session.execute(select(func.count()).select_from(AuditRecord)).scalar_one()

    with pytest.raises(DatabaseError, match="append only"):
        db_session.execute(text("TRUNCATE audit_records"))
    db_session.rollback()

    after = db_session.execute(select(func.count()).select_from(AuditRecord)).scalar_one()
    assert after == before


def test_orm_mutation_is_rejected_too(db_session: Session) -> None:
    """The guarantee must not depend on callers using raw SQL."""
    record = _record(uuid.uuid4().hex)
    db_session.add(record)
    db_session.commit()

    record.actor = "tampered-via-orm"
    with pytest.raises(DatabaseError, match="append only"):
        db_session.commit()
    db_session.rollback()


def test_update_and_delete_grants_are_revoked(db_session: Session) -> None:
    """The revoked grants the design calls for are actually absent.

    Checked independently of the trigger, so that removing one mechanism cannot
    quietly leave the other carrying the whole guarantee.
    """
    granted = (
        db_session.execute(
            text(
                """
            SELECT privilege_type FROM information_schema.role_table_grants
            WHERE table_name = 'audit_records' AND grantee = CURRENT_USER
            """
            )
        )
        .scalars()
        .all()
    )
    assert "INSERT" in granted
    assert "SELECT" in granted
    assert "UPDATE" not in granted
    assert "DELETE" not in granted
    assert "TRUNCATE" not in granted


def test_sequence_is_generated_always(db_session: Session) -> None:
    """The application must not be able to supply or override the sequence."""
    with pytest.raises(DatabaseError):
        db_session.execute(
            text(
                """
                INSERT INTO audit_records
                    (id, sequence, event_type, actor, subject_type, subject_id,
                     request_id, payload, digest)
                VALUES (gen_random_uuid(), 999999, 'model_output', 'attacker', 'wafer',
                        'w', 'r', '{}'::jsonb, :digest)
                """
            ),
            {"digest": _digest("forced-sequence")},
        )
    db_session.rollback()


def test_sequence_is_monotonic_across_inserts(db_session: Session) -> None:
    """Gaps are acceptable; reordering is not. The digest chain relies on it."""
    records = []
    previous: str | None = None
    for index in range(5):
        record = _record(f"{uuid.uuid4().hex}-{index}", prev=previous)
        db_session.add(record)
        db_session.commit()
        db_session.refresh(record)
        records.append(record)
        previous = record.digest

    sequences = [record.sequence for record in records]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)


def test_digest_chain_links_each_record_to_its_predecessor(db_session: Session) -> None:
    previous: str | None = None
    digests: list[tuple[str | None, str]] = []
    for index in range(4):
        marker = f"{uuid.uuid4().hex}-{index}"
        record = _record(marker, prev=previous)
        db_session.add(record)
        db_session.commit()
        digests.append((previous, record.digest))
        previous = record.digest

    for prev_digest, digest in digests[1:]:
        assert prev_digest is not None
        assert digest != prev_digest


def test_duplicate_digest_is_rejected(db_session: Session, migrated_engine: Engine) -> None:
    marker = uuid.uuid4().hex
    db_session.add(_record(marker))
    db_session.commit()

    duplicate = _record(marker)
    duplicate.subject_id = "different-subject"
    db_session.add(duplicate)
    with pytest.raises(DatabaseError):
        db_session.commit()
    db_session.rollback()
