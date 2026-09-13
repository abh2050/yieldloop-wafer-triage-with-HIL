"""Make audit_records append only at the database level.

Revision ID: a1c3e5f70001
Revises: df7f4495e384
Create Date: 2026-09-13

The audit log is the artifact that makes every other claim in this system
checkable, so its immutability cannot depend on application code remembering not
to issue an UPDATE. Two independent mechanisms are applied here.

1. **Revoked grants.** UPDATE, DELETE, and TRUNCATE are revoked from PUBLIC and
   from the application role. This is the mechanism the design calls for and it
   stops any ordinary application path.

2. **A trigger that raises.** Grants alone are not sufficient, because in
   Postgres a table owner retains implicit rights and can re-grant to itself; the
   application connects as the owner in the local Compose stack. A
   ``BEFORE UPDATE OR DELETE OR TRUNCATE`` trigger raises an exception regardless
   of privilege, so the guarantee holds even for the owner. Defeating it requires
   an explicit ``ALTER TABLE ... DISABLE TRIGGER``, which is itself a DDL event
   rather than something a bug or an injected statement can do silently.

Both are asserted by ``tests/integration/test_audit_immutability.py`` against a
real Postgres instance.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "a1c3e5f70001"
down_revision: str | None = "df7f4495e384"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION yieldloop_reject_audit_mutation()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RAISE EXCEPTION
                'audit_records is append only; % is not permitted', TG_OP
                USING ERRCODE = 'restrict_violation';
        END;
        $$;
        """
    )
    op.execute(
        """
        CREATE TRIGGER audit_records_no_update
            BEFORE UPDATE ON audit_records
            FOR EACH ROW EXECUTE FUNCTION yieldloop_reject_audit_mutation();
        """
    )
    op.execute(
        """
        CREATE TRIGGER audit_records_no_delete
            BEFORE DELETE ON audit_records
            FOR EACH ROW EXECUTE FUNCTION yieldloop_reject_audit_mutation();
        """
    )
    op.execute(
        """
        CREATE TRIGGER audit_records_no_truncate
            BEFORE TRUNCATE ON audit_records
            FOR EACH STATEMENT EXECUTE FUNCTION yieldloop_reject_audit_mutation();
        """
    )

    # The grant revocation the design calls for. CURRENT_USER is the role the
    # migration runs as, which is the role the application connects as.
    op.execute("REVOKE UPDATE, DELETE, TRUNCATE ON audit_records FROM PUBLIC;")
    op.execute("REVOKE UPDATE, DELETE, TRUNCATE ON audit_records FROM CURRENT_USER;")
    # Inserting and reading remain available; nothing else does.
    op.execute("GRANT INSERT, SELECT ON audit_records TO CURRENT_USER;")

    # The digest chain is only tamper evident if the sequence is strictly
    # increasing and never reused. Ownership of the sequence stays with the
    # table, but cycling is explicitly disallowed.
    op.execute("ALTER SEQUENCE audit_records_sequence_seq NO CYCLE;")


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS audit_records_no_truncate ON audit_records;")
    op.execute("DROP TRIGGER IF EXISTS audit_records_no_delete ON audit_records;")
    op.execute("DROP TRIGGER IF EXISTS audit_records_no_update ON audit_records;")
    op.execute("DROP FUNCTION IF EXISTS yieldloop_reject_audit_mutation();")
    op.execute("GRANT UPDATE, DELETE, TRUNCATE ON audit_records TO CURRENT_USER;")
