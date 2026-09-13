"""Seed the reason code vocabulary.

Revision ID: c2d4f6a80002
Revises: a1c3e5f70001
Create Date: 2026-09-13

``decisions.reason_code`` is a foreign key into ``reason_codes``, and nothing
populated that table. Every edit and reject in a fresh install therefore failed
with a foreign key violation -- found by running the end-to-end suite against a
real database, where the integration tests had been seeding the vocabulary in a
fixture and hiding it.

The codes are defined in ``yieldloop.review.reason_codes`` and inserted here
rather than being duplicated as literals, so the vocabulary has one definition.
The insert is idempotent: re-running updates the labels and descriptions of codes
that already exist rather than failing, which is what makes it safe to re-apply
after a wording change.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from yieldloop.review.reason_codes import REASON_CODES

revision: str = "c2d4f6a80002"
down_revision: str | None = "a1c3e5f70001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json_array(values: list[str]) -> str:
    inner = ", ".join(f'"{value}"' for value in values)
    return f"[{inner}]"


def upgrade() -> None:
    connection = op.get_bind()
    for spec in REASON_CODES:
        connection.execute(
            sa.text(
                """
                INSERT INTO reason_codes
                    (code, label, description, applies_to, applies_to_gates,
                     is_active, sort_order, created_at)
                VALUES
                    (:code, :label, :description, CAST(:applies_to AS jsonb),
                     CAST(:applies_to_gates AS jsonb), true, :sort_order, now())
                ON CONFLICT (code) DO UPDATE SET
                    label = EXCLUDED.label,
                    description = EXCLUDED.description,
                    applies_to = EXCLUDED.applies_to,
                    applies_to_gates = EXCLUDED.applies_to_gates,
                    sort_order = EXCLUDED.sort_order,
                    is_active = true
                """
            ),
            {
                "code": spec.code,
                "label": spec.label,
                "description": spec.description,
                "applies_to": _json_array([a.value for a in spec.applies_to]),
                "applies_to_gates": _json_array([g.value for g in spec.applies_to_gates]),
                "sort_order": spec.sort_order,
            },
        )


def downgrade() -> None:
    connection = op.get_bind()
    connection.execute(
        sa.text("DELETE FROM reason_codes WHERE code = ANY(:codes)"),
        {"codes": [spec.code for spec in REASON_CODES]},
    )
