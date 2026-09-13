"""Request-scoped dependencies.

One session per request, committed on success and rolled back on any exception.
That boundary is what makes a decision write and its audit record atomic: a
decision that reached the database without its audit entry would be exactly the
gap the audit log exists to close.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Header, Request
from sqlalchemy.orm import Session

from yieldloop.config import Settings, get_settings
from yieldloop.db.session import get_sessionmaker
from yieldloop.guardrails import InputRejectedError
from yieldloop.guardrails.input_filter import InputFilter, validate_reviewer_id


def session_dependency() -> Iterator[Session]:
    """Yield a transactional session for one request."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def settings_dependency() -> Settings:
    return get_settings()


def request_id_dependency(request: Request) -> str:
    """The correlation id shared by every audit record for this request."""
    existing: str | None = getattr(request.state, "request_id", None)
    if existing:
        return existing
    generated = uuid.uuid4().hex
    request.state.request_id = generated
    return generated


def input_filter_dependency(
    settings: Annotated[Settings, Depends(settings_dependency)],
) -> InputFilter:
    return InputFilter.from_settings(settings)


def reviewer_dependency(
    x_reviewer_id: Annotated[str | None, Header(alias="X-Reviewer-Id")] = None,
) -> str:
    """Identify the reviewer.

    A header rather than a session cookie, because this console is deployed
    behind an existing fab SSO proxy that injects it. The value is validated the
    same as any other input: an unvalidated identity string ends up in the audit
    log, which is the last place it should be possible to write arbitrary text.
    """
    if not x_reviewer_id:
        raise InputRejectedError(
            "X-Reviewer-Id header is required; every decision is attributed",
            detail={"field": "X-Reviewer-Id", "code": "reviewer_required"},
        )
    return validate_reviewer_id(x_reviewer_id)


SessionDep = Annotated[Session, Depends(session_dependency)]
SettingsDep = Annotated[Settings, Depends(settings_dependency)]
RequestIdDep = Annotated[str, Depends(request_id_dependency)]
InputFilterDep = Annotated[InputFilter, Depends(input_filter_dependency)]
ReviewerDep = Annotated[str, Depends(reviewer_dependency)]
