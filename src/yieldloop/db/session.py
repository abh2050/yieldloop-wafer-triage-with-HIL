"""Engine and session management.

The application connects as a role that holds no UPDATE or DELETE grant on
``audit_records``. That revocation is applied by the migration, so an attempt to
mutate the audit log raises a database error regardless of what the calling code
believes it is allowed to do.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import Pool

from yieldloop.config import Settings, get_settings


def build_engine(settings: Settings | None = None) -> Engine:
    """Create a new engine. Prefer :func:`get_engine` outside of tests."""
    resolved = settings if settings is not None else get_settings()
    engine = create_engine(
        str(resolved.database_url),
        pool_size=resolved.db_pool_size,
        max_overflow=resolved.db_pool_max_overflow,
        pool_pre_ping=True,
        future=True,
    )
    _install_statement_timeout(engine, resolved.db_statement_timeout_ms)
    return engine


def _install_statement_timeout(engine: Engine, timeout_ms: int) -> None:
    """Apply a server-side statement timeout to every pooled connection.

    A runaway query on the review queue would stall the console, and the
    reviewer-throughput claim is only meaningful if the tail is bounded.
    """

    @event.listens_for(engine, "connect")
    def _set_timeout(dbapi_connection: Any, _record: Pool) -> None:
        with dbapi_connection.cursor() as cursor:
            cursor.execute(f"SET statement_timeout = {timeout_ms}")

    return None


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide engine."""
    return build_engine()


@lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker[Session]:
    """Return the process-wide session factory."""
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back on any exception."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connectivity(engine: Engine | None = None) -> bool:
    """Return True when the database answers a trivial query."""
    target = engine if engine is not None else get_engine()
    with target.connect() as connection:
        result: object = connection.execute(text("SELECT 1")).scalar_one()
    return result == 1
