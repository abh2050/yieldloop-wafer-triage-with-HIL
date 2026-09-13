"""Shared test fixtures.

There are no mocks in this suite. Database tests run against a real Postgres
started for the session by testcontainers and migrated with the real Alembic
chain. Tests that need the OpenAI API call it. Tests that need WM811K read the
real archive. Each of those is expressed as a marker that *skips* when the
resource is genuinely unavailable, never as a substitute that pretends it is.
"""

from __future__ import annotations

import os

# Must be set before torch initializes its thread pool. Running the full suite
# with torch's default multi-threaded CPU backend segfaults on macOS inside
# softmax once several test modules have exercised torch -- two OpenMP runtimes
# end up loaded in one process. Pinning to a single thread is the documented
# workaround, costs nothing here (the tensors in these tests are tiny), and is
# far better than a suite that crashes intermittently and gets rerun until green.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.community.postgres import PostgresContainer

from yieldloop.config import Settings
from yieldloop.db.session import build_engine

torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = REPO_ROOT / "src" / "yieldloop" / "db" / "migrations" / "alembic.ini"

#: Tables truncated between tests. audit_records is absent by design: it is
#: append only and the database refuses to truncate it, which is the property
#: tests/integration/test_audit_immutability.py exists to prove.
_RESETTABLE_TABLES = (
    "decisions",
    "review_tasks",
    "active_rounds",
    "hypothesis_citations",
    "hypotheses",
    "hypothesis_requests",
    "process_events",
    "historical_excursions",
    "predictions",
    "model_artifacts",
    "wafers",
    "lots",
    "guardrail_actions",
    "cost_ledger",
    "breaker_events",
    "threshold_changes",
    "drift_snapshots",
    "reason_codes",
)


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    """Start a real Postgres for the session."""
    container = PostgresContainer("postgres:16-alpine", driver="psycopg")
    container.start()
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="session")
def db_settings(postgres_container: PostgresContainer) -> Settings:
    """Settings pointed at the containerized database."""
    return Settings(database_url=postgres_container.get_connection_url())


@pytest.fixture(scope="session")
def migrated_engine(db_settings: Settings) -> Iterator[Engine]:
    """Apply the real migration chain, including the audit immutability rules."""
    engine = build_engine(db_settings)
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(ALEMBIC_INI.parent))
    config.set_main_option("sqlalchemy.url", str(db_settings.database_url))
    command.upgrade(config, "head")
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_session(migrated_engine: Engine) -> Iterator[Session]:
    """A session against the migrated database, with tables reset afterwards."""
    factory = sessionmaker(bind=migrated_engine, expire_on_commit=False, future=True)
    session = factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        with migrated_engine.begin() as connection:
            connection.execute(
                text(f"TRUNCATE {', '.join(_RESETTABLE_TABLES)} RESTART IDENTITY CASCADE")
            )


@pytest.fixture(scope="session")
def openai_api_key() -> str:
    """The live API key, or skip. Never a placeholder."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        pytest.skip("OPENAI_API_KEY is not set; live contract tests cannot run")
    return key


@pytest.fixture(scope="session")
def wm811k_path() -> Path:
    """The real WM811K archive, or skip. Never a generated substitute."""
    settings = Settings()
    path = settings.wm811k_path
    if not path.is_file():
        pytest.skip(
            f"{path} is not present; run `python scripts/fetch_dataset.py`. "
            "yieldloop has no generated fallback dataset."
        )
    return path
