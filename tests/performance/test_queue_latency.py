"""Latency budgets against real data volume.

Reviewer throughput is the product claim, so these are correctness tests rather
than nice-to-haves: a queue endpoint that degrades as the wafer table grows makes
the sub-four-second decision target unreachable no matter how good the keyboard
flow is.

They run against the real database with the real WM811K volume when it is
present, and skip when it is not. A latency measured against an empty table would
be meaningless.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from yieldloop.config import Settings
from yieldloop.db.enums import SplitName, TaskGate
from yieldloop.db.models import Wafer
from yieldloop.db.session import build_engine
from yieldloop.models.embed import load_samples
from yieldloop.review.queue import next_items, queue_depth

pytestmark = [pytest.mark.performance, pytest.mark.timeout(180)]

#: The queue endpoint is on the critical path of every decision.
QUEUE_BUDGET_SECONDS = 0.5
#: Depth drives the progress counter and is polled alongside the queue.
DEPTH_BUDGET_SECONDS = 1.0
#: A round scores a pool of this size; slower than this and rounds stop being
#: something an operator runs interactively.
LOAD_BUDGET_WAFERS_PER_SECOND = 2_000

MIN_VOLUME = 100_000


@pytest.fixture(scope="module")
def live_session() -> Iterator[Session]:
    """A session against the configured database, not the test container.

    These assert behaviour at production volume, which only the real ingested
    database has.
    """
    engine = build_engine()
    session = Session(engine)
    try:
        count = int(session.execute(select(func.count()).select_from(Wafer)).scalar_one())
    except Exception:
        session.close()
        pytest.skip("configured database is unreachable")
    if count < MIN_VOLUME:
        session.close()
        pytest.skip(
            f"database holds {count:,} wafers, fewer than {MIN_VOLUME:,}; "
            "run scripts/bootstrap_db.py for a meaningful latency measurement"
        )
    yield session
    session.close()


def _timed(operation: object, repeats: int = 5) -> float:
    """Median wall time over repeats, to blunt one-off scheduling noise."""
    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()  # type: ignore[operator]
        timings.append(time.perf_counter() - started)
    return sorted(timings)[len(timings) // 2]


def test_wafer_volume_is_realistic(live_session: Session) -> None:
    count = int(live_session.execute(select(func.count()).select_from(Wafer)).scalar_one())
    assert count >= MIN_VOLUME


@pytest.mark.parametrize("gate", [TaskGate.LABEL, TaskGate.CONFIRM])
def test_queue_fetch_is_within_budget(live_session: Session, gate: TaskGate) -> None:
    """The queue join must not degrade with table size.

    Fetched with a single join rather than per-task lookups; an N+1 here would
    show up directly in seconds per decision.
    """
    elapsed = _timed(lambda: next_items(live_session, gate=gate, limit=25))
    assert elapsed < QUEUE_BUDGET_SECONDS, (
        f"{gate.value} queue took {elapsed:.3f}s, over the {QUEUE_BUDGET_SECONDS}s budget"
    )


def test_queue_depth_is_within_budget(live_session: Session) -> None:
    elapsed = _timed(lambda: queue_depth(live_session, gate=TaskGate.LABEL))
    assert elapsed < DEPTH_BUDGET_SECONDS


def test_a_large_queue_page_stays_within_budget(live_session: Session) -> None:
    """The grid renders dozens at once; the page size must not be the bottleneck."""
    elapsed = _timed(lambda: next_items(live_session, gate=TaskGate.LABEL, limit=200))
    assert elapsed < QUEUE_BUDGET_SECONDS * 2


def test_single_wafer_lookup_is_indexed(live_session: Session) -> None:
    """wafer_id is unique and indexed; without that this scans 811,457 rows."""
    elapsed = _timed(
        lambda: live_session.execute(
            select(Wafer).where(Wafer.wafer_id == "lot1-1")
        ).scalar_one_or_none()
    )
    assert elapsed < 0.1, f"indexed wafer lookup took {elapsed:.3f}s"


def test_bulk_wafer_loading_is_fast_enough_for_a_round(live_session: Session) -> None:
    """Scoring a round must not be dominated by fetching the pool.

    This path was originally an ORM entity load and took minutes for 120,000
    wafers; it selects the four needed columns and streams them instead.
    """
    sample_size = 20_000
    started = time.perf_counter()
    samples = load_samples(live_session, SplitName.TRAIN, labeled=True, limit=sample_size)
    elapsed = time.perf_counter() - started

    assert samples, "no labeled training wafers"
    rate = len(samples) / elapsed
    assert rate > LOAD_BUDGET_WAFERS_PER_SECOND, (
        f"loaded {len(samples):,} wafers at {rate:,.0f}/s, under the "
        f"{LOAD_BUDGET_WAFERS_PER_SECOND:,}/s budget"
    )


def test_connection_pool_is_bounded(live_session: Session) -> None:
    """An unbounded pool exhausts Postgres connections under concurrent review."""
    settings = Settings()
    assert settings.db_pool_size > 0
    assert settings.db_pool_max_overflow >= 0
    # A statement timeout bounds the tail; without it one runaway query stalls
    # the console, and the throughput claim is about the tail, not the mean.
    assert settings.db_statement_timeout_ms > 0
