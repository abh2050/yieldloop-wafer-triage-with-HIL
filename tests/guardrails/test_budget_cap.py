"""Spend ceilings, enforced against the real ledger table.

These run against a real Postgres because that is where the ceiling actually
lives. An in-process counter would pass a test and then fail in production the
moment a second worker started or the API restarted, so the guard reads committed
spend from ``cost_ledger`` and the tests exercise that path.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from yieldloop.config import Settings
from yieldloop.guardrails import BudgetExceededError
from yieldloop.guardrails.budget import BudgetGuard, BudgetLimits, Pricing

pytestmark = pytest.mark.postgres

TODAY = date(2026, 9, 13)
SESSION = "session-under-test"

PRICING = Pricing(input_per_mtok_usd=2.50, output_per_mtok_usd=10.00)
LIMITS = BudgetLimits(
    max_prompt_tokens=12_000,
    max_completion_tokens=2_000,
    session_cost_cap_usd=1.00,
    daily_cost_cap_usd=25.00,
)


@pytest.fixture
def guard(db_session: Session) -> BudgetGuard:
    return BudgetGuard(db_session, LIMITS, PRICING)


# --- pricing ---------------------------------------------------------------


def test_cost_is_computed_from_configured_prices() -> None:
    assert PRICING.cost(1_000_000, 0) == pytest.approx(2.50)
    assert PRICING.cost(0, 1_000_000) == pytest.approx(10.00)
    assert PRICING.cost(12_000, 2_000) == pytest.approx(0.05)
    assert PRICING.cost(0, 0) == 0.0


def test_negative_token_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PRICING.cost(-1, 0)


def test_pricing_comes_from_settings() -> None:
    pricing = Pricing.from_settings(
        Settings(input_cost_per_mtok_usd=1.0, output_cost_per_mtok_usd=4.0)
    )
    assert pricing.cost(1_000_000, 1_000_000) == pytest.approx(5.0)


# --- per-request token caps ------------------------------------------------


def test_prompt_within_the_cap_is_allowed(guard: BudgetGuard) -> None:
    guard.check_tokens(prompt_tokens=12_000, completion_tokens=2_000)


def test_oversized_prompt_is_refused_before_the_call(guard: BudgetGuard) -> None:
    with pytest.raises(BudgetExceededError) as exc:
        guard.check_tokens(prompt_tokens=12_001, completion_tokens=100)
    assert exc.value.detail["code"] == "prompt_tokens_exceeded"
    assert exc.value.reason == "budget_exceeded"
    assert exc.value.status_code == 429


def test_oversized_completion_request_is_refused(guard: BudgetGuard) -> None:
    with pytest.raises(BudgetExceededError) as exc:
        guard.check_tokens(prompt_tokens=100, completion_tokens=2_001)
    assert exc.value.detail["code"] == "completion_tokens_exceeded"


# --- session and daily ceilings --------------------------------------------


def test_spend_starts_at_zero(guard: BudgetGuard) -> None:
    snapshot = guard.spend(session_id=SESSION, spend_date=TODAY)
    assert snapshot.session_usd == 0.0
    assert snapshot.daily_usd == 0.0


def test_recorded_usage_accumulates(guard: BudgetGuard, db_session: Session) -> None:
    for index in range(3):
        guard.record(
            session_id=SESSION,
            request_id=f"req-{index}",
            spend_date=TODAY,
            model="gpt-4o-2024-08-06",
            prompt_tokens=100_000,
            completion_tokens=10_000,
        )
    db_session.flush()
    snapshot = guard.spend(session_id=SESSION, spend_date=TODAY)
    assert snapshot.session_usd == pytest.approx(3 * (0.25 + 0.10))
    assert snapshot.daily_usd == pytest.approx(snapshot.session_usd)


def test_session_cap_refuses_the_call_that_would_breach_it(
    guard: BudgetGuard, db_session: Session
) -> None:
    """Checked before the call, not after. Afterwards the money is already gone."""
    guard.record(
        session_id=SESSION,
        request_id="req-large",
        spend_date=TODAY,
        model="gpt-4o-2024-08-06",
        prompt_tokens=390_000,
        completion_tokens=0,
    )
    db_session.flush()

    with pytest.raises(BudgetExceededError) as exc:
        guard.check_spend(
            session_id=SESSION,
            spend_date=TODAY,
            estimated_prompt_tokens=12_000,
            estimated_completion_tokens=2_000,
        )
    assert exc.value.detail["code"] == "session_cost_exceeded"
    assert float(exc.value.detail["spent_usd"]) == pytest.approx(0.975)  # type: ignore[arg-type]


def test_landing_exactly_on_the_cap_is_permitted(
    guard: BudgetGuard, db_session: Session
) -> None:
    """The cap is a ceiling that may be reached, not one that may be approached.

    $0.95 already spent plus a $0.05 call is exactly the $1.00 session cap, and
    is allowed; one cent more is not. Pinned because an off-by-one here either
    refuses legitimate work or lets every cap be breached by one call.
    """
    guard.record(
        session_id=SESSION,
        request_id="req-exact",
        spend_date=TODAY,
        model="gpt-4o-2024-08-06",
        prompt_tokens=380_000,
        completion_tokens=0,
    )
    db_session.flush()
    assert guard.spend(session_id=SESSION, spend_date=TODAY).session_usd == pytest.approx(0.95)

    guard.check_spend(
        session_id=SESSION,
        spend_date=TODAY,
        estimated_prompt_tokens=12_000,
        estimated_completion_tokens=2_000,
    )

    with pytest.raises(BudgetExceededError):
        guard.check_spend(
            session_id=SESSION,
            spend_date=TODAY,
            estimated_prompt_tokens=16_000,
            estimated_completion_tokens=2_000,
        )


def test_spend_just_under_the_cap_is_allowed(
    guard: BudgetGuard, db_session: Session
) -> None:
    guard.record(
        session_id=SESSION,
        request_id="req-small",
        spend_date=TODAY,
        model="gpt-4o-2024-08-06",
        prompt_tokens=100_000,
        completion_tokens=0,
    )
    db_session.flush()
    estimated = guard.check_spend(
        session_id=SESSION,
        spend_date=TODAY,
        estimated_prompt_tokens=12_000,
        estimated_completion_tokens=2_000,
    )
    assert estimated == pytest.approx(0.05)


def test_daily_cap_spans_sessions(guard: BudgetGuard, db_session: Session) -> None:
    """A per-session cap alone does not bound the bill; a new session resets it."""
    for index in range(30):
        guard.record(
            session_id=f"session-{index}",
            request_id=f"daily-{index}",
            spend_date=TODAY,
            model="gpt-4o-2024-08-06",
            prompt_tokens=340_000,
            completion_tokens=0,
        )
    db_session.flush()

    snapshot = guard.spend(session_id="session-fresh", spend_date=TODAY)
    assert snapshot.session_usd == 0.0
    assert snapshot.daily_usd == pytest.approx(25.5)

    with pytest.raises(BudgetExceededError) as exc:
        guard.check_spend(
            session_id="session-fresh",
            spend_date=TODAY,
            estimated_prompt_tokens=1_000,
            estimated_completion_tokens=100,
        )
    assert exc.value.detail["code"] == "daily_cost_exceeded"


def test_yesterdays_spend_does_not_count_against_today(
    guard: BudgetGuard, db_session: Session
) -> None:
    guard.record(
        session_id="session-yesterday",
        request_id="yesterday-1",
        spend_date=TODAY - timedelta(days=1),
        model="gpt-4o-2024-08-06",
        prompt_tokens=9_000_000,
        completion_tokens=0,
    )
    db_session.flush()
    assert guard.spend(session_id=SESSION, spend_date=TODAY).daily_usd == 0.0
    guard.check_spend(
        session_id=SESSION,
        spend_date=TODAY,
        estimated_prompt_tokens=12_000,
        estimated_completion_tokens=2_000,
    )


def test_estimate_uses_the_maximum_completion_not_an_expectation(
    guard: BudgetGuard,
) -> None:
    """Under-estimating is what lets the very call being checked breach the cap."""
    estimated = guard.check_spend(
        session_id=SESSION,
        spend_date=TODAY,
        estimated_prompt_tokens=12_000,
        estimated_completion_tokens=2_000,
    )
    assert estimated == pytest.approx(PRICING.cost(12_000, 2_000))


def test_a_retried_write_cannot_double_count(
    guard: BudgetGuard, db_session: Session
) -> None:
    guard.record(
        session_id=SESSION,
        request_id="req-idempotent",
        spend_date=TODAY,
        model="gpt-4o-2024-08-06",
        prompt_tokens=1_000,
        completion_tokens=100,
    )
    db_session.flush()
    with pytest.raises(IntegrityError):
        guard.record(
            session_id=SESSION,
            request_id="req-idempotent",
            spend_date=TODAY,
            model="gpt-4o-2024-08-06",
            prompt_tokens=1_000,
            completion_tokens=100,
        )
        db_session.flush()
    db_session.rollback()
