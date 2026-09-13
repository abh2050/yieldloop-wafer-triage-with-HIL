"""Token, session, and daily spend ceilings.

Three ceilings, checked in that order:

* **Per request**, on prompt and completion tokens, refused before the call.
* **Per session**, so one reviewer working through a queue cannot run up an
  unbounded bill.
* **Per day**, across every session, as the backstop.

The session and daily figures are read from the ``cost_ledger`` table rather than
from process memory. An in-process counter resets when the API restarts and is
not shared across workers, which makes it exactly useless as a spend ceiling in
the deployment this is built for.

Cost is checked *before* the call using an estimate, and recorded *after* it
using the usage the API actually reports. Estimating conservatively before and
reconciling after is the only way to refuse a call that would breach the cap;
checking only afterwards means the money is already spent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from yieldloop.config import Settings
from yieldloop.db.models import CostLedgerEntry
from yieldloop.guardrails import BudgetExceededError

#: Tokens per million, for converting the configured per-MTok prices.
_TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True, slots=True)
class Pricing:
    """Per-million-token prices for the configured model."""

    input_per_mtok_usd: float
    output_per_mtok_usd: float

    @classmethod
    def from_settings(cls, settings: Settings) -> Pricing:
        return cls(
            input_per_mtok_usd=settings.input_cost_per_mtok_usd,
            output_per_mtok_usd=settings.output_cost_per_mtok_usd,
        )

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Cost in USD for a given usage."""
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError(
                f"token counts must be non-negative; got prompt={prompt_tokens} "
                f"completion={completion_tokens}"
            )
        return (
            prompt_tokens * self.input_per_mtok_usd
            + completion_tokens * self.output_per_mtok_usd
        ) / _TOKENS_PER_MILLION


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    """The four ceilings, all from configuration."""

    max_prompt_tokens: int
    max_completion_tokens: int
    session_cost_cap_usd: float
    daily_cost_cap_usd: float

    @classmethod
    def from_settings(cls, settings: Settings) -> BudgetLimits:
        return cls(
            max_prompt_tokens=settings.max_prompt_tokens,
            max_completion_tokens=settings.max_completion_tokens,
            session_cost_cap_usd=settings.session_cost_cap_usd,
            daily_cost_cap_usd=settings.daily_cost_cap_usd,
        )


@dataclass(frozen=True, slots=True)
class SpendSnapshot:
    """What has already been spent, as recorded in the ledger."""

    session_usd: float
    daily_usd: float


class BudgetGuard:
    """Enforces the ceilings against the real ledger."""

    def __init__(self, session: Session, limits: BudgetLimits, pricing: Pricing) -> None:
        self._session = session
        self._limits = limits
        self._pricing = pricing

    @property
    def limits(self) -> BudgetLimits:
        return self._limits

    @property
    def pricing(self) -> Pricing:
        return self._pricing

    def spend(self, *, session_id: str, spend_date: date) -> SpendSnapshot:
        """Read committed spend for the session and the day."""
        session_total = self._session.execute(
            select(func.coalesce(func.sum(CostLedgerEntry.cost_usd), 0.0)).where(
                CostLedgerEntry.session_id == session_id
            )
        ).scalar_one()
        daily_total = self._session.execute(
            select(func.coalesce(func.sum(CostLedgerEntry.cost_usd), 0.0)).where(
                CostLedgerEntry.spend_date == spend_date
            )
        ).scalar_one()
        return SpendSnapshot(session_usd=float(session_total), daily_usd=float(daily_total))

    def check_tokens(self, *, prompt_tokens: int, completion_tokens: int) -> None:
        """Refuse a request whose token budget exceeds the per-request caps."""
        if prompt_tokens > self._limits.max_prompt_tokens:
            raise BudgetExceededError(
                f"prompt is {prompt_tokens} tokens, over the {self._limits.max_prompt_tokens} "
                "per-request cap",
                detail={
                    "code": "prompt_tokens_exceeded",
                    "requested": prompt_tokens,
                    "limit": self._limits.max_prompt_tokens,
                },
            )
        if completion_tokens > self._limits.max_completion_tokens:
            raise BudgetExceededError(
                f"requested {completion_tokens} completion tokens, over the "
                f"{self._limits.max_completion_tokens} per-request cap",
                detail={
                    "code": "completion_tokens_exceeded",
                    "requested": completion_tokens,
                    "limit": self._limits.max_completion_tokens,
                },
            )

    def check_spend(
        self,
        *,
        session_id: str,
        spend_date: date,
        estimated_prompt_tokens: int,
        estimated_completion_tokens: int,
    ) -> float:
        """Refuse a call that would push spend past a ceiling.

        The estimate uses the *maximum* completion tokens the call could produce,
        not an expected value. Under-estimating here is what allows a cap to be
        breached by the very call that was checked against it.

        Returns:
            The estimated cost of the call, in USD.
        """
        estimated = self._pricing.cost(estimated_prompt_tokens, estimated_completion_tokens)
        snapshot = self.spend(session_id=session_id, spend_date=spend_date)

        if snapshot.session_usd + estimated > self._limits.session_cost_cap_usd:
            raise BudgetExceededError(
                f"session {session_id} has spent ${snapshot.session_usd:.4f} and this call "
                f"would add ${estimated:.4f}, over the ${self._limits.session_cost_cap_usd:.2f} "
                "session cap",
                detail={
                    "code": "session_cost_exceeded",
                    "spent_usd": snapshot.session_usd,
                    "estimated_usd": estimated,
                    "limit_usd": self._limits.session_cost_cap_usd,
                },
            )

        if snapshot.daily_usd + estimated > self._limits.daily_cost_cap_usd:
            raise BudgetExceededError(
                f"${snapshot.daily_usd:.4f} has been spent on {spend_date.isoformat()} and "
                f"this call would add ${estimated:.4f}, over the "
                f"${self._limits.daily_cost_cap_usd:.2f} daily cap",
                detail={
                    "code": "daily_cost_exceeded",
                    "spent_usd": snapshot.daily_usd,
                    "estimated_usd": estimated,
                    "limit_usd": self._limits.daily_cost_cap_usd,
                },
            )
        return estimated

    def record(
        self,
        *,
        session_id: str,
        request_id: str,
        spend_date: date,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> CostLedgerEntry:
        """Record actual usage after a call.

        Uses the usage the API reported, not the pre-call estimate, so the ledger
        reflects money actually spent. ``request_id`` is unique in the schema, so
        a retried write cannot double-count.
        """
        entry = CostLedgerEntry(
            session_id=session_id,
            request_id=request_id,
            spend_date=spend_date,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=self._pricing.cost(prompt_tokens, completion_tokens),
        )
        self._session.add(entry)
        self._session.flush()
        return entry
