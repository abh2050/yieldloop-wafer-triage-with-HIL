"""Circuit breaker over the agent path.

Three independent trip conditions, any one of which opens the circuit:

* a run of consecutive schema failures,
* a grounding rejection rate above the configured ratio,
* a p95 latency breach.

Opening the circuit does **not** fail the page. It degrades the console to
classifier-only mode: the wafer map, the calibrated prediction, and the review
queue all keep working, and the hypothesis panel renders as unavailable. A triage
console that returns 503 because a language model is misbehaving is worse than
one that quietly drops the feature the model powers, because the reviewer's
actual job does not depend on it.

The state machine is pure and holds its window in memory. That is the right scope
for latency and failure-streak data, which is per-process by nature; the
persistent record of transitions goes to ``breaker_events`` for the dashboards.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Final

from yieldloop.config import Settings
from yieldloop.db.enums import BreakerState

#: Trip reasons, recorded on the transition.
TRIGGER_SCHEMA_STREAK: Final[str] = "schema_failure_streak"
TRIGGER_GROUNDING_RATE: Final[str] = "grounding_rejection_rate"
TRIGGER_LATENCY: Final[str] = "latency_p95_breach"
TRIGGER_PROBE_FAILED: Final[str] = "half_open_probe_failed"
TRIGGER_PROBE_SUCCEEDED: Final[str] = "half_open_probe_succeeded"
TRIGGER_COOLDOWN_ELAPSED: Final[str] = "cooldown_elapsed"


@dataclass(frozen=True, slots=True)
class BreakerThresholds:
    schema_failure_streak: int
    grounding_reject_ratio: float
    latency_p95_seconds: float
    window_size: int
    cooldown_seconds: float

    @classmethod
    def from_settings(cls, settings: Settings) -> BreakerThresholds:
        return cls(
            schema_failure_streak=settings.breaker_schema_failure_streak,
            grounding_reject_ratio=settings.breaker_grounding_reject_ratio,
            latency_p95_seconds=settings.breaker_latency_p95_seconds,
            window_size=settings.breaker_window_size,
            cooldown_seconds=settings.breaker_cooldown_seconds,
        )


@dataclass(frozen=True, slots=True)
class Transition:
    """A state change, for persistence into ``breaker_events``."""

    from_state: BreakerState
    to_state: BreakerState
    trigger: str
    observed: dict[str, float | int]


@dataclass(frozen=True, slots=True)
class Outcome:
    """One agent call's result, fed to the breaker."""

    schema_ok: bool
    latency_seconds: float
    #: Fraction of hypotheses dropped by the grounding gate, in ``[0, 1]``.
    grounding_rejection_rate: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.grounding_rejection_rate <= 1.0:
            raise ValueError(
                f"grounding_rejection_rate must be in [0, 1]; got {self.grounding_rejection_rate}"
            )
        if self.latency_seconds < 0.0:
            raise ValueError(f"latency_seconds must be non-negative; got {self.latency_seconds}")


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Returns 0.0 for an empty sample."""
    if not values:
        return 0.0
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-len(ordered) * fraction // 1))))
    return ordered[rank - 1]


@dataclass(slots=True)
class CircuitBreaker:
    """Sliding-window breaker over agent call outcomes."""

    thresholds: BreakerThresholds
    state: BreakerState = BreakerState.CLOSED
    _schema_streak: int = field(default=0, init=False)
    _outcomes: deque[Outcome] = field(default_factory=deque, init=False)
    _opened_at: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._outcomes = deque(maxlen=self.thresholds.window_size)

    @property
    def is_open(self) -> bool:
        """True when the agent path must not be called."""
        return self.state is BreakerState.OPEN

    @property
    def allows_calls(self) -> bool:
        """True when a call may proceed, including a half-open probe."""
        return self.state is not BreakerState.OPEN

    @property
    def window(self) -> tuple[Outcome, ...]:
        return tuple(self._outcomes)

    def grounding_rejection_rate(self) -> float:
        """Mean rejection rate across the window."""
        if not self._outcomes:
            return 0.0
        return sum(o.grounding_rejection_rate for o in self._outcomes) / len(self._outcomes)

    def latency_p95(self) -> float:
        return percentile([o.latency_seconds for o in self._outcomes], 0.95)

    def record(self, outcome: Outcome, *, now: float) -> Transition | None:
        """Feed one call outcome. Returns a transition when the state changed."""
        self._outcomes.append(outcome)
        self._schema_streak = 0 if outcome.schema_ok else self._schema_streak + 1

        if self.state is BreakerState.HALF_OPEN:
            return self._resolve_probe(outcome, now=now)
        if self.state is BreakerState.OPEN:
            return None
        return self._check_trip(now=now)

    def _resolve_probe(self, outcome: Outcome, *, now: float) -> Transition | None:
        """A half-open probe either restores service or reopens the circuit."""
        probe_ok = (
            outcome.schema_ok
            and outcome.grounding_rejection_rate < self.thresholds.grounding_reject_ratio
        )
        if probe_ok:
            self.state = BreakerState.CLOSED
            self._opened_at = None
            self._schema_streak = 0
            return Transition(
                from_state=BreakerState.HALF_OPEN,
                to_state=BreakerState.CLOSED,
                trigger=TRIGGER_PROBE_SUCCEEDED,
                observed={"grounding_rejection_rate": outcome.grounding_rejection_rate},
            )
        self.state = BreakerState.OPEN
        self._opened_at = now
        return Transition(
            from_state=BreakerState.HALF_OPEN,
            to_state=BreakerState.OPEN,
            trigger=TRIGGER_PROBE_FAILED,
            observed={
                "schema_ok": int(outcome.schema_ok),
                "grounding_rejection_rate": outcome.grounding_rejection_rate,
            },
        )

    def _check_trip(self, *, now: float) -> Transition | None:
        if self._schema_streak >= self.thresholds.schema_failure_streak:
            return self._open(
                TRIGGER_SCHEMA_STREAK,
                {"streak": self._schema_streak, "limit": self.thresholds.schema_failure_streak},
                now=now,
            )

        # Rate and latency need a full window. Tripping on the first two calls of
        # a cold process would make a restart look like an outage.
        capacity = self._outcomes.maxlen
        if capacity is not None and len(self._outcomes) < capacity:
            return None

        rate = self.grounding_rejection_rate()
        if rate > self.thresholds.grounding_reject_ratio:
            return self._open(
                TRIGGER_GROUNDING_RATE,
                {"rate": rate, "limit": self.thresholds.grounding_reject_ratio},
                now=now,
            )

        p95 = self.latency_p95()
        if p95 > self.thresholds.latency_p95_seconds:
            return self._open(
                TRIGGER_LATENCY,
                {"p95_seconds": p95, "limit": self.thresholds.latency_p95_seconds},
                now=now,
            )
        return None

    def _open(self, trigger: str, observed: dict[str, float | int], *, now: float) -> Transition:
        previous = self.state
        self.state = BreakerState.OPEN
        self._opened_at = now
        return Transition(
            from_state=previous, to_state=BreakerState.OPEN, trigger=trigger, observed=observed
        )

    def poll(self, *, now: float) -> Transition | None:
        """Advance an open circuit to half-open once the cooldown has elapsed."""
        if self.state is not BreakerState.OPEN or self._opened_at is None:
            return None
        elapsed = now - self._opened_at
        if elapsed < self.thresholds.cooldown_seconds:
            return None
        self.state = BreakerState.HALF_OPEN
        self._schema_streak = 0
        self._outcomes.clear()
        return Transition(
            from_state=BreakerState.OPEN,
            to_state=BreakerState.HALF_OPEN,
            trigger=TRIGGER_COOLDOWN_ELAPSED,
            observed={"elapsed_seconds": elapsed},
        )
