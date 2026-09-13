"""The guardrail layer.

Every call into the hypothesis agent passes through here, on both sides. This is
not optional and there is no caller flag that turns it off: the agent client is
private to :mod:`yieldloop.agent` and the only exported entry point is the
guarded one.

The layer fails closed. When a guardrail cannot establish that an output is safe
to show, the system returns an explicit abstention rather than a degraded answer,
because a root cause hypothesis a process engineer cannot trust is worse than no
hypothesis at all.

This module holds only the exception hierarchy, so that submodules can import
from the package without a cycle.
"""

from __future__ import annotations


class GuardrailError(Exception):
    """Base for every guardrail rejection.

    Carries a stable ``reason`` code rather than relying on the message text, so
    the API can translate failures and the audit log can aggregate them without
    parsing prose.
    """

    reason: str = "guardrail_error"
    #: HTTP status the API layer should translate this to.
    status_code: int = 400

    def __init__(self, message: str, *, detail: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class InputRejectedError(GuardrailError):
    """An input failed validation before any tokens were spent."""

    reason = "input_rejected"
    status_code = 422


class BudgetExceededError(GuardrailError):
    """A token, session, or daily spend ceiling would be breached."""

    reason = "budget_exceeded"
    status_code = 429


class SchemaViolationError(GuardrailError):
    """Model output did not conform to the response schema."""

    reason = "schema_violation"
    status_code = 502


class CircuitOpenError(GuardrailError):
    """The breaker is open; the console degrades to classifier-only mode."""

    reason = "circuit_open"
    status_code = 503
