"""Translating domain errors into HTTP responses.

Guardrail errors already carry a stable ``reason`` code and the status they
should become, so this layer maps rather than decides. Putting the status on the
exception means a new guardrail cannot accidentally surface as a 500, and the
frontend can branch on the code instead of parsing prose.

Error bodies never include the exception's message verbatim for unexpected
failures. A stack-derived string can carry a connection URL or a fragment of a
prompt, and this API is read by a browser.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from yieldloop.guardrails import GuardrailError
from yieldloop.logging import get_logger

logger = get_logger(__name__)


class ErrorBody(BaseModel):
    """The single error shape every failing endpoint returns."""

    reason: str = Field(description="Stable machine-readable code")
    message: str = Field(description="Human-readable explanation, safe to display")
    detail: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


def error_response(
    status_code: int,
    reason: str,
    message: str,
    *,
    detail: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorBody(
            reason=reason, message=message, detail=detail or {}, request_id=request_id
        ).model_dump(),
    )


async def guardrail_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Render a guardrail rejection at the status the guardrail chose."""
    assert isinstance(exc, GuardrailError)
    request_id = getattr(request.state, "request_id", None)
    logger.info(
        "guardrail_rejected",
        reason=exc.reason,
        status=exc.status_code,
        path=request.url.path,
        request_id=request_id,
    )
    return error_response(
        exc.status_code, exc.reason, exc.message, detail=exc.detail, request_id=request_id
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort. Logs the detail, returns none of it."""
    request_id = getattr(request.state, "request_id", None)
    logger.exception(
        "unhandled_error",
        path=request.url.path,
        error_type=type(exc).__name__,
        request_id=request_id,
    )
    return error_response(
        500,
        "internal_error",
        "The request could not be completed. The failure has been logged.",
        request_id=request_id,
    )
