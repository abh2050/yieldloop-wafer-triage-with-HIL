"""The FastAPI application.

Composition only. Every handler lives in :mod:`yieldloop.api.routes` and every
behaviour lives in a service module, so this file says what the API *is* without
saying what anything does.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from yieldloop.api.errors import guardrail_error_handler, unhandled_error_handler
from yieldloop.api.routes import audit, health, hypothesis, label, triage
from yieldloop.config import get_settings
from yieldloop.guardrails import GuardrailError
from yieldloop.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings)
    logger.info(
        "api_start",
        env=settings.env.value,
        confidence_floor=settings.confidence_floor,
        auto_commit_threshold=settings.auto_commit_threshold,
    )
    yield
    logger.info("api_stop")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="yieldloop",
        version="0.1.0",
        summary=(
            "Wafer map defect triage with a human decision loop, active learning, "
            "and a grounded root cause agent."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Reviewer-Id"],
    )

    @app.middleware("http")
    async def correlate(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Give every request an id and bind it to the logging context.

        The same id goes on every audit record the request produces, so a line in
        the audit trail can be traced back to the call that caused it.
        """
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        request.state.request_id = request_id
        bind_request_context(request_id=request_id, path=request.url.path)
        try:
            response = await call_next(request)
        finally:
            clear_request_context()
        response.headers["X-Request-Id"] = request_id
        return response

    app.add_exception_handler(GuardrailError, guardrail_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

    app.include_router(health.router)
    app.include_router(label.router)
    app.include_router(triage.router)
    app.include_router(hypothesis.router)
    app.include_router(audit.router)
    return app


app = create_app()
