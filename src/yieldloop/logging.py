"""Structured logging.

All logs are structured events. Audit records are *not* written here -- they go
to the append-only audit table via :mod:`yieldloop.guardrails.audit`. This module
is for operational telemetry only, and it redacts free text by default so that
reviewer notes and retrieved report text never leak into log aggregation.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any, Final

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

from yieldloop.config import LogFormat, Settings, get_settings

REDACTED: Final[str] = "[redacted]"

#: Event keys whose values are reviewer- or model-authored free text. They are
#: replaced with :data:`REDACTED` before a record reaches any sink.
SENSITIVE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "note",
        "notes",
        "free_text",
        "reviewer_note",
        "prompt",
        "completion",
        "raw_response",
        "report_text",
        "statement",
        "api_key",
        "openai_api_key",
        "authorization",
    }
)


def redact_sensitive(_logger: WrappedLogger, _method_name: str, event_dict: EventDict) -> EventDict:
    """Replace free-text and secret values with :data:`REDACTED`."""
    for key in list(event_dict):
        if key.lower() in SENSITIVE_KEYS and event_dict[key] is not None:
            event_dict[key] = REDACTED
    return event_dict


def _build_processors(log_format: LogFormat) -> list[Processor]:
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        redact_sensitive,
    ]
    renderer: Processor
    if log_format is LogFormat.JSON:
        shared.append(structlog.processors.format_exc_info)
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    shared.append(renderer)
    return shared


def configure_logging(settings: Settings | None = None) -> None:
    """Configure structlog and the stdlib root logger. Idempotent."""
    resolved = settings if settings is not None else get_settings()
    level = logging.getLevelNamesMapping().get(resolved.log_level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level, force=True)
    structlog.configure(
        processors=_build_processors(resolved.log_format),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for ``name``."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    if initial_values:
        logger = logger.bind(**initial_values)
    return logger


def bind_request_context(**values: Any) -> None:
    """Bind values onto the current context so all downstream events carry them."""
    structlog.contextvars.bind_contextvars(**values)


def clear_request_context() -> None:
    """Clear the per-request logging context."""
    structlog.contextvars.clear_contextvars()


def log_context() -> MutableMapping[str, Any]:
    """Return the currently bound context values."""
    return structlog.contextvars.get_contextvars()
