"""Thin transport wrapper over the OpenAI API.

This module knows how to make one call and report what it cost. It deliberately
knows nothing about grounding, budgets, or the circuit breaker: composing those
is :mod:`yieldloop.agent.hypothesis`'s job, and keeping the transport ignorant of
them is what stops a future caller from reaching the model without them.

It is private by convention and by export: :mod:`yieldloop.agent` exposes only
the guarded entry point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Final

from openai import APIError, APITimeoutError, OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from yieldloop.config import Settings
from yieldloop.logging import get_logger

logger = get_logger(__name__)

#: Retries are for transport faults only. A schema failure is never retried here
#: -- it is a contract failure that must reach the circuit breaker, and silently
#: retrying would hide a model that has started returning malformed output.
_RETRYABLE: Final[tuple[type[Exception], ...]] = (APITimeoutError,)


class AgentTransportError(RuntimeError):
    """The model could not be reached, or returned no usable content."""


@dataclass(frozen=True, slots=True)
class Completion:
    """One raw model response, with the usage the API actually reported."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    latency_seconds: float
    #: True when the model stopped because it hit the token ceiling. Such a
    #: response is truncated and will not parse, and the distinction matters to
    #: whoever reads the audit log.
    truncated: bool


class AgentClient:
    """Calls the model with a strict structured-output schema."""

    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        key = settings.openai_api_key.get_secret_value()
        if not key:
            raise AgentTransportError(
                "OPENAI_API_KEY is not set. The hypothesis agent cannot run without it; "
                "the console degrades to classifier-only mode instead."
            )
        self._settings = settings
        self._client = client if client is not None else OpenAI(api_key=key)

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        reraise=True,
    )
    def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        json_schema: dict[str, Any],
        max_completion_tokens: int,
    ) -> Completion:
        """Make one call and return the raw content plus reported usage.

        The response format is a strict JSON schema, so the model is held to the
        same shape the parser enforces. That does not remove the need for the
        parser: structured output constrains the shape, not the truth of the
        content, and grounding is what checks the latter.
        """
        started = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self._settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "HypothesisResponse",
                        "strict": True,
                        "schema": json_schema,
                    },
                },
                max_tokens=max_completion_tokens,
                temperature=0.0,
                timeout=self._settings.agent_timeout_seconds,
            )
        except APIError as exc:
            raise AgentTransportError(f"OpenAI API call failed: {exc}") from exc

        latency = time.monotonic() - started
        choice = response.choices[0]
        content = choice.message.content

        if content is None:
            raise AgentTransportError(
                f"model returned no content (finish_reason={choice.finish_reason})"
            )

        usage = response.usage
        return Completion(
            content=content,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=response.model,
            latency_seconds=latency,
            truncated=choice.finish_reason == "length",
        )


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt a Pydantic JSON schema for OpenAI strict structured outputs.

    Strict mode requires every object to list all of its properties as required
    and to set ``additionalProperties: false``. Pydantic omits fields that have
    defaults from ``required``, so they are added back here; the schema stays the
    single source of truth and is not maintained twice.
    """
    resolved = dict(schema)

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            walked = {key: walk(value) for key, value in node.items()}
            if walked.get("type") == "object" and "properties" in walked:
                walked["additionalProperties"] = False
                walked["required"] = sorted(walked["properties"])
            return walked
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    result: dict[str, Any] = walk(resolved)
    return result
