"""Strict parsing of agent output. Fails closed.

There is no lenient path. Output that does not parse is rejected outright rather
than partially recovered, because a half-parsed root cause claim is worse than
none: it reaches a process engineer looking exactly as authoritative as a
complete one, with no indication that a field was dropped or defaulted.

Markdown fences are not stripped either. The agent is called with a structured
output schema, so a fenced response means the model did not honour the response
format -- which is a contract failure worth surfacing, and worth counting toward
the circuit breaker, not quietly papering over.
"""

from __future__ import annotations

import json
from typing import Final

from pydantic import ValidationError

from yieldloop.agent.schemas import HypothesisResponse
from yieldloop.guardrails import SchemaViolationError

#: Refuse to even attempt a parse beyond this size. A response far larger than
#: the schema permits is either a malfunction or an attempt to exhaust the
#: parser, and neither deserves the CPU.
MAX_RESPONSE_BYTES: Final[int] = 256 * 1024


def parse_response(raw: str, *, expected_lot_id: str) -> HypothesisResponse:
    """Parse and validate a raw model response.

    Args:
        raw: The model's response text, expected to be a single JSON object.
        expected_lot_id: The lot this request was made for. Checked against the
            response, so a response that has drifted onto a different lot cannot
            be rendered against this one.

    Raises:
        SchemaViolationError: on any parse or validation failure. The exception
            carries a machine-readable ``stage`` in its detail so the circuit
            breaker and the audit log can distinguish a transport problem from a
            schema problem.
    """
    if not raw or not raw.strip():
        raise SchemaViolationError(
            "agent returned an empty response", detail={"stage": "empty"}
        )

    encoded_length = len(raw.encode("utf-8"))
    if encoded_length > MAX_RESPONSE_BYTES:
        raise SchemaViolationError(
            f"agent response is {encoded_length} bytes, over the {MAX_RESPONSE_BYTES} limit",
            detail={"stage": "oversize", "bytes": encoded_length},
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SchemaViolationError(
            f"agent response is not valid JSON: {exc.msg} at position {exc.pos}",
            detail={"stage": "json_decode", "position": exc.pos},
        ) from exc

    if not isinstance(payload, dict):
        raise SchemaViolationError(
            f"agent response must be a JSON object; got {type(payload).__name__}",
            detail={"stage": "not_an_object"},
        )

    try:
        response = HypothesisResponse.model_validate(payload)
    except ValidationError as exc:
        raise SchemaViolationError(
            f"agent response failed schema validation: {exc.error_count()} error(s)",
            detail={
                "stage": "schema",
                "errors": [
                    {"loc": ".".join(str(p) for p in e["loc"]), "msg": e["msg"]}
                    for e in exc.errors()[:10]
                ],
            },
        ) from exc

    if response.lot_id != expected_lot_id:
        raise SchemaViolationError(
            f"agent response is for lot {response.lot_id!r} but the request was for "
            f"{expected_lot_id!r}",
            detail={
                "stage": "lot_mismatch",
                "returned": response.lot_id,
                "expected": expected_lot_id,
            },
        )

    return response


def response_json_schema() -> dict[str, object]:
    """The JSON schema handed to the model as its structured output format.

    Derived from the Pydantic model rather than maintained separately, so the
    schema the model is held to and the schema the parser enforces cannot drift
    apart.
    """
    return HypothesisResponse.model_json_schema()
