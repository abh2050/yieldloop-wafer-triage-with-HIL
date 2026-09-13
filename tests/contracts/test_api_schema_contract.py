"""The API's published schema is a contract with the console.

The frontend's TypeScript types in ``frontend/src/api/types.ts`` are written by
hand against these models. These tests assert the shapes the console depends on
are actually present, so a backend rename fails here rather than surfacing as
``undefined`` in a reviewer's browser.

The most important assertion is that response models which may withhold a
prediction declare those fields as nullable. A field the API sometimes omits but
declares as required is a contract that lies.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from yieldloop.api.main import create_app

pytestmark = pytest.mark.postgres


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    with TestClient(create_app()) as client:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        return dict(response.json())


def _model(schema: dict[str, Any], name: str) -> dict[str, Any]:
    components = schema["components"]["schemas"]
    assert name in components, f"{name} is missing from the published schema"
    return dict(components[name])


EXPECTED_PATHS = [
    ("/health/live", "get"),
    ("/health/ready", "get"),
    ("/health/model", "get"),
    ("/label/queue", "get"),
    ("/label/reason-codes", "get"),
    ("/label/decision", "post"),
    ("/triage/queue", "get"),
    ("/triage/wafer/{wafer_id}", "get"),
    ("/hypothesis", "post"),
    ("/audit", "get"),
    ("/audit/guardrails", "get"),
    ("/audit/verify", "get"),
]


@pytest.mark.parametrize(("path", "method"), EXPECTED_PATHS)
def test_expected_endpoints_are_published(schema: dict[str, Any], path: str, method: str) -> None:
    assert path in schema["paths"], f"{path} is not published"
    assert method in schema["paths"][path], f"{method.upper()} {path} is not published"


def test_queue_item_declares_withheld_prediction_fields_as_nullable(
    schema: dict[str, Any],
) -> None:
    """The console relies on these being absent below the confidence floor."""
    model = _model(schema, "QueueItemResponse")
    properties = model["properties"]
    for field in ("predicted_label", "confidence"):
        assert field in properties
        assert field not in model.get("required", []), (
            f"{field} is declared required but the API omits it below the floor"
        )
    assert "show_prediction" in model["required"]


def test_wafer_detail_declares_the_same_contract(schema: dict[str, Any]) -> None:
    model = _model(schema, "WaferDetailResponse")
    assert "show_prediction" in model["required"]
    for field in ("predicted_label", "confidence", "probabilities"):
        assert field in model["properties"]


def test_hypothesis_envelope_treats_abstention_as_first_class(
    schema: dict[str, Any],
) -> None:
    """Abstention is an outcome, not an error, so it must be in the 201 body."""
    model = _model(schema, "HypothesisEnvelope")
    required = model["required"]
    for field in ("abstained", "hypotheses", "citations", "degraded", "injection_suspected"):
        assert field in required, f"{field} must always be present"


def test_hypothesis_response_requires_citations(schema: dict[str, Any]) -> None:
    """A claim without evidence ids is invalid output by definition."""
    model = _model(schema, "HypothesisResponseModel")
    assert "evidence_ids" in model["required"]
    assert "rank" in model["required"]


def test_decision_payload_requires_a_measured_duration(schema: dict[str, Any]) -> None:
    """The throughput claim is measured from this column, not estimated."""
    model = _model(schema, "DecisionPayload")
    assert "decision_ms" in model["required"]
    assert model["properties"]["decision_ms"].get("minimum") == 0


def test_defect_patterns_match_the_dataset(schema: dict[str, Any]) -> None:
    """Exactly the nine real WM811K labels, no more."""
    from yieldloop.db.enums import DefectPattern

    published = set(_model(schema, "DefectPattern")["enum"])
    assert published == {pattern.value for pattern in DefectPattern}
    assert len(published) == 9


def test_cause_categories_match_the_agent_schema(schema: dict[str, Any]) -> None:
    from yieldloop.db.enums import CauseCategory

    assert set(_model(schema, "CauseCategory")["enum"]) == {
        category.value for category in CauseCategory
    }


def test_routing_bands_are_published(schema: dict[str, Any]) -> None:
    from yieldloop.db.enums import RoutingBand

    assert set(_model(schema, "RoutingBand")["enum"]) == {band.value for band in RoutingBand}


def test_error_body_exposes_a_stable_reason_code(schema: dict[str, Any]) -> None:
    """The console branches on the code; matching message text would be brittle."""
    with TestClient(create_app()) as client:
        response = client.post(
            "/label/decision",
            json={
                "task_id": "00000000-0000-4000-8000-000000000000",
                "action": "accept",
                "decision_ms": 10,
            },
        )
    assert response.status_code == 422
    body = response.json()
    assert body["reason"] == "input_rejected"
    assert "message" in body
    assert "detail" in body


def test_every_request_carries_a_correlation_id() -> None:
    """The same id appears on every audit record the request produces."""
    with TestClient(create_app()) as client:
        response = client.get("/health/live")
    assert response.headers.get("X-Request-Id")


def test_a_supplied_correlation_id_is_preserved() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health/live", headers={"X-Request-Id": "trace-me"})
    assert response.headers["X-Request-Id"] == "trace-me"
