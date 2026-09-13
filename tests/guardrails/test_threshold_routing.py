"""Routing bands, including the anchoring rule.

The band boundaries themselves are simple arithmetic. The property worth testing
is the one that is easy to regress: below the confidence floor the reviewer must
not be shown the prediction. A change that starts showing it would look harmless,
pass every accuracy test, and quietly convert model errors into human labels.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from yieldloop.config import Settings
from yieldloop.db.enums import RoutingBand
from yieldloop.guardrails.thresholds import RoutingBands, automation_rate, classify

BANDS = RoutingBands(confidence_floor=0.55, auto_commit_threshold=0.95)

probabilities = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=64
)


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (1.00, RoutingBand.AUTO_COMMIT),
        (0.96, RoutingBand.AUTO_COMMIT),
        (0.95, RoutingBand.AUTO_COMMIT),
        (0.9499, RoutingBand.UNCERTAINTY_BAND),
        (0.70, RoutingBand.UNCERTAINTY_BAND),
        (0.55, RoutingBand.UNCERTAINTY_BAND),
        (0.5499, RoutingBand.BELOW_FLOOR),
        (0.10, RoutingBand.BELOW_FLOOR),
        (0.00, RoutingBand.BELOW_FLOOR),
    ],
)
def test_band_boundaries_are_inclusive_from_below(confidence: float, expected: RoutingBand) -> None:
    assert classify(confidence, BANDS).band is expected


@given(confidence=probabilities)
def test_prediction_is_hidden_below_the_floor(confidence: float) -> None:
    """The anchoring rule. This is the assertion that must never be relaxed."""
    decision = classify(confidence, BANDS)
    if confidence < BANDS.confidence_floor:
        assert not decision.show_prediction
    else:
        assert decision.show_prediction


@given(confidence=probabilities)
def test_only_the_auto_commit_band_skips_the_human(confidence: float) -> None:
    decision = classify(confidence, BANDS)
    assert decision.requires_human == (confidence < BANDS.auto_commit_threshold)
    assert decision.is_auto_committed == (not decision.requires_human)


@given(confidence=probabilities)
def test_every_confidence_lands_in_exactly_one_band(confidence: float) -> None:
    assert classify(confidence, BANDS).band in set(RoutingBand)


@pytest.mark.parametrize("confidence", [-0.01, 1.01, 2.0, -5.0])
def test_uncalibrated_scores_are_rejected(confidence: float) -> None:
    """A caller passing a logit would otherwise auto-commit everything."""
    with pytest.raises(ValueError, match="calibrated probability"):
        classify(confidence, BANDS)


@pytest.mark.parametrize(
    ("floor", "auto_commit"),
    [(0.95, 0.95), (0.96, 0.95), (0.9, 0.5), (0.0, 0.9), (0.5, 1.5), (-0.1, 0.9)],
)
def test_incoherent_bands_are_rejected(floor: float, auto_commit: float) -> None:
    with pytest.raises(ValueError):
        RoutingBands(confidence_floor=floor, auto_commit_threshold=auto_commit)


def test_bands_come_from_settings_not_constants() -> None:
    """The eval harness sweeps these, so they must be configuration."""
    settings = Settings(confidence_floor=0.4, auto_commit_threshold=0.8)
    bands = RoutingBands.from_settings(settings)
    assert bands.confidence_floor == 0.4
    assert bands.auto_commit_threshold == 0.8
    assert classify(0.5, bands).band is RoutingBand.UNCERTAINTY_BAND
    assert classify(0.3, bands).band is RoutingBand.BELOW_FLOOR


def test_automation_rate_is_the_tradeoff_curve_y_axis() -> None:
    confidences = [0.99, 0.97, 0.96, 0.5, 0.2]
    assert automation_rate(confidences, BANDS) == pytest.approx(0.6)
    assert automation_rate([], BANDS) == 0.0


@given(
    floor=st.floats(min_value=0.05, max_value=0.5),
    gap=st.floats(min_value=0.05, max_value=0.45),
    confidences=st.lists(probabilities, min_size=1, max_size=50),
)
def test_raising_the_auto_commit_threshold_never_increases_automation(
    floor: float, gap: float, confidences: list[float]
) -> None:
    """Monotonicity of the tradeoff curve.

    Sweeping the threshold upward must move work toward humans, never away from
    them. A violation would make the routing curve meaningless.
    """
    lower = RoutingBands(confidence_floor=floor, auto_commit_threshold=floor + gap)
    higher = RoutingBands(
        confidence_floor=floor, auto_commit_threshold=min(1.0, floor + gap + 0.05)
    )
    assert automation_rate(confidences, higher) <= automation_rate(confidences, lower)


@given(
    floor=st.floats(min_value=0.05, max_value=0.8),
    confidences=st.lists(probabilities, min_size=1, max_size=50),
)
def test_raising_the_floor_never_shows_more_predictions(
    floor: float, confidences: list[float]
) -> None:
    """Raising the floor must withhold more predictions, never fewer."""
    lower = RoutingBands(confidence_floor=floor, auto_commit_threshold=0.95)
    raised = RoutingBands(confidence_floor=min(0.94, floor + 0.05), auto_commit_threshold=0.95)
    shown_lower = sum(1 for c in confidences if classify(c, lower).show_prediction)
    shown_raised = sum(1 for c in confidences if classify(c, raised).show_prediction)
    assert shown_raised <= shown_lower
