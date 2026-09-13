"""Property tests for temperature scaling.

The defining property is that calibration cannot change a prediction. One scalar
divides every logit, so the argmax is invariant and accuracy is unchanged by
construction. If that ever broke, "calibration improved ECE without costing
accuracy" would stop being a guarantee and become a claim needing its own proof.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import torch
from hypothesis import HealthCheck, given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from yieldloop.models.calibrate import (
    MAX_TEMPERATURE,
    MIN_TEMPERATURE,
    CalibrationError,
    apply_temperature,
    calibrated_probabilities,
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
)

CLASSES = 9

# Realistic logit magnitudes. Values between zero and float32's smallest normal
# (~1.2e-38) are excluded deliberately: dividing a denormal by the temperature
# underflows it to exactly zero, which can flip an argmax between two entries
# that are both already indistinguishable from zero. That is a property of
# float32 underflow rather than of temperature scaling, and no trained network
# produces logits in that regime -- Hypothesis found it at 1.4e-45.
finite = st.one_of(
    st.just(0.0),
    st.floats(min_value=1e-6, max_value=20.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-20.0, max_value=-1e-6, allow_nan=False, allow_infinity=False),
)
temperatures = st.floats(min_value=0.05, max_value=10.0, allow_nan=False, allow_infinity=False)


def _logits(array: npt.NDArray[np.float64]) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32))


@given(
    array=arrays(np.float64, (40, CLASSES), elements=finite),
    temperature=temperatures,
)
def test_temperature_never_changes_the_predicted_class(
    array: npt.NDArray[np.float64], temperature: float
) -> None:
    """The property everything else rests on."""
    logits = _logits(array)
    before = logits.argmax(dim=1)
    after = apply_temperature(logits, temperature).argmax(dim=1)
    assert torch.equal(before, after)


@given(
    array=arrays(np.float64, (40, CLASSES), elements=finite),
    temperature=temperatures,
)
def test_temperature_preserves_rank_order(
    array: npt.NDArray[np.float64], temperature: float
) -> None:
    """Scaling is monotonic, so the full ranking is preserved, not just the top."""
    logits = _logits(array)
    before = logits.argsort(dim=1)
    after = apply_temperature(logits, temperature).argsort(dim=1)
    assert torch.equal(before, after)


@given(array=arrays(np.float64, (30, CLASSES), elements=finite), temperature=temperatures)
def test_calibrated_probabilities_are_a_distribution(
    array: npt.NDArray[np.float64], temperature: float
) -> None:
    probabilities = calibrated_probabilities(_logits(array), temperature)
    assert torch.all(probabilities >= 0.0)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(probabilities.shape[0]), atol=1e-5)


@given(array=arrays(np.float64, (30, CLASSES), elements=finite))
def test_higher_temperature_never_increases_confidence(
    array: npt.NDArray[np.float64],
) -> None:
    """Raising the temperature softens the distribution, always.

    This is the direction of the fix: an overconfident model gets a temperature
    above one, which lowers its stated confidence toward its real accuracy.
    """
    logits = _logits(array)
    cooler = calibrated_probabilities(logits, 1.0).max(dim=1).values
    warmer = calibrated_probabilities(logits, 2.0).max(dim=1).values
    assert torch.all(warmer <= cooler + 1e-6)


@hyp_settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_fitting_never_worsens_negative_log_likelihood(seed: int) -> None:
    """Temperature 1.0 is always in the search space, so the fit cannot lose."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, CLASSES, (300,), generator=generator)
    logits = torch.randn(300, CLASSES, generator=generator) * 2.0
    logits[torch.arange(300), labels] += 3.0

    result = fit_temperature(logits, labels)
    assert result.nll_after <= result.nll_before + 1e-4
    assert MIN_TEMPERATURE <= result.temperature <= MAX_TEMPERATURE


@hyp_settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_fitting_leaves_accuracy_untouched(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, CLASSES, (300,), generator=generator)
    logits = torch.randn(300, CLASSES, generator=generator) * 2.0
    logits[torch.arange(300), labels] += 3.0

    before = float(logits.argmax(dim=1).eq(labels).float().mean())
    result = fit_temperature(logits, labels)
    after = float(
        apply_temperature(logits, result.temperature).argmax(dim=1).eq(labels).float().mean()
    )
    assert before == after


def test_overconfident_model_receives_a_temperature_above_one() -> None:
    """The sign of the correction must be right, not just its magnitude."""
    generator = torch.Generator().manual_seed(7)
    labels = torch.randint(0, CLASSES, (2000,), generator=generator)
    logits = torch.randn(2000, CLASSES, generator=generator) * 0.5
    logits[torch.arange(2000), labels] += 6.0
    # Make a quarter of them actually wrong, so the confidence is unjustified.
    wrong = torch.rand(2000, generator=generator) < 0.25
    logits[wrong] = logits[wrong].roll(1, dims=1)

    result = fit_temperature(logits, labels)
    assert result.temperature > 1.0
    assert result.ece_after < result.ece_before


def test_perfectly_calibrated_predictions_have_zero_error() -> None:
    probabilities = torch.zeros(200, CLASSES)
    probabilities[:, 0] = 1.0
    labels = torch.zeros(200, dtype=torch.long)
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.0)


@pytest.mark.parametrize("temperature", [0.0, -1.0, -0.5])
def test_non_positive_temperature_is_rejected(temperature: float) -> None:
    with pytest.raises(CalibrationError, match="positive"):
        apply_temperature(torch.randn(4, CLASSES), temperature)


def test_empty_validation_set_is_rejected() -> None:
    """Silently returning temperature 1.0 would look like a successful fit."""
    with pytest.raises(CalibrationError, match="empty"):
        fit_temperature(torch.zeros(0, CLASSES), torch.zeros(0, dtype=torch.long))


def test_reliability_curve_bins_every_sample_exactly_once() -> None:
    generator = torch.Generator().manual_seed(3)
    probabilities = torch.softmax(torch.randn(500, CLASSES, generator=generator), dim=1)
    labels = torch.randint(0, CLASSES, (500,), generator=generator)
    curve = reliability_curve(probabilities, labels)
    assert sum(int(b["count"]) for b in curve) == 500
