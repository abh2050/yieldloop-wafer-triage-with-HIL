"""Property tests for the drift measures.

The one that matters most asserts the two calibration-error implementations
agree. `telemetry.drift` deliberately reimplements ECE in numpy rather than
importing `eval`, because the evaluation package depends on the runtime and not
the reverse. Duplicated logic drifts apart silently, so the duplication is only
acceptable with a test pinning them together.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import torch
from eval.metrics.calibration import expected_calibration_error as eval_ece
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from yieldloop.telemetry.drift import (
    PSI_MAJOR,
    PSI_MINOR,
    DriftError,
    binned_calibration_error,
    calibration_drift,
    embedding_psi,
    population_stability_index,
)

probabilities = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=64
)
values = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)


# --- the two ECE implementations must agree ------------------------------


@given(
    confidence=arrays(np.float64, 200, elements=probabilities),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_runtime_and_eval_calibration_error_agree(
    confidence: npt.NDArray[np.float64], seed: int
) -> None:
    rng = np.random.default_rng(seed)
    correct = rng.random(confidence.shape[0]) < confidence
    assert binned_calibration_error(confidence, correct) == pytest.approx(
        eval_ece(confidence, correct), abs=1e-12
    )


def test_both_agree_with_the_torch_implementation() -> None:
    """And with the third, in models.calibrate, which works on tensors."""
    from yieldloop.models.calibrate import expected_calibration_error as torch_ece

    rng = np.random.default_rng(11)
    probability_matrix = rng.dirichlet(np.ones(9), size=500)
    labels = rng.integers(0, 9, 500)

    confidence = probability_matrix.max(axis=1)
    correct = probability_matrix.argmax(axis=1) == labels

    # 1e-7 rather than exact: the torch path accumulates in float32 internally,
    # so the two agree to about nine significant figures. Any real divergence
    # between the implementations would be orders of magnitude larger.
    assert binned_calibration_error(confidence, correct) == pytest.approx(
        torch_ece(torch.from_numpy(probability_matrix), torch.from_numpy(labels)), abs=1e-7
    )


# --- PSI -----------------------------------------------------------------


@given(sample=arrays(np.float64, 500, elements=values))
def test_identical_distributions_have_no_drift(sample: npt.NDArray[np.float64]) -> None:
    assert population_stability_index(sample, sample).value == pytest.approx(0.0, abs=1e-6)


@given(
    sample=arrays(np.float64, 400, elements=values),
    shift=st.floats(min_value=5.0, max_value=40.0),
)
def test_a_shifted_distribution_registers_drift(
    sample: npt.NDArray[np.float64], shift: float
) -> None:
    """A shift larger than the spread must be detected, given enough resolution.

    Scoped to references that fill all ten requested bins. Quantile-binned PSI's
    sensitivity is a function of its bin resolution: a sample of 399 identical
    values and one outlier yields a single bin where no shift is detectable at
    all, and two bins detect only shifts that cross the one boundary. Reporting a
    small number there is correct behaviour, so it is the premise that fails
    rather than the measure -- which is why `measurable` exists and why the
    monotonicity test below uses a well-formed reference.
    """
    result = population_stability_index(sample, sample + shift, bins=10)
    if result.bins < 10:
        return
    assert result.value > PSI_MINOR


@given(sample=arrays(np.float64, 300, elements=values))
def test_psi_is_non_negative(sample: npt.NDArray[np.float64]) -> None:
    rng = np.random.default_rng(5)
    assert population_stability_index(sample, sample + rng.normal(0, 1, 300)).value >= 0.0


def test_larger_shifts_produce_larger_psi() -> None:
    """Monotonicity: PSI must order shifts the way an operator expects."""
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 1, 2000)
    values_by_shift = [
        population_stability_index(reference, reference + shift).value
        for shift in (0.1, 0.5, 1.0, 2.0)
    ]
    assert values_by_shift == sorted(values_by_shift)


def test_severity_bands_are_reported() -> None:
    rng = np.random.default_rng(7)
    reference = rng.normal(0, 1, 2000)
    assert population_stability_index(reference, reference).severity == "stable"
    assert population_stability_index(reference, reference + 3.0).severity == "major"
    assert population_stability_index(reference, reference + 3.0).actionable


def test_empty_populations_report_no_drift_rather_than_failing() -> None:
    """A cold start has nothing to compare, which is not the same as drift."""
    assert population_stability_index(np.zeros(0), np.ones(10)).value == 0.0
    assert population_stability_index(np.ones(10), np.zeros(0)).value == 0.0


def test_constant_reference_reports_unmeasurable_rather_than_stable() -> None:
    """No spread means no quantile edges.

    The distinction matters: a zero from an unmeasurable comparison is not
    evidence of stability, and an operator acting on it would be reassured by
    nothing.
    """
    result = population_stability_index(np.full(100, 3.0), np.linspace(0, 1, 100))
    assert result.value == 0.0
    assert not result.measurable
    assert not result.actionable


def test_a_real_comparison_is_measurable() -> None:
    rng = np.random.default_rng(23)
    result = population_stability_index(rng.normal(0, 1, 500), rng.normal(0, 1, 500))
    assert result.measurable


@pytest.mark.parametrize("bins", [0, 1, -5])
def test_degenerate_bin_counts_are_rejected(bins: int) -> None:
    with pytest.raises(DriftError, match="bins"):
        population_stability_index(np.zeros(10), np.ones(10), bins=bins)


def test_non_finite_inputs_are_rejected() -> None:
    with pytest.raises(DriftError, match="NaN"):
        population_stability_index(np.array([np.nan, 1.0]), np.ones(2))


# --- embedding PSI -------------------------------------------------------


def test_embedding_psi_detects_a_shift_in_a_few_dimensions() -> None:
    """Per-dimension, because a projection would average away exactly the shape
    of drift that matters: one defect signature becoming more common."""
    rng = np.random.default_rng(13)
    reference = rng.normal(0, 1, (800, 16))
    current = reference.copy()
    current[:, :2] += 3.0
    assert embedding_psi(reference, current).value > PSI_MINOR
    assert embedding_psi(reference, reference).value == pytest.approx(0.0, abs=1e-6)


def test_embedding_dimension_mismatch_is_rejected() -> None:
    with pytest.raises(DriftError, match="dimension mismatch"):
        embedding_psi(np.zeros((5, 8)), np.zeros((5, 16)))


def test_embedding_psi_requires_matrices() -> None:
    with pytest.raises(DriftError, match="2-D"):
        embedding_psi(np.zeros(8), np.zeros(8))


# --- calibration drift ---------------------------------------------------


def test_calibration_degradation_is_relative_not_absolute() -> None:
    """A model that started at 0.01 and drifted to 0.04 has a real problem that
    an absolute threshold tuned for one starting at 0.05 would miss."""
    rng = np.random.default_rng(17)
    confidence = rng.uniform(0.5, 1.0, 500)
    # Badly calibrated: confident but right only half the time.
    correct = rng.random(500) < 0.5

    assert calibration_drift(0.01, confidence, correct).degraded
    assert not calibration_drift(0.60, confidence, correct).degraded


def test_calibration_drift_reports_the_delta() -> None:
    rng = np.random.default_rng(19)
    confidence = rng.uniform(0.0, 1.0, 400)
    correct = rng.random(400) < confidence
    drift = calibration_drift(0.05, confidence, correct)
    assert drift.delta == pytest.approx(drift.current_ece - 0.05)


def test_an_empty_window_is_never_reported_as_degraded() -> None:
    """No decisions is not evidence of anything."""
    drift = calibration_drift(0.05, np.zeros(0), np.zeros(0, dtype=bool))
    assert drift.samples == 0
    assert not drift.degraded


def test_calibration_error_shape_mismatch_is_rejected() -> None:
    with pytest.raises(DriftError, match="same shape"):
        binned_calibration_error(np.zeros(5), np.zeros(3, dtype=bool))


def test_psi_thresholds_are_ordered() -> None:
    assert 0.0 < PSI_MINOR < PSI_MAJOR


# --- agreement: unmeasurable is not the same as zero ----------------------


def test_anchoring_delta_is_none_when_one_regime_is_empty() -> None:
    """Reporting 0.0 would read as "no anchoring effect" when it means "not
    measured", and those call for opposite responses."""
    from eval.metrics.agreement import DecisionObservation
    from eval.metrics.agreement import summarize as summarize_agreement

    blind_only = [
        DecisionObservation("w1", "none", "loc", False, True, 1000, "wrong_class")
        for _ in range(20)
    ]
    report = summarize_agreement(blind_only)
    assert report.blind_decisions == 20
    assert report.shown_decisions == 0
    assert not report.anchoring_measurable
    assert report.anchoring_delta is None


def test_anchoring_delta_is_reported_when_both_regimes_exist() -> None:
    from eval.metrics.agreement import DecisionObservation
    from eval.metrics.agreement import summarize as summarize_agreement

    observations = [
        *[DecisionObservation("w", "none", "none", True, False, 900, None) for _ in range(10)],
        *[
            DecisionObservation("w", "none", "loc", False, True, 900, "wrong_class")
            for _ in range(10)
        ],
    ]
    report = summarize_agreement(observations)
    assert report.anchoring_measurable
    assert report.anchoring_delta == pytest.approx(1.0)
