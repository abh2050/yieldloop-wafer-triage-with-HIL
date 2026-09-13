"""Calibration reporting must not present sampling noise as a finding.

A real holdout run produced a max calibration error of 0.704 from a bin holding
two wafers, while the bin the auto-commit threshold actually sits in -- holding
89% of the wafers -- was calibrated to within 0.003. Both numbers were true; only
one was informative, and the misleading one was the headline.

These tests pin the fix, because the failure mode is silent: nothing errors, a
plausible number is simply wrong in a way that misdirects the person reading the
report to decide whether automation is safe.
"""

from __future__ import annotations

import numpy as np
import pytest
from eval.metrics.calibration import (
    MIN_BIN_COUNT_FOR_MCE,
    expected_calibration_error,
    high_confidence_calibration_error,
    maximum_calibration_error,
)


def _population(confidence: float, count: int, accuracy: float) -> tuple[np.ndarray, np.ndarray]:
    correct = np.zeros(count, dtype=bool)
    correct[: round(count * accuracy)] = True
    return np.full(count, confidence), correct


def test_a_sparse_bin_does_not_dominate_max_calibration_error() -> None:
    """The exact shape of the real defect: two samples against twenty thousand."""
    sparse_conf, sparse_correct = _population(0.30, 2, 1.0)  # gap of 0.70
    dense_conf, dense_correct = _population(0.99, 20_000, 0.99)  # gap of 0.00

    confidence = np.concatenate([sparse_conf, dense_conf])
    correct = np.concatenate([sparse_correct, dense_correct])

    assert maximum_calibration_error(confidence, correct) < 0.05
    # Without the filter the two-sample bin is the headline number.
    assert maximum_calibration_error(confidence, correct, min_bin_count=1) > 0.5


def test_a_well_populated_bad_bin_is_still_reported() -> None:
    """The filter must not hide real miscalibration, only noise."""
    bad_conf, bad_correct = _population(0.90, 500, 0.40)  # gap of 0.50
    good_conf, good_correct = _population(0.99, 5_000, 0.99)

    confidence = np.concatenate([bad_conf, good_conf])
    correct = np.concatenate([bad_correct, good_correct])

    assert maximum_calibration_error(confidence, correct) > 0.45


@pytest.mark.parametrize("count", [1, 5, 29])
def test_bins_below_the_threshold_are_excluded(count: int) -> None:
    sparse_conf, sparse_correct = _population(0.10, count, 1.0)
    dense_conf, dense_correct = _population(0.99, 1_000, 0.99)
    confidence = np.concatenate([sparse_conf, dense_conf])
    correct = np.concatenate([sparse_correct, dense_correct])
    assert maximum_calibration_error(confidence, correct) < 0.05


def test_a_bin_at_the_threshold_is_included() -> None:
    sparse_conf, sparse_correct = _population(0.10, MIN_BIN_COUNT_FOR_MCE, 1.0)
    dense_conf, dense_correct = _population(0.99, 1_000, 0.99)
    confidence = np.concatenate([sparse_conf, dense_conf])
    correct = np.concatenate([sparse_correct, dense_correct])
    assert maximum_calibration_error(confidence, correct) > 0.8


def test_ece_is_unaffected_by_the_filter() -> None:
    """ECE is population-weighted, so a tiny bin already contributes almost
    nothing. The filter belongs to MCE alone."""
    sparse_conf, sparse_correct = _population(0.30, 2, 1.0)
    dense_conf, dense_correct = _population(0.99, 20_000, 0.99)
    confidence = np.concatenate([sparse_conf, dense_conf])
    correct = np.concatenate([sparse_correct, dense_correct])
    assert expected_calibration_error(confidence, correct) < 0.01


# --- the number that bears on automation safety --------------------------


def test_auto_commit_gap_measures_only_the_governed_band() -> None:
    """Averaging over the whole range hides what the threshold actually governs."""
    low_conf, low_correct = _population(0.60, 5_000, 0.20)  # terrible, but reviewed
    high_conf, high_correct = _population(0.99, 5_000, 0.99)  # what gets committed

    confidence = np.concatenate([low_conf, high_conf])
    correct = np.concatenate([low_correct, high_correct])

    gap, count = high_confidence_calibration_error(confidence, correct, threshold=0.95)
    assert count == 5_000
    assert gap < 0.01
    # The whole-range figure is dominated by predictions a human would have seen.
    assert expected_calibration_error(confidence, correct) > 0.15


def test_auto_commit_gap_reports_a_real_problem_in_the_band() -> None:
    confidence, correct = _population(0.99, 5_000, 0.70)
    gap, count = high_confidence_calibration_error(confidence, correct, threshold=0.95)
    assert count == 5_000
    assert gap == pytest.approx(0.29, abs=0.01)


def test_an_empty_band_reports_no_population_rather_than_zero_gap() -> None:
    """Nothing above the threshold is not the same as perfect calibration."""
    confidence, correct = _population(0.50, 100, 0.5)
    gap, count = high_confidence_calibration_error(confidence, correct, threshold=0.95)
    assert count == 0
    assert gap == 0.0


def test_empty_input_is_handled() -> None:
    empty_conf = np.zeros(0)
    empty_correct = np.zeros(0, dtype=bool)
    assert maximum_calibration_error(empty_conf, empty_correct) == 0.0
    assert high_confidence_calibration_error(empty_conf, empty_correct, threshold=0.95) == (
        0.0,
        0,
    )
