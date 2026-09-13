"""Drift measures.

Three questions, deliberately separated because they fail for different reasons
and call for different responses.

*Has the input changed?* Population stability index over the embedding
distribution. A rise means the fab is producing wafers unlike the training set,
and the model may be extrapolating.

*Have the probabilities stopped meaning what they say?* Calibration error on
recent decisions, compared against what was measured at training time. A rise
means the routing thresholds are no longer buying what they were tuned to buy.

*Have the humans started disagreeing more?* Override rate, split by whether the
prediction was visible. A rise in the shown rate is the model degrading; a rise
only in the blind rate is something else.

Pure functions. Nothing here reads the database or decides anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

#: Conventional PSI thresholds. Below the first, nothing has moved; above the
#: second, the input distribution has shifted enough to act on.
PSI_MINOR: Final[float] = 0.10
PSI_MAJOR: Final[float] = 0.25

#: Added to empty bins so the log ratio stays finite. A bin that is empty in one
#: population and not the other is real signal, and dropping it would hide
#: exactly the shift PSI exists to detect.
_EPSILON: Final[float] = 1e-6


class DriftError(ValueError):
    """Raised when drift inputs are malformed."""


@dataclass(frozen=True, slots=True)
class PSIResult:
    value: float
    bins: int
    reference_size: int
    current_size: int

    @property
    def severity(self) -> str:
        if self.value < PSI_MINOR:
            return "stable"
        if self.value < PSI_MAJOR:
            return "minor"
        return "major"

    @property
    def actionable(self) -> bool:
        return self.measurable and self.value >= PSI_MAJOR

    @property
    def measurable(self) -> bool:
        """False when the reference had too little spread to form bins.

        A zero from an unmeasurable comparison is not evidence of stability.
        """
        return self.bins >= 2


def population_stability_index(
    reference: npt.ArrayLike, current: npt.ArrayLike, *, bins: int = 10
) -> PSIResult:
    """PSI between a reference and a current distribution of scalars.

    Bin edges come from the **reference** quantiles, not from the combined
    sample. Using the combined sample would let the current distribution move the
    edges and partly mask its own shift.
    """
    reference_array = np.asarray(reference, dtype=np.float64).ravel()
    current_array = np.asarray(current, dtype=np.float64).ravel()

    if bins < 2:
        raise DriftError(f"bins must be at least 2; got {bins}")
    if reference_array.size == 0 or current_array.size == 0:
        return PSIResult(
            value=0.0,
            bins=bins,
            reference_size=int(reference_array.size),
            current_size=int(current_array.size),
        )
    if not np.isfinite(reference_array).all() or not np.isfinite(current_array).all():
        raise DriftError("drift inputs contain NaN or infinity")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference_array, quantiles))
    if edges.size < 2:
        # A reference with no spread has no quantile edges to bin against, so no
        # shift is detectable. Reports the *actual* bin count rather than the
        # requested one, so a caller can tell "measured, no drift" apart from
        # "could not be measured" -- those call for different responses, and
        # reporting the requested count would make them indistinguishable.
        return PSIResult(
            value=0.0,
            bins=max(int(edges.size) - 1, 0),
            reference_size=int(reference_array.size),
            current_size=int(current_array.size),
        )
    edges[0], edges[-1] = -np.inf, np.inf

    reference_counts = np.histogram(reference_array, bins=edges)[0].astype(np.float64)
    current_counts = np.histogram(current_array, bins=edges)[0].astype(np.float64)

    reference_share = reference_counts / reference_counts.sum() + _EPSILON
    current_share = current_counts / current_counts.sum() + _EPSILON

    value = float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )
    return PSIResult(
        value=value,
        bins=int(edges.size - 1),
        reference_size=int(reference_array.size),
        current_size=int(current_array.size),
    )


def embedding_psi(reference: npt.ArrayLike, current: npt.ArrayLike, *, bins: int = 10) -> PSIResult:
    """PSI over an embedding matrix, averaged across dimensions.

    Per-dimension rather than over a single projection: a shift confined to a few
    dimensions is exactly what a projection would average away, and it is the
    shape of drift that matters most -- one defect signature becoming more common
    moves a handful of dimensions, not all of them.
    """
    reference_array = np.asarray(reference, dtype=np.float64)
    current_array = np.asarray(current, dtype=np.float64)
    if reference_array.ndim != 2 or current_array.ndim != 2:
        raise DriftError("embeddings must be 2-D (n, d) arrays")
    if reference_array.size == 0 or current_array.size == 0:
        return PSIResult(0.0, bins, int(reference_array.shape[0]), int(current_array.shape[0]))
    if reference_array.shape[1] != current_array.shape[1]:
        raise DriftError(
            f"dimension mismatch: reference is {reference_array.shape[1]}-D, "
            f"current is {current_array.shape[1]}-D"
        )

    per_dimension = [
        population_stability_index(
            reference_array[:, index], current_array[:, index], bins=bins
        ).value
        for index in range(reference_array.shape[1])
    ]
    return PSIResult(
        value=float(np.mean(per_dimension)),
        bins=bins,
        reference_size=int(reference_array.shape[0]),
        current_size=int(current_array.shape[0]),
    )


@dataclass(frozen=True, slots=True)
class CalibrationDrift:
    """Calibration now against calibration at training time."""

    baseline_ece: float
    current_ece: float
    samples: int

    @property
    def delta(self) -> float:
        return self.current_ece - self.baseline_ece

    @property
    def degraded(self) -> bool:
        """True when calibration error has grown by more than half again.

        A relative test rather than absolute: a model that started at 0.01 and
        drifted to 0.04 has a real problem that an absolute threshold tuned for a
        model starting at 0.05 would miss entirely.
        """
        if self.samples == 0:
            return False
        if self.baseline_ece <= 0.0:
            return self.current_ece > PSI_MINOR
        return self.current_ece > self.baseline_ece * 1.5


def binned_calibration_error(
    confidence: FloatArray, correct: npt.NDArray[np.bool_], bins: int = 15
) -> float:
    """Equal-width binned expected calibration error.

    Implemented here in numpy rather than imported from ``eval``: the evaluation
    package depends on this one, and reaching back the other way would make the
    runtime import the harness. ``eval.metrics.calibration`` computes the same
    measure for offline use, and a test asserts the two agree.
    """
    if confidence.shape != correct.shape:
        raise DriftError("confidence and correct must have the same shape")
    if confidence.size == 0:
        return 0.0
    if bins < 1:
        raise DriftError(f"bins must be positive; got {bins}")

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = (confidence > low) & (confidence <= high) if index else (confidence <= high)
        count = int(mask.sum())
        if count == 0:
            continue
        total += (count / confidence.size) * abs(
            float(correct[mask].mean()) - float(confidence[mask].mean())
        )
    return total


def calibration_drift(
    baseline_ece: float, confidence: FloatArray, correct: npt.NDArray[np.bool_], bins: int = 15
) -> CalibrationDrift:
    """Compare current calibration error against the training-time figure."""
    return CalibrationDrift(
        baseline_ece=baseline_ece,
        current_ece=binned_calibration_error(confidence, correct, bins),
        samples=int(confidence.size),
    )
