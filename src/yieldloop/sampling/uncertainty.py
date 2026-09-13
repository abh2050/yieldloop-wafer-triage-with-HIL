"""Uncertainty scores over calibrated probability distributions.

Every function takes an ``(n, c)`` array of calibrated probabilities and returns
an ``(n,)`` array of scores where **higher means more informative**. That sign
convention is uniform across the module so the scheduler can combine scores
without special-casing.

Calibration matters here. Entropy over uncalibrated softmax outputs ranks by the
network's overconfidence as much as by genuine ambiguity, which is why the
scheduler consumes the temperature-scaled distribution rather than raw logits.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

Probabilities = npt.NDArray[np.float64]
Scores = npt.NDArray[np.float64]

#: Tolerance when checking that each row sums to one.
SUM_TOLERANCE: Final[float] = 1e-6


class ProbabilityError(ValueError):
    """Raised when an array is not a valid batch of probability distributions."""


def validate_probabilities(probabilities: npt.ArrayLike) -> Probabilities:
    """Coerce and validate an ``(n, c)`` batch of distributions.

    Raises rather than normalizing. A caller passing unnormalized scores has a
    bug, and quietly renormalizing would hide it while still producing
    plausible-looking sample selections.
    """
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim != 2:
        raise ProbabilityError(f"expected a 2-D (n, c) array; got shape {array.shape}")
    if array.shape[0] == 0:
        return array
    if array.shape[1] < 2:
        raise ProbabilityError(f"need at least two classes; got {array.shape[1]}")
    if not np.isfinite(array).all():
        raise ProbabilityError("probabilities contain NaN or infinity")
    if (array < 0.0).any():
        raise ProbabilityError("probabilities contain negative values")
    sums = array.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=SUM_TOLERANCE):
        worst = int(np.argmax(np.abs(sums - 1.0)))
        raise ProbabilityError(
            f"each row must sum to 1.0 within {SUM_TOLERANCE}; row {worst} sums to {sums[worst]!r}"
        )
    return array


def shannon_entropy(probabilities: npt.ArrayLike) -> Scores:
    """Entropy of each distribution, in nats.

    ``0 * log(0)`` is defined as 0, the measure-theoretic limit, so a confident
    one-hot prediction scores exactly zero instead of producing NaN.
    """
    array = validate_probabilities(probabilities)
    if array.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(array > 0.0, array * np.log(array), 0.0)
    # Adding +0.0 turns the IEEE -0.0 that negation produces for a confident
    # one-hot row into +0.0, so the value serializes as 0 rather than -0.
    return -terms.sum(axis=1) + 0.0


def max_entropy(class_count: int) -> float:
    """Entropy of the uniform distribution, the upper bound for ``class_count``."""
    if class_count < 2:
        raise ProbabilityError(f"need at least two classes; got {class_count}")
    return float(np.log(class_count))


def normalized_entropy(probabilities: npt.ArrayLike) -> Scores:
    """Entropy rescaled to ``[0, 1]``, so it is comparable with distance scores."""
    array = validate_probabilities(probabilities)
    if array.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return shannon_entropy(array) / max_entropy(array.shape[1])


def least_confidence(probabilities: npt.ArrayLike) -> Scores:
    """``1 - max(p)``. Higher means the top class is weaker."""
    array = validate_probabilities(probabilities)
    if array.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return 1.0 - array.max(axis=1)


def margin_score(probabilities: npt.ArrayLike) -> Scores:
    """``1 - (p_top1 - p_top2)``. Higher means the top two classes are closer.

    Distinct from entropy: a wafer split between exactly two plausible defect
    patterns is more useful to a reviewer than one spread thinly over nine, and
    margin ranks it higher while entropy does not.
    """
    array = validate_probabilities(probabilities)
    if array.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    partitioned = np.partition(array, -2, axis=1)
    top1 = partitioned[:, -1]
    top2 = partitioned[:, -2]
    return 1.0 - (top1 - top2)
