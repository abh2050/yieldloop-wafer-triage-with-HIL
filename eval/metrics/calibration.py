"""Calibration and routing metrics.

This module produces one of the two curves the repository is built around: the
routing tradeoff. Sweeping the confidence floor and the auto-commit threshold
shows exactly what automation rate costs in accuracy, which is the question a
fab actually asks before letting a model commit anything unreviewed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        return abs(self.accuracy - self.confidence)


@dataclass(frozen=True, slots=True)
class RoutingPoint:
    """One point on the tradeoff curve."""

    confidence_floor: float
    auto_commit_threshold: float
    #: Fraction committed without a human.
    automation_rate: float
    #: Accuracy among auto-committed predictions. The number that matters: these
    #: are the errors that reach production unreviewed.
    auto_commit_accuracy: float
    #: Errors that escaped review, as a fraction of all samples.
    escaped_error_rate: float
    #: Fraction routed to a human with the prediction shown.
    review_rate: float
    #: Fraction routed to a human with the prediction withheld.
    blind_review_rate: float
    human_load: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "confidence_floor": self.confidence_floor,
            "auto_commit_threshold": self.auto_commit_threshold,
            "automation_rate": self.automation_rate,
            "auto_commit_accuracy": self.auto_commit_accuracy,
            "escaped_error_rate": self.escaped_error_rate,
            "review_rate": self.review_rate,
            "blind_review_rate": self.blind_review_rate,
            "human_load": self.human_load,
        }


def expected_calibration_error(
    confidence: FloatArray, correct: npt.NDArray[np.bool_], bins: int = 15
) -> float:
    """Equal-width binned ECE."""
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must have the same shape")
    if confidence.size == 0:
        return 0.0
    if bins < 1:
        raise ValueError(f"bins must be positive; got {bins}")

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


def maximum_calibration_error(
    confidence: FloatArray, correct: npt.NDArray[np.bool_], bins: int = 15
) -> float:
    """Worst-case bin gap.

    Reported alongside ECE because a model can have a small average gap while
    being badly wrong in exactly the high-confidence bin the auto-commit
    threshold sits in.
    """
    if confidence.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    worst = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = (confidence > low) & (confidence <= high) if index else (confidence <= high)
        if not bool(mask.any()):
            continue
        worst = max(worst, abs(float(correct[mask].mean()) - float(confidence[mask].mean())))
    return worst


def reliability_bins(
    confidence: FloatArray, correct: npt.NDArray[np.bool_], bins: int = 15
) -> list[ReliabilityBin]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result: list[ReliabilityBin] = []
    for index in range(bins):
        low, high = float(edges[index]), float(edges[index + 1])
        mask = (confidence > low) & (confidence <= high) if index else (confidence <= high)
        count = int(mask.sum())
        result.append(
            ReliabilityBin(
                lower=low,
                upper=high,
                count=count,
                confidence=float(confidence[mask].mean()) if count else 0.0,
                accuracy=float(correct[mask].mean()) if count else 0.0,
            )
        )
    return result


def routing_point(
    confidence: FloatArray,
    correct: npt.NDArray[np.bool_],
    *,
    confidence_floor: float,
    auto_commit_threshold: float,
) -> RoutingPoint:
    """Evaluate one threshold pair."""
    if not 0.0 < confidence_floor < auto_commit_threshold <= 1.0:
        raise ValueError(
            "require 0 < confidence_floor < auto_commit_threshold <= 1; got "
            f"{confidence_floor} and {auto_commit_threshold}"
        )
    if confidence.size == 0:
        return RoutingPoint(
            confidence_floor=confidence_floor,
            auto_commit_threshold=auto_commit_threshold,
            automation_rate=0.0,
            auto_commit_accuracy=0.0,
            escaped_error_rate=0.0,
            review_rate=0.0,
            blind_review_rate=0.0,
            human_load=0,
        )

    committed = confidence >= auto_commit_threshold
    blind = confidence < confidence_floor
    reviewed = ~committed & ~blind
    total = confidence.size
    committed_count = int(committed.sum())

    escaped_errors = int((committed & ~correct).sum())
    return RoutingPoint(
        confidence_floor=confidence_floor,
        auto_commit_threshold=auto_commit_threshold,
        automation_rate=committed_count / total,
        auto_commit_accuracy=(float(correct[committed].mean()) if committed_count else 0.0),
        escaped_error_rate=escaped_errors / total,
        review_rate=int(reviewed.sum()) / total,
        blind_review_rate=int(blind.sum()) / total,
        human_load=total - committed_count,
    )


def routing_curve(
    confidence: FloatArray,
    correct: npt.NDArray[np.bool_],
    *,
    confidence_floor: float = 0.55,
    thresholds: list[float] | None = None,
) -> list[RoutingPoint]:
    """Sweep the auto-commit threshold. One of the repository's two charts.

    The floor is held fixed while the auto-commit threshold moves, because they
    answer different questions: the threshold decides how much is automated, and
    the floor decides how much of what a human sees is shown to them.
    """
    sweep = thresholds or [round(0.50 + 0.02 * i, 2) for i in range(26)]
    return [
        routing_point(
            confidence,
            correct,
            confidence_floor=confidence_floor,
            auto_commit_threshold=threshold,
        )
        for threshold in sweep
        if threshold > confidence_floor
    ]


def floor_sweep(
    confidence: FloatArray,
    correct: npt.NDArray[np.bool_],
    *,
    auto_commit_threshold: float = 0.95,
    floors: list[float] | None = None,
) -> list[RoutingPoint]:
    """Sweep the confidence floor, holding auto-commit fixed."""
    sweep = floors or [round(0.10 + 0.05 * i, 2) for i in range(17)]
    return [
        routing_point(
            confidence,
            correct,
            confidence_floor=floor,
            auto_commit_threshold=auto_commit_threshold,
        )
        for floor in sweep
        if floor < auto_commit_threshold
    ]
