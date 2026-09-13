"""Routing bands: who sees what, and what they are allowed to see.

Three bands, two boundaries, both configuration rather than constants so the eval
harness can sweep them and produce the routing tradeoff curve.

* At or above ``auto_commit_threshold``: committed without a human.
* Between the two: routed to a human **with** the prediction shown.
* Below ``confidence_floor``: routed to a human with the prediction **withheld**.

The third band is the one that is easy to get wrong. Showing a low-confidence
prediction to a reviewer does not help them; it anchors them. A reviewer told
"the model thinks this is a scratch, but it is not sure" agrees more often than a
reviewer shown the same wafer cold, and that agreement is not signal -- it is the
model's error laundered into the training set as a human label. So below the
floor the console does not render the prediction at all, and
``decisions.prediction_was_shown`` records which regime each decision was made
under so the anchoring effect can be measured rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass

from yieldloop.config import Settings
from yieldloop.db.enums import RoutingBand


@dataclass(frozen=True, slots=True)
class RoutingBands:
    """An immutable pair of band boundaries."""

    confidence_floor: float
    auto_commit_threshold: float

    def __post_init__(self) -> None:
        for name, value in (
            ("confidence_floor", self.confidence_floor),
            ("auto_commit_threshold", self.auto_commit_threshold),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]; got {value}")
        if self.confidence_floor >= self.auto_commit_threshold:
            raise ValueError(
                "confidence_floor must be strictly below auto_commit_threshold so the "
                f"uncertainty band is non-empty; got floor={self.confidence_floor} "
                f"auto_commit={self.auto_commit_threshold}"
            )

    @classmethod
    def from_settings(cls, settings: Settings) -> RoutingBands:
        return cls(
            confidence_floor=settings.confidence_floor,
            auto_commit_threshold=settings.auto_commit_threshold,
        )


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Where a prediction goes, and what the reviewer may be shown."""

    band: RoutingBand
    #: True only in the auto-commit band.
    requires_human: bool
    #: False below the floor. The console must honour this.
    show_prediction: bool
    bands: RoutingBands

    @property
    def is_auto_committed(self) -> bool:
        return self.band is RoutingBand.AUTO_COMMIT


def classify(confidence: float, bands: RoutingBands) -> RoutingDecision:
    """Route one calibrated confidence.

    Raises:
        ValueError: if ``confidence`` is not a probability. A caller passing a
            logit or an uncalibrated score would otherwise silently auto-commit
            everything.
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be a calibrated probability in [0, 1]; got {confidence}")

    if confidence >= bands.auto_commit_threshold:
        return RoutingDecision(
            band=RoutingBand.AUTO_COMMIT,
            requires_human=False,
            show_prediction=True,
            bands=bands,
        )
    if confidence >= bands.confidence_floor:
        return RoutingDecision(
            band=RoutingBand.UNCERTAINTY_BAND,
            requires_human=True,
            show_prediction=True,
            bands=bands,
        )
    return RoutingDecision(
        band=RoutingBand.BELOW_FLOOR,
        requires_human=True,
        show_prediction=False,
        bands=bands,
    )


def automation_rate(confidences: list[float], bands: RoutingBands) -> float:
    """Fraction of predictions that would be committed without a human.

    The y-axis of the routing tradeoff curve.
    """
    if not confidences:
        return 0.0
    committed = sum(1 for c in confidences if classify(c, bands).is_auto_committed)
    return committed / len(confidences)
