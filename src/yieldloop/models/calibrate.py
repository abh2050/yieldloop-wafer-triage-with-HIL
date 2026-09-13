"""Temperature scaling.

Every routing decision in this system is a comparison between a confidence and a
threshold, so those confidences have to mean something. A raw softmax over a
network trained with cross-entropy does not: it is systematically overconfident,
and an uncalibrated 0.95 is not a 95% chance of being right. Setting an
auto-commit threshold against uncalibrated scores would commit a known fraction
of errors without review while appearing rigorous.

Temperature scaling fits a single scalar on the validation split, dividing the
logits before the softmax. One parameter is the point: it cannot change which
class is predicted, only how confident the model claims to be. Accuracy is
therefore unchanged by construction, and any improvement in expected calibration
error is real rather than a re-fit.

Fitting on validation, never on training or holdout: training logits are
overfit and would yield a temperature near 1, and fitting on holdout would leak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor

#: Bounds for the fitted temperature. A value pinned at a bound means something
#: is wrong upstream rather than that the model is unusually (mis)calibrated.
MIN_TEMPERATURE: Final[float] = 0.05
MAX_TEMPERATURE: Final[float] = 10.0

#: Bins used for expected calibration error.
DEFAULT_ECE_BINS: Final[int] = 15


class CalibrationError(ValueError):
    """Raised when calibration inputs are malformed."""


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """A fitted temperature and what it achieved."""

    temperature: float
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float
    samples: int

    @property
    def improved(self) -> bool:
        return self.ece_after <= self.ece_before

    @property
    def at_bound(self) -> bool:
        """True when the fit hit a limit, which is a signal to investigate."""
        return self.temperature <= MIN_TEMPERATURE * 1.001 or (
            self.temperature >= MAX_TEMPERATURE * 0.999
        )

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "temperature": self.temperature,
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "samples": self.samples,
            "improved": self.improved,
        }


def _validate(logits: Tensor, labels: Tensor) -> None:
    if logits.dim() != 2:
        raise CalibrationError(f"logits must be (n, c); got {tuple(logits.shape)}")
    if labels.dim() != 1:
        raise CalibrationError(f"labels must be (n,); got {tuple(labels.shape)}")
    if logits.shape[0] != labels.shape[0]:
        raise CalibrationError(
            f"{logits.shape[0]} logit rows against {labels.shape[0]} labels"
        )
    if logits.shape[0] == 0:
        raise CalibrationError("cannot calibrate on an empty validation set")
    if int(labels.max()) >= logits.shape[1] or int(labels.min()) < 0:
        raise CalibrationError("labels contain a class index outside the logit width")


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Divide logits by the temperature."""
    if temperature <= 0.0:
        raise CalibrationError(f"temperature must be positive; got {temperature}")
    return logits / temperature


def expected_calibration_error(
    probabilities: Tensor, labels: Tensor, bins: int = DEFAULT_ECE_BINS
) -> float:
    """Equal-width binned ECE over the predicted class.

    The gap between confidence and accuracy in each bin, weighted by bin
    population. Zero means that among predictions made at confidence p, exactly a
    p fraction were correct.
    """
    if bins < 1:
        raise CalibrationError(f"bins must be positive; got {bins}")
    if probabilities.shape[0] == 0:
        return 0.0

    confidence, predicted = probabilities.max(dim=1)
    correct = predicted.eq(labels).float()

    edges = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)
    total = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        # Lower-exclusive except in the first bin, so every sample lands once.
        in_bin = (confidence > low) & (confidence <= high) if index else (confidence <= high)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_accuracy = float(correct[in_bin].mean())
        bin_confidence = float(confidence[in_bin].mean())
        total += (count / probabilities.shape[0]) * abs(bin_accuracy - bin_confidence)
    return total


def fit_temperature(
    logits: Tensor,
    labels: Tensor,
    *,
    max_iterations: int = 200,
    bins: int = DEFAULT_ECE_BINS,
) -> CalibrationResult:
    """Fit the temperature that minimizes validation NLL.

    Optimized with LBFGS over ``log(temperature)`` rather than the temperature
    itself, which keeps the parameter positive without a constraint and makes the
    search scale-free.
    """
    _validate(logits, labels)
    working = logits.detach().float()
    targets = labels.detach().long()

    criterion = torch.nn.CrossEntropyLoss()
    nll_before = float(criterion(working, targets))
    ece_before = expected_calibration_error(working.softmax(dim=1), targets, bins)

    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iterations)

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(MIN_TEMPERATURE, MAX_TEMPERATURE)
        loss: Tensor = criterion(working / temperature, targets)
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    # torch.optim.LBFGS.step is untyped in the stubs.
    optimizer.step(closure)  # type: ignore[no-untyped-call]

    temperature = float(log_temperature.detach().exp().clamp(MIN_TEMPERATURE, MAX_TEMPERATURE))
    scaled = apply_temperature(working, temperature)
    nll_after = float(criterion(scaled, targets))
    ece_after = expected_calibration_error(scaled.softmax(dim=1), targets, bins)

    return CalibrationResult(
        temperature=temperature,
        nll_before=nll_before,
        nll_after=nll_after,
        ece_before=ece_before,
        ece_after=ece_after,
        samples=int(working.shape[0]),
    )


def calibrated_probabilities(logits: Tensor, temperature: float) -> Tensor:
    """Softmax over temperature-scaled logits."""
    return apply_temperature(logits, temperature).softmax(dim=1)


def reliability_curve(
    probabilities: Tensor, labels: Tensor, bins: int = DEFAULT_ECE_BINS
) -> list[dict[str, float]]:
    """Per-bin confidence, accuracy, and population, for the model health screen."""
    confidence, predicted = probabilities.max(dim=1)
    correct = predicted.eq(labels).float()
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)

    curve: list[dict[str, float]] = []
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        in_bin = (confidence > low) & (confidence <= high) if index else (confidence <= high)
        count = int(in_bin.sum())
        curve.append(
            {
                "bin_lower": float(low),
                "bin_upper": float(high),
                "count": count,
                "confidence": float(confidence[in_bin].mean()) if count else 0.0,
                "accuracy": float(correct[in_bin].mean()) if count else 0.0,
            }
        )
    return curve
