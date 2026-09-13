"""Classifier evaluation on the real holdout split.

Holdout is touched here and nowhere else in the system. Training fits on train,
temperature fits on validation, and early stopping watches validation, so every
number below comes from wafers the model has provably never seen -- and because
partitioning is lot-keyed, from lots it has never seen either.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sqlalchemy.orm import Session

from eval.datasets import GoldenSet, load_golden_set, rare_classes
from eval.metrics.calibration import (
    expected_calibration_error,
    floor_sweep,
    maximum_calibration_error,
    reliability_bins,
    routing_curve,
)
from eval.metrics.classification import (
    ClassificationReport,
    classification_report,
    majority_class_baseline,
)
from eval.metrics.escalation import summarize as summarize_escalation
from yieldloop.config import Settings
from yieldloop.db.enums import SplitName
from yieldloop.logging import get_logger
from yieldloop.models.calibrate import calibrated_probabilities
from yieldloop.models.classifier import CLASS_ORDER, WaferCNN
from yieldloop.models.embed import WaferDataset, build_loader, compute_logits, select_device
from yieldloop.models.train import load_trained

logger = get_logger(__name__)

LABELS = [cls.value for cls in CLASS_ORDER]


@dataclass(frozen=True, slots=True)
class ClassifierSuiteResult:
    report: ClassificationReport
    baseline: ClassificationReport
    ece_calibrated: float
    ece_uncalibrated: float
    mce_calibrated: float
    temperature: float
    routing: list[dict[str, Any]]
    floor_sweep: list[dict[str, Any]]
    reliability: list[dict[str, Any]]
    escalation: dict[str, Any]
    rare_class_recall: dict[str, float]
    samples: int
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "holdout": self.report.as_dict(),
            "majority_class_baseline": self.baseline.as_dict(),
            "calibration": {
                "temperature": self.temperature,
                "ece_uncalibrated": self.ece_uncalibrated,
                "ece_calibrated": self.ece_calibrated,
                "mce_calibrated": self.mce_calibrated,
                "reliability": self.reliability,
            },
            "routing_curve": self.routing,
            "floor_sweep": self.floor_sweep,
            "escalation": self.escalation,
            "rare_class_recall": self.rare_class_recall,
            "samples": self.samples,
            "seconds": self.seconds,
        }


def run(
    session: Session, settings: Settings, *, limit: int | None = None
) -> ClassifierSuiteResult | None:
    """Evaluate the active classifier. Returns None when there is none."""
    loaded = load_trained(session, settings)
    if loaded is None:
        logger.warning("classifier_suite_skipped", reason="no active classifier")
        return None
    model, temperature = loaded

    golden = load_golden_set(session, SplitName.HOLDOUT, limit=limit)
    if len(golden) == 0:
        logger.warning("classifier_suite_skipped", reason="no labeled holdout wafers")
        return None

    started = time.monotonic()
    device = select_device()
    model.to(device)
    logits, actual = _predict(model, golden, device)

    uncalibrated = logits.softmax(dim=1)
    probabilities = calibrated_probabilities(logits, temperature)
    predicted = probabilities.argmax(dim=1)

    predicted_np = predicted.numpy().astype(np.int64)
    actual_np = actual.numpy().astype(np.int64)
    confidence = probabilities.max(dim=1).values.numpy().astype(np.float64)
    correct = predicted.eq(actual).numpy()

    report = classification_report(predicted_np, actual_np, LABELS)
    rare = rare_classes(session, SplitName.HOLDOUT)

    return ClassifierSuiteResult(
        report=report,
        baseline=majority_class_baseline(actual_np, LABELS),
        ece_calibrated=expected_calibration_error(confidence, correct),
        ece_uncalibrated=expected_calibration_error(
            uncalibrated.max(dim=1).values.numpy().astype(np.float64),
            uncalibrated.argmax(dim=1).eq(actual).numpy(),
        ),
        mce_calibrated=maximum_calibration_error(confidence, correct),
        temperature=temperature,
        routing=[
            point.as_dict()
            for point in routing_curve(
                confidence, correct, confidence_floor=settings.confidence_floor
            )
        ],
        floor_sweep=[
            point.as_dict()
            for point in floor_sweep(
                confidence, correct, auto_commit_threshold=settings.auto_commit_threshold
            )
        ],
        reliability=[
            {
                "lower": b.lower,
                "upper": b.upper,
                "count": b.count,
                "confidence": b.confidence,
                "accuracy": b.accuracy,
                "gap": b.gap,
            }
            for b in reliability_bins(confidence, correct)
        ],
        escalation=summarize_escalation(
            list(map(float, confidence)),
            [bool(c) for c in correct],
            confidence_floor=settings.confidence_floor,
            auto_commit_threshold=settings.auto_commit_threshold,
        ).as_dict(),
        rare_class_recall={label: report.recall_for(label) or 0.0 for label in rare},
        samples=int(predicted.numel()),
        seconds=time.monotonic() - started,
    )


def _predict(
    model: WaferCNN, golden: GoldenSet, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = WaferDataset(list(golden.samples))
    loader = build_loader(dataset, batch_size=256, shuffle=False, seed=0)
    return compute_logits(model, loader, device)
