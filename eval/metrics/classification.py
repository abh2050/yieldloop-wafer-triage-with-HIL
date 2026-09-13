"""Classification metrics.

Accuracy is reported but never alone. 85% of labeled WM811K wafers are `none`, so
a model that predicts `none` unconditionally scores 85% and has learned nothing.
Macro F1 and per-class recall are what actually distinguish a useful classifier
here, and the harness prints them together for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """Per-class precision, recall, F1, and support."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    accuracy: float
    macro_f1: float
    #: Macro recall, equally weighting every class regardless of frequency.
    macro_recall: float
    balanced_accuracy: float
    per_class: tuple[ClassMetrics, ...]
    confusion: list[list[int]]
    samples: int

    def recall_for(self, label: str) -> float | None:
        for metrics in self.per_class:
            if metrics.label == label:
                return metrics.recall
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "macro_recall": self.macro_recall,
            "balanced_accuracy": self.balanced_accuracy,
            "samples": self.samples,
            "per_class": {
                m.label: {
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "support": m.support,
                }
                for m in self.per_class
            },
        }


def _validate(predicted: IntArray, actual: IntArray, class_count: int) -> None:
    if predicted.shape != actual.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} against {actual.shape}")
    if predicted.ndim != 1:
        raise ValueError(f"expected 1-D arrays; got {predicted.ndim}-D")
    for name, array in (("predicted", predicted), ("actual", actual)):
        if array.size and (int(array.max()) >= class_count or int(array.min()) < 0):
            raise ValueError(f"{name} contains a class index outside [0, {class_count})")


def confusion_matrix(
    predicted: IntArray, actual: IntArray, class_count: int
) -> npt.NDArray[np.int64]:
    """Rows are actual classes, columns predicted."""
    _validate(predicted, actual, class_count)
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (actual, predicted), 1)
    return matrix


def classification_report(
    predicted: IntArray, actual: IntArray, labels: list[str]
) -> ClassificationReport:
    """Compute the full report.

    Classes with zero support are omitted from the macro averages rather than
    counted as zero. Averaging in a class that does not appear in the evaluation
    set measures the size of the label space, not the quality of the model.
    """
    class_count = len(labels)
    _validate(predicted, actual, class_count)
    if predicted.size == 0:
        return ClassificationReport(
            accuracy=0.0,
            macro_f1=0.0,
            macro_recall=0.0,
            balanced_accuracy=0.0,
            per_class=(),
            confusion=[[0] * class_count for _ in range(class_count)],
            samples=0,
        )

    matrix = confusion_matrix(predicted, actual, class_count)
    per_class: list[ClassMetrics] = []
    for index, label in enumerate(labels):
        support = int(matrix[index].sum())
        if support == 0:
            continue
        true_positive = int(matrix[index, index])
        predicted_positive = int(matrix[:, index].sum())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / support
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            ClassMetrics(label=label, precision=precision, recall=recall, f1=f1, support=support)
        )

    accuracy = float((predicted == actual).mean())
    macro_f1 = float(np.mean([m.f1 for m in per_class])) if per_class else 0.0
    macro_recall = float(np.mean([m.recall for m in per_class])) if per_class else 0.0

    return ClassificationReport(
        accuracy=accuracy,
        macro_f1=macro_f1,
        macro_recall=macro_recall,
        # Balanced accuracy is macro recall; named separately because it is the
        # figure to quote against a majority-class baseline.
        balanced_accuracy=macro_recall,
        per_class=tuple(per_class),
        confusion=matrix.tolist(),
        samples=int(predicted.size),
    )


def majority_class_baseline(actual: IntArray, labels: list[str]) -> ClassificationReport:
    """What a model that always predicts the commonest class would score.

    Reported alongside the real numbers so accuracy cannot be read as impressive
    without that context.
    """
    if actual.size == 0:
        return classification_report(actual, actual, labels)
    majority = int(np.bincount(actual, minlength=len(labels)).argmax())
    return classification_report(np.full_like(actual, majority), actual, labels)
