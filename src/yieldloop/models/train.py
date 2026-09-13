"""Training, calibration, and registration in one reproducible pass.

A run is fully determined by its seed, the label set it was given, and the
config. All three are hashed into the registry alongside the artifact, so a
number in the eval report can be traced to a run that can be repeated.

The split discipline is strict and worth stating: the model fits on train, the
temperature fits on validation, early stopping watches validation, and holdout is
touched exactly once, by the eval harness. Nothing here reads holdout.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sqlalchemy.orm import Session
from torch import Tensor, nn
from torch.utils.data import DataLoader

from yieldloop.config import Settings
from yieldloop.db.enums import ArtifactKind, DefectPattern, SplitName
from yieldloop.logging import get_logger
from yieldloop.models.calibrate import (
    CalibrationResult,
    calibrated_probabilities,
    expected_calibration_error,
    fit_temperature,
)
from yieldloop.models.classifier import (
    CLASS_ORDER,
    NUM_CLASSES,
    ClassCounts,
    ClassifierConfig,
    WaferCNN,
)
from yieldloop.models.embed import (
    WaferDataset,
    build_loader,
    compute_logits,
    load_samples,
    select_device,
)
from yieldloop.models.registry import ArtifactRegistry, StoredArtifact, hash_training_data

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Seed every source of randomness a run touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    seconds: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Everything a training run produced."""

    artifact: StoredArtifact
    calibration: CalibrationResult
    history: tuple[EpochMetrics, ...]
    best_epoch: int
    train_size: int
    val_size: int
    metrics: dict[str, Any]


def macro_f1(predicted: Tensor, labels: Tensor) -> float:
    """Unweighted mean F1 across classes that appear in ``labels``.

    Macro rather than micro because the class distribution is extreme: `none` is
    85% of labeled wafers, so a micro average would be dominated by it and a
    model that never predicted `near_full` would look fine.
    """
    scores: list[float] = []
    for index in range(NUM_CLASSES):
        actual = labels.eq(index)
        if not bool(actual.any()):
            continue
        guessed = predicted.eq(index)
        true_positive = float((guessed & actual).sum())
        precision_denominator = float(guessed.sum())
        recall_denominator = float(actual.sum())
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def per_class_recall(predicted: Tensor, labels: Tensor) -> dict[str, float]:
    """Recall for each class present in ``labels``. The number that matters for
    rare defect patterns."""
    recalls: dict[str, float] = {}
    for index, cls in enumerate(CLASS_ORDER):
        actual = labels.eq(index)
        support = float(actual.sum())
        if support == 0.0:
            continue
        recalls[cls.value] = float((predicted.eq(index) & actual).sum()) / support
    return recalls


def support_counts(labels: Tensor) -> dict[str, int]:
    return {
        cls.value: int(labels.eq(index).sum())
        for index, cls in enumerate(CLASS_ORDER)
        if int(labels.eq(index).sum()) > 0
    }


def _evaluate(
    model: WaferCNN,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, float, Tensor, Tensor]:
    logits, labels = compute_logits(model, loader, device)
    loss = float(criterion(logits, labels))
    predicted = logits.argmax(dim=1)
    accuracy = float(predicted.eq(labels).float().mean())
    return loss, accuracy, macro_f1(predicted, labels), logits, labels


def train_classifier(
    session: Session,
    settings: Settings,
    *,
    config: ClassifierConfig | None = None,
    train_limit: int | None = None,
    activate: bool = True,
) -> TrainingResult:
    """Train, calibrate, and register a classifier.

    Args:
        train_limit: Cap on labeled training wafers. This is the x-axis of the
            label efficiency curve: the same call with different limits produces
            the comparable points on it.
    """
    resolved = config or ClassifierConfig(
        grid_height=settings.grid_height,
        grid_width=settings.grid_width,
        embedding_dim=settings.embedding_dim,
        seed=settings.train_seed,
    )
    set_seed(resolved.seed)
    device = select_device()

    train_samples = load_samples(session, SplitName.TRAIN, limit=train_limit)
    val_samples = load_samples(session, SplitName.VAL)
    if not train_samples:
        raise ValueError("no labeled training wafers found; run scripts/bootstrap_db.py first")
    if not val_samples:
        raise ValueError("no labeled validation wafers found")

    train_set = WaferDataset(train_samples)
    val_set = WaferDataset(val_samples)
    train_loader = build_loader(
        train_set, batch_size=resolved.batch_size, shuffle=True, seed=resolved.seed
    )
    val_loader = build_loader(
        val_set, batch_size=resolved.batch_size, shuffle=False, seed=resolved.seed
    )

    model = WaferCNN(resolved).to(device)
    weights = (
        ClassCounts(train_set.class_counts()).weight_tensor(device)
        if resolved.class_weighting
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    # Evaluation is unweighted: weighting the loss is a training device to stop
    # rare classes being ignored, not a claim about how errors should be scored.
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=resolved.learning_rate, weight_decay=resolved.weight_decay
    )

    logger.info(
        "training_start",
        train=len(train_set),
        val=len(val_set),
        device=str(device),
        classes=train_set.class_counts(),
    )

    history: list[EpochMetrics] = []
    best_state: dict[str, Tensor] = {}
    best_loss = float("inf")
    best_epoch = 0
    patience = 0

    for epoch in range(1, resolved.max_epochs + 1):
        started = time.monotonic()
        model.train()
        running = 0.0
        seen = 0
        for grids, labels in train_loader:
            grids, labels = grids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(grids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * grids.shape[0]
            seen += grids.shape[0]

        val_loss, val_accuracy, val_f1, _, _ = _evaluate(model, val_loader, device, eval_criterion)
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=running / max(seen, 1),
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            val_macro_f1=val_f1,
            seconds=time.monotonic() - started,
        )
        history.append(metrics)
        logger.info(
            "epoch_complete",
            epoch=epoch,
            train_loss=round(metrics.train_loss, 4),
            val_loss=round(val_loss, 4),
            val_accuracy=round(val_accuracy, 4),
            val_macro_f1=round(val_f1, 4),
            seconds=round(metrics.seconds, 1),
        )

        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, patience = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= resolved.early_stopping_patience:
                logger.info("early_stop", epoch=epoch, best_epoch=best_epoch)
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    # Calibrate on validation, never on train or holdout.
    val_logits, val_labels = compute_logits(model, val_loader, device)
    calibration = fit_temperature(val_logits, val_labels)

    predicted = val_logits.argmax(dim=1)
    probabilities = calibrated_probabilities(val_logits, calibration.temperature)
    metrics_payload: dict[str, Any] = {
        "val_accuracy": float(predicted.eq(val_labels).float().mean()),
        "val_macro_f1": macro_f1(predicted, val_labels),
        "val_per_class_recall": per_class_recall(predicted, val_labels),
        "val_support": support_counts(val_labels),
        "val_ece_uncalibrated": calibration.ece_before,
        "val_ece_calibrated": expected_calibration_error(probabilities, val_labels),
        "temperature": calibration.temperature,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_size": len(train_set),
        "val_size": len(val_set),
        "device": str(device),
        "temperature_at_bound": calibration.at_bound,
        # How much of this model came from the console rather than the dataset.
        # Recorded so a human-corrected artifact is distinguishable from one
        # trained only on WM811K's own annotations, and so the effect of the
        # human loop is traceable rather than assumed.
        "reviewer_label_count": train_set.reviewer_label_count(),
    }

    # A temperature pinned at a bound is a degenerate fit, not a calibrated
    # model. It usually means the logits are badly scaled -- class weighting
    # flattens them, and an undertrained network flattens them further -- and the
    # routing bands would then be reading confidences that do not mean what they
    # say. Surfaced loudly rather than buried in the metrics blob.
    if calibration.at_bound:
        logger.warning(
            "calibration_hit_bound",
            temperature=calibration.temperature,
            ece_before=calibration.ece_before,
            ece_after=calibration.ece_after,
            detail=(
                "temperature scaling hit its limit; confidences are not trustworthy "
                "and the routing thresholds should not be tuned against this artifact"
            ),
        )

    registry = ArtifactRegistry(session, settings.registry_dir)
    artifact = registry.save(
        kind=ArtifactKind.CLASSIFIER,
        state={
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": resolved.as_dict(),
            "temperature": calibration.temperature,
            "class_order": [cls.value for cls in CLASS_ORDER],
        },
        data_hash=hash_training_data(train_set.label_pairs()),
        seed=resolved.seed,
        hyperparameters=resolved.as_dict(),
        metrics=metrics_payload,
        label_count=len(train_set),
        activate=activate,
    )

    logger.info(
        "training_complete",
        content_hash=artifact.content_hash[:12],
        val_accuracy=round(metrics_payload["val_accuracy"], 4),
        val_macro_f1=round(metrics_payload["val_macro_f1"], 4),
        ece_before=round(calibration.ece_before, 4),
        ece_after=round(metrics_payload["val_ece_calibrated"], 4),
        temperature=round(calibration.temperature, 4),
    )

    return TrainingResult(
        artifact=artifact,
        calibration=calibration,
        history=tuple(history),
        best_epoch=best_epoch,
        train_size=len(train_set),
        val_size=len(val_set),
        metrics=metrics_payload,
    )


def load_trained(
    session: Session, settings: Settings, *, kind: ArtifactKind = ArtifactKind.CLASSIFIER
) -> tuple[WaferCNN, float] | None:
    """Load the active classifier and its temperature, or None if there is none."""
    registry = ArtifactRegistry(session, settings.registry_dir)
    record = registry.active(kind)
    if record is None:
        return None
    state = registry.load_state(record)
    stored = state["config"]
    config = ClassifierConfig(
        grid_height=int(stored["grid_height"]),
        grid_width=int(stored["grid_width"]),
        embedding_dim=int(stored["embedding_dim"]),
        channels=tuple(int(c) for c in stored["channels"]),
        dropout=float(stored["dropout"]),
        seed=int(stored["seed"]),
    )
    model = WaferCNN(config)
    model.load_state_dict(state["model_state"])
    model.eval()

    # The stored class order must match the current enum, or every probability
    # vector read back would be silently mislabeled.
    stored_order = [str(c) for c in state["class_order"]]
    if stored_order != [cls.value for cls in CLASS_ORDER]:
        raise ValueError(
            f"artifact {record.content_hash[:12]} was trained with class order "
            f"{stored_order}, which differs from the current {[c.value for c in CLASS_ORDER]}"
        )
    return model, float(state["temperature"])


def label_distribution(samples: list[Any]) -> dict[DefectPattern, int]:
    counts = dict.fromkeys(CLASS_ORDER, 0)
    for sample in samples:
        if sample.label is not None:
            counts[sample.label] += 1
    return counts
