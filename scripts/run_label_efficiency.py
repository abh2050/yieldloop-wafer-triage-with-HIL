"""Run the label efficiency experiment: active learning against random sampling.

The claim under test is that entropy-plus-diversity selection reaches a given
quality with fewer human labels than random selection. The only honest way to
measure it is at **matched label counts**, which is what this script produces:
both arms are trained at the same budgets, on the same splits, with the same
hyperparameters and seed, differing only in which wafers were chosen.

The active arm is simulated iteratively, not in one shot. It starts from a random
seed set, trains, scores the pool with *that* model, selects the next batch, and
retrains. Selecting all at once with a model trained on everything would leak the
full-data model into the selection and overstate the result -- which is the most
common way this experiment is reported wrongly.

WM811K's human labels stand in for the reviewer: a wafer is "unlabeled" until the
simulated round selects it, at which point its real label is revealed. No label is
invented, and the reviewer being simulated is the one who annotated the dataset.

Usage::

    python -m scripts.run_label_efficiency --budgets 1000 2000 4000 8000
    python -m scripts.run_label_efficiency --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from eval.metrics.classification import classification_report
from eval.metrics.label_efficiency import EfficiencyPoint, compare, curve_points
from sqlalchemy.orm import Session

from yieldloop.config import Settings, get_settings
from yieldloop.db.enums import SamplingStrategy, SplitName
from yieldloop.db.session import build_engine
from yieldloop.logging import configure_logging, get_logger
from yieldloop.models.classifier import CLASS_ORDER, ClassCounts, ClassifierConfig, WaferCNN
from yieldloop.models.embed import (
    WaferDataset,
    WaferSample,
    build_loader,
    compute_logits,
    load_samples,
    select_device,
)
from yieldloop.models.train import macro_f1, per_class_recall, set_seed
from yieldloop.sampling.scheduler import SelectionRequest, select_batch

logger = get_logger(__name__)

LABELS = [cls.value for cls in CLASS_ORDER]

#: Classes below this share of the labeled population. Active learning should
#: help most here, and random sampling struggles to find examples at all.
RARE_SHARE = 0.01

#: A class needs at least this many evaluation wafers for its recall to mean
#: anything. With 8 examples, recall can only take the values 0, 0.125, 0.25 and
#: so on, and the movement between arms is quantization rather than signal. The
#: rare-class figure is reported with its support so it cannot be read as a
#: finding when it is not one.
MIN_SUPPORT_FOR_RECALL = 30


@dataclass(frozen=True, slots=True)
class ArmResult:
    strategy: SamplingStrategy
    points: list[EfficiencyPoint]
    #: Smallest per-class evaluation support among the rare classes.
    rare_class_support: int


def _rare_class_recall(recalls: dict[str, float], counts: dict[str, int]) -> tuple[float, int]:
    """Mean recall over rare classes, and the *smallest* per-class support in it.

    The minimum rather than the sum, because the average is only as trustworthy
    as its weakest term: a mean over one class with 74 wafers and another with 8
    is dominated by the quantization of the second, and summing them to 82 would
    make a figure look well-supported precisely when it is not.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0
    rare = [label for label, count in counts.items() if count / total < RARE_SHARE]
    if not rare:
        return 0.0, 0
    weakest = min(counts[label] for label in rare)
    return float(np.mean([recalls.get(label, 0.0) for label in rare])), weakest


def _train_once(
    samples: list[WaferSample],
    validation: WaferDataset,
    config: ClassifierConfig,
    device: torch.device,
) -> tuple[float, float, dict[str, float], dict[str, int]]:
    """Train on ``samples`` and score on ``validation``.

    Returns macro F1, accuracy, per-class recall, and support.
    """
    set_seed(config.seed)
    dataset = WaferDataset(samples)
    loader = build_loader(dataset, batch_size=config.batch_size, shuffle=True, seed=config.seed)
    val_loader = build_loader(
        validation, batch_size=config.batch_size, shuffle=False, seed=config.seed
    )

    model = WaferCNN(config).to(device)
    weights = ClassCounts(dataset.class_counts()).weight_tensor(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    patience = 0
    evaluation = torch.nn.CrossEntropyLoss()

    for _epoch in range(config.max_epochs):
        model.train()
        for grids, targets in loader:
            grids, targets = grids.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(grids)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        logits, targets = compute_logits(model, val_loader, device)
        val_loss = float(evaluation(logits, targets))
        if val_loss < best_loss - 1e-5:
            best_loss, patience = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    logits, targets = compute_logits(model, val_loader, device)
    predicted = logits.argmax(dim=1)
    report = classification_report(
        predicted.numpy().astype(np.int64), targets.numpy().astype(np.int64), LABELS
    )
    return (
        macro_f1(predicted, targets),
        report.accuracy,
        per_class_recall(predicted, targets),
        {m.label: m.support for m in report.per_class},
    )


def _score_pool(
    samples: list[WaferSample],
    config: ClassifierConfig,
    device: torch.device,
    model: WaferCNN,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated-ish probabilities and embeddings for a candidate pool.

    Uses the softmax of the current arm's model. No temperature is fitted here:
    the selection only needs a ranking, and rank order is invariant under
    temperature scaling.
    """
    dataset = WaferDataset(samples)
    loader = build_loader(dataset, batch_size=config.batch_size, shuffle=False, seed=config.seed)
    logit_batches: list[torch.Tensor] = []
    embedding_batches: list[torch.Tensor] = []
    with torch.no_grad():
        model.eval()
        for grids, _ in loader:
            logits, embeddings = model(grids.to(device))
            logit_batches.append(logits.cpu())
            embedding_batches.append(embeddings.cpu())
    probabilities = torch.cat(logit_batches).softmax(dim=1).numpy().astype(np.float64)
    return probabilities, torch.cat(embedding_batches).numpy().astype(np.float64)


def _fit_selector(
    samples: list[WaferSample], config: ClassifierConfig, device: torch.device
) -> WaferCNN:
    """Train a model on the currently-labeled set, for use as the selector."""
    set_seed(config.seed)
    dataset = WaferDataset(samples)
    loader = build_loader(dataset, batch_size=config.batch_size, shuffle=True, seed=config.seed)
    model = WaferCNN(config).to(device)
    weights = ClassCounts(dataset.class_counts()).weight_tensor(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    for _epoch in range(min(config.max_epochs, 6)):
        model.train()
        for grids, targets in loader:
            grids, targets = grids.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(grids)
            criterion(logits, targets).backward()
            optimizer.step()
    return model


def run_arm(
    *,
    strategy: SamplingStrategy,
    pool: list[WaferSample],
    validation: WaferDataset,
    budgets: list[int],
    config: ClassifierConfig,
    device: torch.device,
    seed: int,
) -> ArmResult:
    """Train one arm at every budget and return its curve points."""
    rng = np.random.default_rng(seed)
    by_id = {sample.wafer_id: sample for sample in pool}
    ordered_budgets = sorted(budgets)

    if strategy is SamplingStrategy.RANDOM:
        permutation = rng.permutation(len(pool))
        selected_ids = [pool[int(i)].wafer_id for i in permutation]
        acquisition = {budget: selected_ids[:budget] for budget in ordered_budgets}
    else:
        acquisition = _simulate_active(
            pool=pool, budgets=ordered_budgets, config=config, device=device, rng=rng
        )

    points: list[EfficiencyPoint] = []
    rare_supports: list[int] = []
    for budget in ordered_budgets:
        chosen = [by_id[wafer_id] for wafer_id in acquisition[budget]]
        started = time.monotonic()
        f1, accuracy, recalls, support = _train_once(chosen, validation, config, device)
        rare_recall, rare_support = _rare_class_recall(recalls, support)
        points.append(
            EfficiencyPoint(
                strategy=strategy.value,
                label_count=len(chosen),
                macro_f1=f1,
                accuracy=accuracy,
                rare_class_recall=rare_recall,
            )
        )
        rare_supports.append(rare_support)
        logger.info(
            "efficiency_point",
            strategy=strategy.value,
            labels=len(chosen),
            macro_f1=round(f1, 4),
            accuracy=round(accuracy, 4),
            seconds=round(time.monotonic() - started, 1),
        )
    return ArmResult(
        strategy=strategy,
        points=points,
        rare_class_support=rare_supports[0] if rare_supports else 0,
    )


def _simulate_active(
    *,
    pool: list[WaferSample],
    budgets: list[int],
    config: ClassifierConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[int, list[str]]:
    """Acquire labels iteratively, retraining the selector at each step.

    The selector at step k is trained only on what has been acquired by step k.
    Scoring the pool with a model that has seen the whole dataset would leak it
    into the selection and inflate the result.
    """
    seed_size = budgets[0]
    permutation = rng.permutation(len(pool))
    acquired = [pool[int(i)].wafer_id for i in permutation[:seed_size]]
    acquisition = {seed_size: list(acquired)}

    by_id = {sample.wafer_id: sample for sample in pool}
    for budget in budgets[1:]:
        need = budget - len(acquired)
        acquired_set = set(acquired)
        remaining = [s for s in pool if s.wafer_id not in acquired_set]
        if not remaining or need <= 0:
            acquisition[budget] = list(acquired)
            continue

        selector = _fit_selector([by_id[i] for i in acquired], config, device)
        probabilities, embeddings = _score_pool(remaining, config, device, selector)
        selection = select_batch(
            SelectionRequest(
                wafer_ids=[s.wafer_id for s in remaining],
                probabilities=probabilities,
                embeddings=embeddings,
                labeled_ids=frozenset(acquired_set),
                strategy=SamplingStrategy.ENTROPY_DIVERSITY,
                batch_size=need,
                diversity_weight=0.5,
                min_distance=0.0,
                seed=config.seed,
            )
        )
        acquired.extend(selection.wafer_ids)
        acquisition[budget] = list(acquired)
        logger.info("active_acquired", budget=budget, total=len(acquired))
    return acquisition


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[1000, 2000, 4000, 8000, 16000])
    parser.add_argument("--quick", action="store_true", help="small budgets, for a smoke run")
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--val-limit", type=int, default=8000)
    parser.add_argument("--pool-limit", type=int, default=None)
    parser.add_argument("--target", type=float, default=0.60)
    parser.add_argument("--output", type=Path, default=Path("eval/label_efficiency.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings: Settings = get_settings()
    configure_logging(settings)

    budgets = [250, 500, 1000, 2000] if args.quick else args.budgets
    if len(budgets) < 2:
        print("need at least two budgets to draw a curve", file=sys.stderr)
        return 2

    config = ClassifierConfig(
        grid_height=settings.grid_height,
        grid_width=settings.grid_width,
        embedding_dim=settings.embedding_dim,
        seed=settings.train_seed,
        max_epochs=args.max_epochs,
        early_stopping_patience=3,
    )
    device = select_device()

    with Session(build_engine(settings)) as session:
        pool = load_samples(session, SplitName.TRAIN, labeled=True, limit=args.pool_limit)
        validation = WaferDataset(
            load_samples(session, SplitName.VAL, labeled=True, limit=args.val_limit)
        )

    if len(pool) < max(budgets):
        print(
            f"pool has {len(pool):,} labeled wafers, fewer than the largest budget "
            f"{max(budgets):,}",
            file=sys.stderr,
        )
        return 1

    logger.info(
        "label_efficiency_start",
        pool=len(pool),
        validation=len(validation),
        budgets=budgets,
        device=str(device),
    )
    started = time.monotonic()

    active = run_arm(
        strategy=SamplingStrategy.ENTROPY_DIVERSITY,
        pool=pool,
        validation=validation,
        budgets=budgets,
        config=config,
        device=device,
        seed=settings.partition_seed,
    )
    control = run_arm(
        strategy=SamplingStrategy.RANDOM,
        pool=pool,
        validation=validation,
        budgets=budgets,
        config=config,
        device=device,
        seed=settings.partition_seed,
    )

    comparison = compare(active.points, control.points, target=args.target)
    payload = {
        "budgets": budgets,
        "pool_size": len(pool),
        "validation_size": len(validation),
        "target_macro_f1": args.target,
        "active": curve_points(active.points),
        "random": curve_points(control.points),
        "comparison": comparison.as_dict(),
        "rare_class_support": active.rare_class_support,
        "min_support_for_recall": MIN_SUPPORT_FOR_RECALL,
        "rare_class_recall_measurable": active.rare_class_support >= MIN_SUPPORT_FOR_RECALL,
        "seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nlabel efficiency written to {args.output}")
    print(f"  mean macro F1 delta (active - random): {comparison.mean_delta:+.4f}")
    print(
        f"  labels to reach {args.target}: active={comparison.strategy_labels_to_target} "
        f"random={comparison.baseline_labels_to_target}"
    )
    if comparison.label_saving_ratio is not None:
        print(f"  label saving: {comparison.label_saving_ratio:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
