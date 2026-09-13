"""Label efficiency: the other curve the repository is built around.

The claim under test is that entropy-plus-diversity sampling reaches a given
quality with fewer human labels than random sampling. The only honest way to
measure it is to compare at *matched label counts*, which is what this module
enforces: a strategy that reached macro F1 0.6 after 4,000 labels is not
comparable to one that reached it after 12,000, and averaging over unmatched
points is how active learning results get overstated.

The headline number is labels-to-target: how many labels each strategy needed to
first reach a quality threshold, and the ratio between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class EfficiencyPoint:
    """One (labels, quality) observation for one strategy."""

    strategy: str
    label_count: int
    macro_f1: float
    accuracy: float
    #: Recall on the rarest classes, which is where active learning should help
    #: most and where a random sampler struggles to find examples at all.
    rare_class_recall: float


@dataclass(frozen=True, slots=True)
class EfficiencyComparison:
    """A strategy against the random control at matched label counts."""

    strategy: str
    baseline: str
    matched_counts: tuple[int, ...]
    strategy_scores: tuple[float, ...]
    baseline_scores: tuple[float, ...]
    #: Mean improvement in macro F1 across matched points.
    mean_delta: float
    #: Labels each needed to first reach the target, and the saving.
    target: float
    strategy_labels_to_target: int | None
    baseline_labels_to_target: int | None

    @property
    def label_saving_ratio(self) -> float | None:
        """How many times fewer labels the strategy needed. None if either never
        reached the target, which is itself a result worth reporting."""
        if not self.strategy_labels_to_target or not self.baseline_labels_to_target:
            return None
        return self.baseline_labels_to_target / self.strategy_labels_to_target

    @property
    def wins(self) -> bool:
        return self.mean_delta > 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "baseline": self.baseline,
            "matched_counts": list(self.matched_counts),
            "strategy_scores": list(self.strategy_scores),
            "baseline_scores": list(self.baseline_scores),
            "mean_delta": self.mean_delta,
            "target": self.target,
            "strategy_labels_to_target": self.strategy_labels_to_target,
            "baseline_labels_to_target": self.baseline_labels_to_target,
            "label_saving_ratio": self.label_saving_ratio,
        }


def labels_to_target(points: list[EfficiencyPoint], target: float) -> int | None:
    """Fewest labels at which a strategy first reached ``target`` macro F1."""
    reaching = [
        p.label_count for p in sorted(points, key=lambda p: p.label_count) if p.macro_f1 >= target
    ]
    return reaching[0] if reaching else None


def compare(
    strategy_points: list[EfficiencyPoint],
    baseline_points: list[EfficiencyPoint],
    *,
    target: float = 0.60,
) -> EfficiencyComparison:
    """Compare a strategy against the control at matched label counts.

    Raises:
        ValueError: if the two share no label counts. Comparing them anyway would
            be interpolating between experiments that were never run.
    """
    if not strategy_points or not baseline_points:
        raise ValueError("both strategy and baseline need at least one point")

    strategy_by_count = {p.label_count: p for p in strategy_points}
    baseline_by_count = {p.label_count: p for p in baseline_points}
    matched = sorted(set(strategy_by_count) & set(baseline_by_count))
    if not matched:
        raise ValueError(
            "no matched label counts between "
            f"{sorted(strategy_by_count)} and {sorted(baseline_by_count)}; "
            "label efficiency is only meaningful at equal label budgets"
        )

    strategy_scores = tuple(strategy_by_count[c].macro_f1 for c in matched)
    baseline_scores = tuple(baseline_by_count[c].macro_f1 for c in matched)

    return EfficiencyComparison(
        strategy=strategy_points[0].strategy,
        baseline=baseline_points[0].strategy,
        matched_counts=tuple(matched),
        strategy_scores=strategy_scores,
        baseline_scores=baseline_scores,
        mean_delta=float(np.mean(np.array(strategy_scores) - np.array(baseline_scores))),
        target=target,
        strategy_labels_to_target=labels_to_target(strategy_points, target),
        baseline_labels_to_target=labels_to_target(baseline_points, target),
    )


def curve_points(points: list[EfficiencyPoint]) -> list[dict[str, Any]]:
    """Serialize a curve for the report, ordered by label count."""
    return [
        {
            "strategy": p.strategy,
            "label_count": p.label_count,
            "macro_f1": p.macro_f1,
            "accuracy": p.accuracy,
            "rare_class_recall": p.rare_class_recall,
        }
        for p in sorted(points, key=lambda p: p.label_count)
    ]
