"""Golden sets, built from real labels and real decisions.

Two sources only. Defect labels come from WM811K's human annotations; reviewer
resolutions come from decisions this console actually captured. There is no
synthetic ground truth anywhere in the harness, which is why several suites
report "insufficient data" rather than a number when a console has not yet been
used -- an honest gap is more useful than a figure computed against invented
answers.

The holdout split is read here and nowhere else. Training and calibration touch
train and validation only, so a holdout number is a number from data the model
has provably never seen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from yieldloop.db.enums import SplitName
from yieldloop.db.models import Decision, Lot, ReviewTask, Wafer
from yieldloop.models.classifier import CLASS_ORDER, label_index
from yieldloop.models.embed import WaferSample, load_samples


@dataclass(frozen=True, slots=True)
class GoldenSet:
    """Wafers with real human labels, for classifier evaluation."""

    split: SplitName
    samples: tuple[WaferSample, ...]

    @property
    def labels(self) -> npt.NDArray[np.int64]:
        return np.array(
            [label_index(s.label) for s in self.samples if s.label is not None],
            dtype=np.int64,
        )

    @property
    def wafer_ids(self) -> tuple[str, ...]:
        return tuple(s.wafer_id for s in self.samples)

    def __len__(self) -> int:
        return len(self.samples)


@dataclass(frozen=True, slots=True)
class ReviewerResolution:
    """A real decision captured by the console."""

    wafer_id: str
    lot_name: str
    chosen_label: str
    model_label: str | None
    prediction_was_shown: bool
    is_override: bool
    decision_ms: int
    reason_code: str | None


def load_golden_set(
    session: Session, split: SplitName = SplitName.HOLDOUT, *, limit: int | None = None
) -> GoldenSet:
    """Load labeled wafers from a split.

    Defaults to holdout, which is the only split the harness should ever score
    the final numbers on.
    """
    samples = load_samples(session, split, labeled=True, limit=limit)
    return GoldenSet(split=split, samples=tuple(samples))


def load_reviewer_resolutions(session: Session) -> list[ReviewerResolution]:
    """Every decision a human has actually made in this console.

    Returns an empty list on a fresh install, which is a real state the suites
    must handle by reporting insufficient data rather than by fabricating any.
    """
    rows = session.execute(
        select(Decision, Wafer, Lot)
        .join(Wafer, Decision.wafer_id == Wafer.id)
        .join(Lot, Wafer.lot_id == Lot.id)
        .where(Decision.chosen_label.is_not(None))
        .order_by(Decision.created_at)
    ).all()
    return [
        ReviewerResolution(
            wafer_id=wafer.wafer_id,
            lot_name=lot.lot_name,
            chosen_label=decision.chosen_label.value if decision.chosen_label else "",
            model_label=decision.model_label.value if decision.model_label else None,
            prediction_was_shown=decision.prediction_was_shown,
            is_override=decision.is_override,
            decision_ms=decision.decision_ms,
            reason_code=decision.reason_code,
        )
        for decision, wafer, lot in rows
    ]


def label_counts(session: Session, split: SplitName) -> dict[str, int]:
    """Real label distribution in a split."""
    rows = session.execute(
        select(Wafer.dataset_label, func.count())
        .where(Wafer.split == split, Wafer.dataset_label.is_not(None))
        .group_by(Wafer.dataset_label)
    ).all()
    counts = dict.fromkeys((c.value for c in CLASS_ORDER), 0)
    for label, count in rows:
        if label is not None:
            counts[label.value] = int(count)
    return counts


def rare_classes(session: Session, split: SplitName, *, threshold: float = 0.01) -> list[str]:
    """Classes below ``threshold`` of the labeled population.

    Active learning should help most here, and a random sampler struggles to find
    examples of them at all, so they are tracked separately in the efficiency
    curve.
    """
    counts = label_counts(session, split)
    total = sum(counts.values())
    if total == 0:
        return []
    return [label for label, count in counts.items() if count / total < threshold]


def decision_count(session: Session) -> int:
    return int(session.execute(select(func.count()).select_from(Decision)).scalar_one())


def completed_task_count(session: Session) -> int:
    return int(session.execute(select(func.count()).select_from(ReviewTask)).scalar_one())
