"""Round scheduling: which unlabeled wafers a reviewer sees next.

The scheduler combines uncertainty and diversity into a single greedy selection,
and it owns the invariants the rest of the system relies on:

* a batch never contains a wafer that already has a label,
* a batch never contains the same wafer twice,
* a batch is reproducible from ``(pool, strategy, seed)`` alone.

The greedy strategies are fully deterministic: ``seed`` governs only tie-breaking
among exactly equal scores and the random control arm. Two different seeds
therefore produce the *same* entropy or entropy-diversity batch, which is
intended. It does mean those strategies have no seed-to-seed variance, so the
error bars on the label efficiency curve come from the random arm and from
retraining, not from resampling a deterministic selector.

``SamplingStrategy.RANDOM`` is not a fallback. It is the control arm of the label
efficiency experiment, and it must be selected by exactly the same code path as
the real strategies so the comparison is not confounded by a difference in
plumbing.

Pure functions, no I/O.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from yieldloop.db.enums import SamplingStrategy
from yieldloop.sampling.diversity import (
    Embeddings,
    min_distance_to,
    validate_embeddings,
)
from yieldloop.sampling.uncertainty import (
    Probabilities,
    Scores,
    normalized_entropy,
    validate_probabilities,
)

#: Diversity contribution assigned to the first pick, when nothing has been
#: selected yet and every distance is infinite. Uniform across candidates, so
#: uncertainty alone decides the opening move.
_FIRST_PICK_DIVERSITY: Final[float] = 1.0


class SchedulingError(ValueError):
    """Raised when a selection request is internally inconsistent."""


@dataclass(frozen=True, slots=True)
class SelectionRequest:
    """A pool of candidates and the rules for drawing a batch from it."""

    wafer_ids: Sequence[str]
    probabilities: Probabilities
    embeddings: Embeddings
    #: Wafers that already carry a label. Excluded before scoring, so a labeled
    #: wafer cannot consume a slot in the batch.
    labeled_ids: frozenset[str]
    strategy: SamplingStrategy
    batch_size: int
    diversity_weight: float
    min_distance: float
    seed: int


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """The selected batch, with enough provenance to reproduce it."""

    wafer_ids: tuple[str, ...]
    scores: tuple[float, ...]
    strategy: SamplingStrategy
    seed: int
    #: Candidates offered, before exclusions.
    pool_size: int
    #: Candidates dropped because they were already labeled.
    excluded_labeled: int
    #: True when the batch is shorter than requested because the eligible pool
    #: ran out, rather than because of an error.
    truncated: bool

    def __len__(self) -> int:
        return len(self.wafer_ids)


def _validate_request(request: SelectionRequest) -> None:
    count = len(request.wafer_ids)
    if request.probabilities.shape[0] != count:
        raise SchedulingError(
            f"probabilities has {request.probabilities.shape[0]} rows but there are "
            f"{count} wafer ids"
        )
    if request.embeddings.shape[0] != count:
        raise SchedulingError(
            f"embeddings has {request.embeddings.shape[0]} rows but there are {count} wafer ids"
        )
    if len(set(request.wafer_ids)) != count:
        raise SchedulingError("wafer_ids contains duplicates")
    if request.batch_size < 0:
        raise SchedulingError(f"batch_size must be non-negative; got {request.batch_size}")
    if not 0.0 <= request.diversity_weight <= 1.0:
        raise SchedulingError(f"diversity_weight must be in [0, 1]; got {request.diversity_weight}")
    if request.min_distance < 0.0:
        raise SchedulingError(f"min_distance must be non-negative; got {request.min_distance}")


def _uncertainty_scores(strategy: SamplingStrategy, probabilities: Probabilities) -> Scores:
    """Score in ``[0, 1]``, higher meaning more informative."""
    match strategy:
        case SamplingStrategy.ENTROPY | SamplingStrategy.ENTROPY_DIVERSITY:
            return normalized_entropy(probabilities)
        case SamplingStrategy.DIVERSITY | SamplingStrategy.RANDOM:
            return np.zeros(probabilities.shape[0], dtype=np.float64)
    raise SchedulingError(f"unhandled strategy {strategy}")


def _effective_diversity_weight(strategy: SamplingStrategy, configured: float) -> float:
    """How much the distance term counts, given the strategy.

    The pure strategies pin the weight rather than honouring the configured
    value, so that an experiment comparing strategies is not silently also
    comparing weights.
    """
    match strategy:
        case SamplingStrategy.ENTROPY:
            return 0.0
        case SamplingStrategy.DIVERSITY:
            return 1.0
        case SamplingStrategy.ENTROPY_DIVERSITY:
            return configured
        case SamplingStrategy.RANDOM:
            return 0.0
    raise SchedulingError(f"unhandled strategy {strategy}")


def _random_batch(
    eligible_ids: Sequence[str], batch_size: int, seed: int
) -> tuple[list[int], list[float]]:
    """The control arm: a uniform sample without replacement."""
    rng = np.random.default_rng(seed)
    take = min(batch_size, len(eligible_ids))
    order = rng.permutation(len(eligible_ids))[:take]
    return [int(i) for i in order], [0.0] * take


def select_batch(request: SelectionRequest) -> SelectionResult:
    """Select the next batch of wafers for review.

    Returns fewer than ``batch_size`` wafers when the eligible pool is exhausted
    or ``min_distance`` rules out the remainder. That is a correct outcome: a
    short batch of genuinely informative wafers is worth more reviewer time than
    a full one padded with near-duplicates.
    """
    _validate_request(request)
    probabilities = validate_probabilities(request.probabilities)
    embeddings = validate_embeddings(request.embeddings)

    pool_size = len(request.wafer_ids)
    keep = [i for i, wafer in enumerate(request.wafer_ids) if wafer not in request.labeled_ids]
    excluded = pool_size - len(keep)

    if not keep or request.batch_size == 0:
        return SelectionResult(
            wafer_ids=(),
            scores=(),
            strategy=request.strategy,
            seed=request.seed,
            pool_size=pool_size,
            excluded_labeled=excluded,
            truncated=request.batch_size > 0,
        )

    index = np.asarray(keep, dtype=np.int64)
    eligible_ids = [request.wafer_ids[i] for i in keep]

    if request.strategy is SamplingStrategy.RANDOM:
        local, scores = _random_batch(eligible_ids, request.batch_size, request.seed)
    else:
        local, scores = _greedy_select(
            uncertainty=_uncertainty_scores(request.strategy, probabilities[index]),
            embeddings=embeddings[index],
            batch_size=request.batch_size,
            weight=_effective_diversity_weight(request.strategy, request.diversity_weight),
            min_distance=request.min_distance,
            seed=request.seed,
        )

    chosen = tuple(eligible_ids[i] for i in local)
    return SelectionResult(
        wafer_ids=chosen,
        scores=tuple(scores),
        strategy=request.strategy,
        seed=request.seed,
        pool_size=pool_size,
        excluded_labeled=excluded,
        truncated=len(chosen) < request.batch_size,
    )


def _greedy_select(
    *,
    uncertainty: Scores,
    embeddings: Embeddings,
    batch_size: int,
    weight: float,
    min_distance: float,
    seed: int,
) -> tuple[list[int], list[float]]:
    """Greedily maximize ``(1 - w) * uncertainty + w * distance-to-selected``.

    The distance term is rescaled against the current maximum at every step.
    Without that rescaling the two terms live on different scales -- uncertainty
    is bounded in ``[0, 1]`` while distance is not -- and the weight would not
    mean what it says.
    """
    count = embeddings.shape[0]
    rng = np.random.default_rng(seed)
    tie_break: Scores = np.asarray(rng.random(size=count), dtype=np.float64)

    selected: list[int] = []
    selected_scores: list[float] = []
    eligible = np.ones(count, dtype=bool)
    distances = np.full(count, np.inf, dtype=np.float64)

    for _ in range(min(batch_size, count)):
        if min_distance > 0.0 and selected:
            eligible &= distances >= min_distance
        if not eligible.any():
            break

        if not selected:
            diversity = np.full(count, _FIRST_PICK_DIVERSITY, dtype=np.float64)
        else:
            finite = distances[np.isfinite(distances)]
            scale = float(finite.max()) if finite.size and finite.max() > 0.0 else 1.0
            diversity = np.minimum(distances / scale, 1.0)

        combined = (1.0 - weight) * uncertainty + weight * diversity
        masked = np.where(eligible, combined, -np.inf)
        best = float(np.max(masked))
        close: npt.NDArray[np.bool_] = np.isclose(masked, best)
        tied = np.flatnonzero(eligible & close)
        pick = int(tied[np.argmin(tie_break[tied])])

        selected.append(pick)
        selected_scores.append(float(combined[pick]))
        eligible[pick] = False
        distances = np.minimum(distances, min_distance_to(embeddings, embeddings[pick : pick + 1]))

    return selected, selected_scores


def labeled_id_set(wafer_ids: Iterable[str]) -> frozenset[str]:
    """Build the exclusion set for a request."""
    return frozenset(wafer_ids)
