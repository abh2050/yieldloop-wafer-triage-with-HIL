"""Embedding-diversity selection.

Uncertainty alone selects redundantly. The most ambiguous wafers in a pool tend
to look alike, so an entropy-only batch can spend an entire review round on
near-duplicates of one pattern and teach the model almost nothing. This module
supplies the counterweight: greedy max-min (k-center) selection, which repeatedly
takes the candidate furthest from everything already chosen.

Pure functions, no I/O.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

Embeddings = npt.NDArray[np.float64]
Scores = npt.NDArray[np.float64]

#: Vectors shorter than this are treated as degenerate rather than normalized.
_NORM_EPSILON: Final[float] = 1e-12


class EmbeddingError(ValueError):
    """Raised when an embedding matrix is malformed."""


def validate_embeddings(embeddings: npt.ArrayLike) -> Embeddings:
    """Coerce and validate an ``(n, d)`` embedding matrix."""
    array = np.asarray(embeddings, dtype=np.float64)
    if array.ndim != 2:
        raise EmbeddingError(f"expected a 2-D (n, d) array; got shape {array.shape}")
    if array.shape[0] > 0 and array.shape[1] == 0:
        raise EmbeddingError("embeddings must have at least one dimension")
    if not np.isfinite(array).all():
        raise EmbeddingError("embeddings contain NaN or infinity")
    return array


def l2_normalize(embeddings: npt.ArrayLike) -> Embeddings:
    """Scale each row to unit length.

    Zero-length rows are left as zeros rather than divided by zero. They remain
    valid candidates; they simply sit at the origin and are maximally distant
    from nothing in particular.
    """
    array = validate_embeddings(embeddings)
    if array.shape[0] == 0:
        return array
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe = np.where(norms < _NORM_EPSILON, 1.0, norms)
    return array / safe


def pairwise_distances(points: Embeddings, reference: Embeddings) -> Embeddings:
    """Euclidean distances between every point and every reference, ``(n, m)``."""
    if points.shape[0] == 0 or reference.shape[0] == 0:
        return np.zeros((points.shape[0], reference.shape[0]), dtype=np.float64)
    if points.shape[1] != reference.shape[1]:
        raise EmbeddingError(
            f"dimension mismatch: points are {points.shape[1]}-D, "
            f"reference is {reference.shape[1]}-D"
        )
    # Computed from the expansion rather than by materializing an (n, m, d)
    # difference tensor, which would not fit for a full unlabeled pool.
    point_sq = np.einsum("ij,ij->i", points, points)[:, None]
    reference_sq = np.einsum("ij,ij->i", reference, reference)[None, :]
    cross = points @ reference.T
    squared = np.maximum(point_sq + reference_sq - 2.0 * cross, 0.0)
    distances: Embeddings = np.sqrt(squared)
    return distances


def min_distance_to(points: Embeddings, reference: Embeddings) -> Scores:
    """Distance from each point to its nearest reference point.

    With no reference points every distance is infinite: nothing has been chosen
    yet, so no candidate is crowded by anything.
    """
    if reference.shape[0] == 0:
        return np.full(points.shape[0], np.inf, dtype=np.float64)
    return pairwise_distances(points, reference).min(axis=1)


def greedy_max_min(
    embeddings: npt.ArrayLike,
    batch_size: int,
    *,
    min_distance: float = 0.0,
    already_selected: npt.ArrayLike | None = None,
    seed: int = 0,
) -> list[int]:
    """Select up to ``batch_size`` indices by greedy max-min (k-center) coverage.

    Args:
        embeddings: ``(n, d)`` candidate embeddings.
        batch_size: Maximum number to select.
        min_distance: Candidates closer than this to an already-chosen point are
            never selected. This is what prevents near-duplicate wafers from
            occupying two slots in one review round. Zero disables the rule.
        already_selected: ``(m, d)`` embeddings of points chosen earlier, so a
            round can extend a previous selection without re-choosing near it.
        seed: Breaks ties among equidistant candidates deterministically, so a
            batch is reproducible from the seed alone.

    Returns:
        Indices into ``embeddings``, in selection order. May be shorter than
        ``batch_size`` when ``min_distance`` exhausts the eligible candidates --
        returning fewer real choices is correct, and padding the batch with
        near-duplicates to hit a number would defeat the purpose.
    """
    array = validate_embeddings(embeddings)
    candidate_count = array.shape[0]
    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative; got {batch_size}")
    if min_distance < 0.0:
        raise ValueError(f"min_distance must be non-negative; got {min_distance}")
    if candidate_count == 0 or batch_size == 0:
        return []

    reference = (
        validate_embeddings(already_selected)
        if already_selected is not None
        else np.zeros((0, array.shape[1]), dtype=np.float64)
    )
    if reference.shape[0] and reference.shape[1] != array.shape[1]:
        raise EmbeddingError(
            f"already_selected is {reference.shape[1]}-D but candidates are {array.shape[1]}-D"
        )

    rng = np.random.default_rng(seed)
    # A fixed random key per candidate resolves ties the same way every run.
    # np.random.Generator.random is typed as returning a float in its scalar
    # overload; the size argument makes it an array.
    tie_break: Scores = np.asarray(rng.random(size=candidate_count), dtype=np.float64)

    distances = min_distance_to(array, reference)
    chosen: list[int] = []
    eligible = np.ones(candidate_count, dtype=bool)

    for _ in range(min(batch_size, candidate_count)):
        if min_distance > 0.0:
            eligible &= (distances >= min_distance) | np.isinf(distances)
        if not eligible.any():
            break

        masked = np.where(eligible, distances, -np.inf)
        best = float(np.max(masked))
        # np.isclose rather than equality: the distance expansion above is not
        # exact, and two genuinely equidistant candidates must tie-break by key
        # rather than by floating point noise.
        close: npt.NDArray[np.bool_] = np.isclose(masked, best)
        tied = np.flatnonzero(eligible & close)
        pick = int(tied[np.argmin(tie_break[tied])])

        chosen.append(pick)
        eligible[pick] = False
        distances = np.minimum(distances, min_distance_to(array, array[pick : pick + 1]))

    return chosen
