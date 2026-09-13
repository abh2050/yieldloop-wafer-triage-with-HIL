"""Property tests for active learning sample selection.

The invariants here are the ones the rest of the system leans on. A violation is
not a cosmetic bug: re-showing an already-labeled wafer wastes the scarce
resource the whole project exists to conserve, and a duplicate in a batch
silently shortens the round.

These exercise pure functions over generated inputs. No generated wafer data is
persisted; the arrays exist only to probe the function under test.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import HealthCheck, given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from yieldloop.db.enums import SamplingStrategy
from yieldloop.sampling.diversity import greedy_max_min, l2_normalize, min_distance_to
from yieldloop.sampling.scheduler import (
    SchedulingError,
    SelectionRequest,
    select_batch,
)
from yieldloop.sampling.uncertainty import (
    ProbabilityError,
    max_entropy,
    normalized_entropy,
    shannon_entropy,
    validate_probabilities,
)

CLASS_COUNT = 9
EMBED_DIM = 8

finite_floats = st.floats(
    min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False, width=64
)


def _softmax(logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


@st.composite
def pools(draw: st.DrawFn, min_size: int = 1, max_size: int = 40) -> dict[str, object]:
    """A candidate pool: ids, calibrated probabilities, and embeddings."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    logits = draw(arrays(np.float64, (size, CLASS_COUNT), elements=finite_floats))
    embeddings = draw(arrays(np.float64, (size, EMBED_DIM), elements=finite_floats))
    labeled_count = draw(st.integers(min_value=0, max_value=size))
    ids = [f"lot{i // 25:04d}-{i % 25}" for i in range(size)]
    return {
        "wafer_ids": ids,
        "probabilities": _softmax(logits),
        "embeddings": embeddings,
        "labeled_ids": frozenset(ids[:labeled_count]),
    }


def _request(pool: dict[str, object], **overrides: object) -> SelectionRequest:
    kwargs: dict[str, object] = {
        **pool,
        "strategy": SamplingStrategy.ENTROPY_DIVERSITY,
        "batch_size": 8,
        "diversity_weight": 0.5,
        "min_distance": 0.0,
        "seed": 17,
    }
    kwargs.update(overrides)
    return SelectionRequest(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Scheduler invariants
# ---------------------------------------------------------------------------


@hyp_settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(pool=pools(), strategy=st.sampled_from(SamplingStrategy))
def test_batch_never_contains_an_already_labeled_wafer(
    pool: dict[str, object], strategy: SamplingStrategy
) -> None:
    result = select_batch(_request(pool, strategy=strategy))
    labeled = pool["labeled_ids"]
    assert isinstance(labeled, frozenset)
    assert not set(result.wafer_ids) & labeled


@hyp_settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(pool=pools(), strategy=st.sampled_from(SamplingStrategy))
def test_batch_never_contains_duplicates(
    pool: dict[str, object], strategy: SamplingStrategy
) -> None:
    result = select_batch(_request(pool, strategy=strategy))
    assert len(set(result.wafer_ids)) == len(result.wafer_ids)


@hyp_settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(pool=pools(), batch_size=st.integers(min_value=0, max_value=60))
def test_batch_never_exceeds_the_requested_size(pool: dict[str, object], batch_size: int) -> None:
    result = select_batch(_request(pool, batch_size=batch_size))
    assert len(result) <= batch_size


@hyp_settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(pool=pools(), strategy=st.sampled_from(SamplingStrategy))
def test_every_selected_wafer_came_from_the_pool(
    pool: dict[str, object], strategy: SamplingStrategy
) -> None:
    result = select_batch(_request(pool, strategy=strategy))
    ids = pool["wafer_ids"]
    assert isinstance(ids, list)
    assert set(result.wafer_ids) <= set(ids)


@hyp_settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(pool=pools(), strategy=st.sampled_from(SamplingStrategy))
def test_selection_is_reproducible(pool: dict[str, object], strategy: SamplingStrategy) -> None:
    first = select_batch(_request(pool, strategy=strategy))
    second = select_batch(_request(pool, strategy=strategy))
    assert first.wafer_ids == second.wafer_ids


@hyp_settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(pool=pools())
def test_truncation_is_reported_accurately(pool: dict[str, object]) -> None:
    result = select_batch(_request(pool, batch_size=8))
    assert result.truncated == (len(result) < 8)


@hyp_settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(pool=pools())
def test_exclusion_count_is_accurate(pool: dict[str, object]) -> None:
    result = select_batch(_request(pool))
    ids = pool["wafer_ids"]
    labeled = pool["labeled_ids"]
    assert isinstance(ids, list)
    assert isinstance(labeled, frozenset)
    assert result.excluded_labeled == len([i for i in ids if i in labeled])
    assert result.pool_size == len(ids)


@hyp_settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(pool=pools(min_size=2))
def test_min_distance_is_respected_between_selected_wafers(
    pool: dict[str, object],
) -> None:
    """No two wafers in a batch sit closer than the configured tolerance."""
    embeddings = np.asarray(pool["embeddings"], dtype=np.float64)
    normalized = l2_normalize(embeddings)
    result = select_batch(
        _request(
            pool,
            embeddings=normalized,
            min_distance=0.25,
            strategy=SamplingStrategy.DIVERSITY,
        )
    )
    ids = pool["wafer_ids"]
    assert isinstance(ids, list)
    positions = [ids.index(w) for w in result.wafer_ids]
    for rank, position in enumerate(positions[1:], start=1):
        earlier = normalized[positions[:rank]]
        assert float(min_distance_to(normalized[position : position + 1], earlier)[0]) >= 0.25


def test_duplicate_wafer_ids_are_rejected() -> None:
    with pytest.raises(SchedulingError, match="duplicates"):
        select_batch(
            _request(
                {
                    "wafer_ids": ["lot0001-1", "lot0001-1"],
                    "probabilities": np.full((2, CLASS_COUNT), 1.0 / CLASS_COUNT),
                    "embeddings": np.zeros((2, EMBED_DIM)),
                    "labeled_ids": frozenset(),
                }
            )
        )


def test_mismatched_array_lengths_are_rejected() -> None:
    with pytest.raises(SchedulingError, match="rows"):
        select_batch(
            _request(
                {
                    "wafer_ids": ["lot0001-1", "lot0001-2"],
                    "probabilities": np.full((1, CLASS_COUNT), 1.0 / CLASS_COUNT),
                    "embeddings": np.zeros((2, EMBED_DIM)),
                    "labeled_ids": frozenset(),
                }
            )
        )


# ---------------------------------------------------------------------------
# Uncertainty invariants
# ---------------------------------------------------------------------------


@given(
    logits=arrays(
        np.float64, array_shapes(min_dims=2, max_dims=2, min_side=2), elements=finite_floats
    )
)
def test_entropy_is_bounded_by_the_uniform_distribution(
    logits: npt.NDArray[np.float64],
) -> None:
    probabilities = _softmax(logits)
    entropy = shannon_entropy(probabilities)
    assert (entropy >= -1e-12).all()
    assert (entropy <= max_entropy(probabilities.shape[1]) + 1e-9).all()


@given(
    logits=arrays(
        np.float64, array_shapes(min_dims=2, max_dims=2, min_side=2), elements=finite_floats
    )
)
def test_normalized_entropy_stays_in_the_unit_interval(
    logits: npt.NDArray[np.float64],
) -> None:
    values = normalized_entropy(_softmax(logits))
    assert (values >= -1e-12).all()
    assert (values <= 1.0 + 1e-9).all()


@given(
    logits=arrays(
        np.float64, array_shapes(min_dims=2, max_dims=2, min_side=2), elements=finite_floats
    ),
    permutation_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_entropy_ignores_class_order(
    logits: npt.NDArray[np.float64], permutation_seed: int
) -> None:
    """Entropy is a property of the distribution, not of the label ordering."""
    probabilities = _softmax(logits)
    rng = np.random.default_rng(permutation_seed)
    order = rng.permutation(probabilities.shape[1])
    assert np.allclose(shannon_entropy(probabilities), shannon_entropy(probabilities[:, order]))


def test_confident_prediction_has_exactly_zero_entropy() -> None:
    one_hot = np.zeros((1, CLASS_COUNT))
    one_hot[0, 4] = 1.0
    value = float(shannon_entropy(one_hot)[0])
    assert value == 0.0
    assert not np.signbit(value)


@given(
    probabilities=arrays(
        np.float64, (3, CLASS_COUNT), elements=st.floats(0.01, 1.0, allow_nan=False)
    )
)
def test_unnormalized_rows_are_rejected_rather_than_rescaled(
    probabilities: npt.NDArray[np.float64],
) -> None:
    """A caller passing raw scores has a bug; silently renormalizing hides it."""
    sums = probabilities.sum(axis=1)
    if np.allclose(sums, 1.0, atol=1e-6):
        return
    with pytest.raises(ProbabilityError, match=r"sum to 1\.0"):
        validate_probabilities(probabilities)


# ---------------------------------------------------------------------------
# Diversity invariants
# ---------------------------------------------------------------------------


@given(
    embeddings=arrays(np.float64, (25, EMBED_DIM), elements=finite_floats),
    batch_size=st.integers(min_value=0, max_value=30),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_greedy_selection_returns_unique_in_range_indices(
    embeddings: npt.NDArray[np.float64], batch_size: int, seed: int
) -> None:
    picked = greedy_max_min(embeddings, batch_size, seed=seed)
    assert len(set(picked)) == len(picked)
    assert len(picked) <= min(batch_size, embeddings.shape[0])
    assert all(0 <= i < embeddings.shape[0] for i in picked)


@given(
    embeddings=arrays(np.float64, (20, EMBED_DIM), elements=finite_floats),
    min_distance=st.floats(min_value=0.01, max_value=2.0),
)
def test_greedy_selection_never_returns_near_duplicates(
    embeddings: npt.NDArray[np.float64], min_distance: float
) -> None:
    normalized = l2_normalize(embeddings)
    picked = greedy_max_min(normalized, 10, min_distance=min_distance, seed=5)
    for rank, index in enumerate(picked[1:], start=1):
        earlier = normalized[picked[:rank]]
        distance = float(min_distance_to(normalized[index : index + 1], earlier)[0])
        assert distance >= min_distance - 1e-9


@given(embeddings=arrays(np.float64, (15, EMBED_DIM), elements=finite_floats))
def test_l2_normalize_produces_unit_or_zero_rows(
    embeddings: npt.NDArray[np.float64],
) -> None:
    norms = np.linalg.norm(l2_normalize(embeddings), axis=1)
    assert np.all((np.isclose(norms, 1.0)) | (np.isclose(norms, 0.0)))
