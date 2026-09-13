"""Retrieval against the real index and the real excursion corpus.

Two things are asserted here. That search is *exact* -- the results match a
brute-force computation over the same vectors, which is what `IndexFlatIP`
guarantees and what the numpy implementation must preserve. And that retrieval
finds real precedent: lots sharing a defect signature should retrieve each other,
because if they do not, the agent abstains for reasons that have nothing to do
with the agent.

Runs against whatever the seeding script actually built. Skips when the index is
absent rather than constructing a fake one: an index of invented centroids would
measure nothing.
"""

from __future__ import annotations

from itertools import pairwise

import faiss
import numpy as np
import pytest
import torch
from sqlalchemy import select
from sqlalchemy.orm import Session

from yieldloop.config import Settings
from yieldloop.db.models import HistoricalExcursion
from yieldloop.db.session import build_engine
from yieldloop.retrieval.index import (
    IndexManifest,
    RetrievalIndexError,
    build_index,
    load_index,
    materialize,
    normalize,
)
from yieldloop.retrieval.search import recall_at_k, search, search_many

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def live_index() -> tuple[faiss.Index, IndexManifest]:
    settings = Settings()
    try:
        return load_index(settings.faiss_index_path)
    except (FileNotFoundError, RetrievalIndexError) as exc:
        pytest.skip(f"no retrieval index: {exc}")


# --- exactness ------------------------------------------------------------


def test_search_is_exact_against_brute_force() -> None:
    """The numpy path must return precisely what IndexFlatIP would.

    Exact search is why an empty result can be trusted to mean "no precedent"
    rather than "the approximate index missed it", which is the distinction the
    abstention path depends on.
    """
    rng = np.random.default_rng(11)
    vectors = rng.normal(0, 1, (500, 64))
    identifiers = [f"hx:lot{i:05d}" for i in range(500)]
    index, manifest = build_index(vectors, identifiers, "a" * 64)

    matrix = materialize(index)
    for query_index in (0, 7, 199, 499):
        query = matrix[query_index]
        similarities = (matrix @ normalize(query.reshape(1, -1)).T).ravel()
        expected = [identifiers[i] for i in np.argsort(-similarities)[:8]]
        actual = [hit.identifier for hit in search(index, manifest, query, top_k=8)]
        assert actual == expected


def test_search_runs_in_a_process_that_has_used_torch() -> None:
    """faiss and torch each ship an OpenMP runtime and the second to initialize
    aborts the process. Only `search` spawns OpenMP threads, which is why it runs
    in numpy. This asserts the two genuinely coexist, since the API, the eval
    harness and the seeding script all use both."""
    torch.nn.Linear(4, 4)(torch.randn(2, 4))

    rng = np.random.default_rng(5)
    vectors = rng.normal(0, 1, (128, 32))
    identifiers = [f"hx:lot{i:04d}" for i in range(128)]
    index, manifest = build_index(vectors, identifiers, "b" * 64)
    assert len(search(index, manifest, vectors[3], top_k=5)) == 5


def test_a_vector_is_its_own_nearest_neighbour() -> None:
    rng = np.random.default_rng(2)
    vectors = rng.normal(0, 1, (200, 32))
    identifiers = [f"hx:lot{i:04d}" for i in range(200)]
    index, manifest = build_index(vectors, identifiers, "c" * 64)

    hits = search(index, manifest, vectors[42], top_k=3)
    assert hits[0].identifier == identifiers[42]
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)


def test_results_are_ordered_by_similarity() -> None:
    rng = np.random.default_rng(8)
    vectors = rng.normal(0, 1, (300, 48))
    identifiers = [f"hx:lot{i:04d}" for i in range(300)]
    index, manifest = build_index(vectors, identifiers, "d" * 64)

    hits = search(index, manifest, vectors[10], top_k=20)
    assert [h.rank for h in hits] == list(range(1, 21))
    assert all(a.similarity >= b.similarity for a, b in pairwise(hits))


def test_excluding_the_query_lot_still_fills_the_batch() -> None:
    """A lot is its own nearest neighbour, and citing itself as precedent would
    be circular."""
    rng = np.random.default_rng(4)
    vectors = rng.normal(0, 1, (100, 32))
    identifiers = [f"hx:lot{i:04d}" for i in range(100)]
    index, manifest = build_index(vectors, identifiers, "e" * 64)

    hits = search(index, manifest, vectors[9], top_k=5, exclude=frozenset({identifiers[9]}))
    assert len(hits) == 5
    assert identifiers[9] not in {h.identifier for h in hits}


def test_a_weak_neighbour_is_dropped_rather_than_returned() -> None:
    """A weak neighbour becomes citable evidence that looks like support."""
    rng = np.random.default_rng(6)
    vectors = rng.normal(0, 1, (200, 64))
    identifiers = [f"hx:lot{i:04d}" for i in range(200)]
    index, manifest = build_index(vectors, identifiers, "f" * 64)

    strict = search(index, manifest, vectors[1], top_k=20, min_similarity=0.99)
    assert len(strict) < 20
    assert all(hit.similarity >= 0.99 for hit in strict)


# --- against the real corpus ---------------------------------------------


def test_the_real_index_matches_the_excursion_corpus(
    live_index: tuple[faiss.Index, IndexManifest],
) -> None:
    """Every indexed identifier must resolve to a real excursion row.

    An identifier in the index with no row behind it would be an evidence id the
    agent could cite and the console could not render.
    """
    _, manifest = live_index
    with Session(build_engine(Settings())) as session:
        stored = set(session.execute(select(HistoricalExcursion.evidence_id)).scalars().all())
    assert set(manifest.identifiers) <= stored


def test_real_lots_retrieve_real_precedent(live_index: tuple[faiss.Index, IndexManifest]) -> None:
    """Retrieval on the real corpus must return something for a real lot."""
    index, manifest = live_index
    settings = Settings()

    with Session(build_engine(settings)) as session:
        excursions = (
            session.execute(
                select(HistoricalExcursion)
                .where(HistoricalExcursion.centroid.is_not(None))
                .limit(20)
            )
            .scalars()
            .all()
        )
    if not excursions:
        pytest.skip("no excursions with centroids")

    queries = [
        np.frombuffer(row.centroid, dtype=np.float32)
        for row in excursions
        if row.centroid is not None
    ]
    results = search_many(index, manifest, np.vstack(queries), top_k=5)

    assert all(len(hits) > 0 for hits in results), "a real lot retrieved no precedent"
    # Each lot is its own nearest neighbour in the corpus it was indexed into.
    relevant = [frozenset({row.evidence_id}) for row in excursions]
    assert recall_at_k(results, relevant, 5) == pytest.approx(1.0)


def test_same_pattern_lots_are_closer_than_different_pattern_lots(
    live_index: tuple[faiss.Index, IndexManifest],
) -> None:
    """The embedding must actually encode the defect signature.

    If it did not, retrieval would return arbitrary precedent and every grounded
    hypothesis would be built on an irrelevant comparison.
    """
    index, manifest = live_index
    with Session(build_engine(Settings())) as session:
        rows = (
            session.execute(
                select(HistoricalExcursion).where(HistoricalExcursion.centroid.is_not(None))
            )
            .scalars()
            .all()
        )
    if len(rows) < 40:
        pytest.skip("corpus too small to compare patterns")

    by_id = {row.evidence_id: row for row in rows}
    same_rank_positions: list[float] = []
    for row in rows[:30]:
        if row.centroid is None:
            continue
        hits = search(
            index,
            manifest,
            np.frombuffer(row.centroid, dtype=np.float32),
            top_k=10,
            exclude=frozenset({row.evidence_id}),
        )
        neighbours = [by_id[h.identifier] for h in hits if h.identifier in by_id]
        if not neighbours:
            continue
        matching = sum(1 for n in neighbours if n.observed_pattern is row.observed_pattern)
        same_rank_positions.append(matching / len(neighbours))

    if not same_rank_positions:
        pytest.skip("no neighbours resolved")

    # Chance agreement would be roughly the share of the commonest pattern; a
    # useful embedding must beat that comfortably.
    assert float(np.mean(same_rank_positions)) > 0.5
