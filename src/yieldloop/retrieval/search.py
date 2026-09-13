"""Querying the retrieval index.

Search returns evidence, not answers. Every hit carries the identifier the agent
is allowed to cite and the similarity that justified retrieving it, and the
caller turns those into a context bundle. Nothing here decides what the evidence
means.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
import numpy.typing as npt

from yieldloop.retrieval.index import IndexManifest, normalize


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieved neighbour."""

    identifier: str
    #: Cosine similarity in ``[-1, 1]``; for non-negative embeddings, ``[0, 1]``.
    similarity: float
    rank: int


def search(
    index: faiss.Index,
    manifest: IndexManifest,
    query: npt.ArrayLike,
    *,
    top_k: int,
    exclude: frozenset[str] = frozenset(),
    min_similarity: float = 0.0,
) -> list[Hit]:
    """Return the nearest neighbours of one query vector.

    Args:
        exclude: Identifiers to drop from the results. The lot under review is
            always its own nearest neighbour, and returning it as precedent for
            itself would be circular.
        min_similarity: Hits below this are discarded. A weak neighbour is worse
            than no neighbour: it becomes citable evidence that looks like
            support while being noise.

    Returns fewer than ``top_k`` hits when the filters remove them, which is a
    correct outcome that the abstention path is designed to handle.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be positive; got {top_k}")
    if index.ntotal == 0:
        return []

    vector = normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))
    if vector.shape[1] != manifest.dimension:
        raise ValueError(
            f"query is {vector.shape[1]}-D but the index is {manifest.dimension}-D"
        )

    # Over-fetch so that exclusions cannot starve the result below top_k.
    fetch = min(index.ntotal, top_k + len(exclude) + 1)
    similarities, positions = index.search(vector, fetch)

    hits: list[Hit] = []
    for similarity, position in zip(similarities[0], positions[0], strict=True):
        if position < 0:
            continue
        identifier = manifest.identifiers[int(position)]
        if identifier in exclude:
            continue
        if float(similarity) < min_similarity:
            continue
        hits.append(
            Hit(identifier=identifier, similarity=float(similarity), rank=len(hits) + 1)
        )
        if len(hits) >= top_k:
            break
    return hits


def search_many(
    index: faiss.Index,
    manifest: IndexManifest,
    queries: npt.ArrayLike,
    *,
    top_k: int,
    min_similarity: float = 0.0,
) -> list[list[Hit]]:
    """Batch search. Returns one hit list per query row."""
    array = normalize(queries)
    return [
        search(index, manifest, row, top_k=top_k, min_similarity=min_similarity)
        for row in array
    ]


def recall_at_k(
    retrieved: list[list[Hit]], relevant: list[frozenset[str]], k: int
) -> float:
    """Fraction of queries whose top-k contains at least one relevant item.

    Reported by the eval harness: if retrieval recall is poor, the agent's
    abstention rate rises for reasons that have nothing to do with the agent.
    """
    if not retrieved:
        return 0.0
    if len(retrieved) != len(relevant):
        raise ValueError(
            f"{len(retrieved)} result lists against {len(relevant)} relevance sets"
        )
    found = sum(
        1
        for hits, targets in zip(retrieved, relevant, strict=True)
        if targets and {hit.identifier for hit in hits[:k]} & targets
    )
    return found / len(retrieved)
