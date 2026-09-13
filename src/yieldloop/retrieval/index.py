"""Building and persisting the FAISS index.

The index is what makes the grounding gate meaningful. Without retrieval the
agent would have nothing to cite, and a grounding rule over an empty evidence set
is just a guarantee that the system always abstains. With it, the set of citable
evidence is a concrete, inspectable list that the gate checks against.

Embeddings are L2-normalized by the classifier, so inner product is cosine
similarity and ``IndexFlatIP`` gives exact search. Exact rather than approximate
is a deliberate choice at this corpus size: an approximate index would introduce
recall loss that is indistinguishable, from the outside, from the agent failing
to find evidence -- and those two need to stay distinguishable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import numpy.typing as npt

from yieldloop.logging import get_logger

logger = get_logger(__name__)

Embeddings = npt.NDArray[np.float32]


class RetrievalIndexError(RuntimeError):
    """Raised when an index is malformed or inconsistent with its labels."""


@dataclass(frozen=True, slots=True)
class IndexManifest:
    """What is in an index, stored beside it.

    FAISS stores vectors but not what they refer to. The manifest carries the
    identifier for each row, so a search result can be resolved back to a lot,
    and the dimension and count, so a mismatched index fails loudly at load
    rather than returning neighbours from a stale corpus.
    """

    identifiers: tuple[str, ...]
    dimension: int
    #: Content hash of the embedder artifact that produced these vectors.
    artifact_hash: str

    def as_dict(self) -> dict[str, object]:
        return {
            "identifiers": list(self.identifiers),
            "dimension": self.dimension,
            "artifact_hash": self.artifact_hash,
            "count": len(self.identifiers),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IndexManifest:
        raw_identifiers = payload["identifiers"]
        if not isinstance(raw_identifiers, list):
            raise RetrievalIndexError(
                f"manifest identifiers must be a list; got {type(raw_identifiers).__name__}"
            )
        return cls(
            identifiers=tuple(str(i) for i in raw_identifiers),
            dimension=int(payload["dimension"]),
            artifact_hash=str(payload["artifact_hash"]),
        )


def normalize(vectors: npt.ArrayLike) -> Embeddings:
    """Coerce to float32 and L2-normalize rows.

    Applied defensively even though the classifier already normalizes: an index
    built from vectors that are not unit length silently stops being a cosine
    index, and the failure mode is subtly wrong neighbours rather than an error.
    """
    array = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    if array.ndim != 2:
        raise RetrievalIndexError(f"expected a 2-D (n, d) array; got shape {array.shape}")
    if not np.isfinite(array).all():
        raise RetrievalIndexError("embeddings contain NaN or infinity")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe = np.where(norms < 1e-12, 1.0, norms)
    return np.ascontiguousarray(array / safe, dtype=np.float32)


def build_index(
    vectors: npt.ArrayLike, identifiers: list[str], artifact_hash: str
) -> tuple[faiss.Index, IndexManifest]:
    """Build an exact inner-product index over normalized vectors."""
    array = normalize(vectors)
    if array.shape[0] != len(identifiers):
        raise RetrievalIndexError(
            f"{array.shape[0]} vectors against {len(identifiers)} identifiers"
        )
    if len(set(identifiers)) != len(identifiers):
        raise RetrievalIndexError("identifiers contain duplicates")

    index = faiss.IndexFlatIP(array.shape[1])
    if array.shape[0]:
        index.add(array)
    manifest = IndexManifest(
        identifiers=tuple(identifiers),
        dimension=int(array.shape[1]),
        artifact_hash=artifact_hash,
    )
    logger.info(
        "index_built",
        vectors=int(array.shape[0]),
        dimension=manifest.dimension,
        artifact_hash=artifact_hash[:12],
    )
    return index, manifest


def manifest_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".manifest.json")


def save_index(index: faiss.Index, manifest: IndexManifest, path: Path) -> None:
    """Write the index and its manifest together."""
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    manifest_path(path).write_text(json.dumps(manifest.as_dict(), indent=2, sort_keys=True))
    logger.info("index_saved", path=str(path), vectors=len(manifest.identifiers))


def load_index(path: Path) -> tuple[faiss.Index, IndexManifest]:
    """Load an index and verify it agrees with its manifest."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; build it with scripts/seed_from_real_labels.py"
        )
    manifest_file = manifest_path(path)
    if not manifest_file.is_file():
        raise RetrievalIndexError(
            f"{path} has no manifest at {manifest_file}; a FAISS index without one "
            "cannot resolve results back to lots"
        )

    index = faiss.read_index(str(path))
    manifest = IndexManifest.from_dict(json.loads(manifest_file.read_text()))

    if index.ntotal != len(manifest.identifiers):
        raise RetrievalIndexError(
            f"index holds {index.ntotal} vectors but the manifest lists "
            f"{len(manifest.identifiers)} identifiers; they are out of sync"
        )
    if index.d != manifest.dimension:
        raise RetrievalIndexError(
            f"index dimension {index.d} does not match manifest {manifest.dimension}"
        )
    return index, manifest
