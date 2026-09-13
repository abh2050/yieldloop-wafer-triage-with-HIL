"""Content-addressed artifact registry.

Every artifact is stored under the sha256 of its own bytes and recorded with the
hash of the data it was fit on, the git commit that produced it, the seed, and
the full hyperparameter set. The point is that any number in the eval report can
be traced back to a training run that can be reproduced, and that two artifacts
claiming to be "the model" can be told apart.

The data hash covers the ordered ``(wafer_id, label)`` pairs the artifact was fit
on, not the file on disk. That is the thing that actually determines what was
learned: two runs over the same archive but different label sets are different
experiments, and the label efficiency curve depends on being able to distinguish
them.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from yieldloop.db.enums import ArtifactKind, DefectPattern
from yieldloop.db.models import ModelArtifact
from yieldloop.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GitState:
    """The commit an artifact was produced at."""

    commit: str
    dirty: bool

    @property
    def is_reproducible(self) -> bool:
        """False when the tree had uncommitted changes.

        An artifact trained from a dirty tree cannot be reproduced from its
        recorded commit. It is still stored -- refusing to train during
        development would be obstructive -- but the flag travels with it so a
        result can never be presented as reproducible when it is not.
        """
        return not self.dirty


def git_state(repo_root: Path | None = None) -> GitState:
    """Read the current commit and whether the tree is dirty."""
    root = repo_root or Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        # Not a git checkout, e.g. inside a container built from a tarball.
        return GitState(commit="0" * 40, dirty=True)
    return GitState(commit=commit, dirty=bool(status))


def hash_training_data(pairs: Iterable[tuple[str, DefectPattern | None]]) -> str:
    """Hash the ordered wafer/label pairs an artifact was fit on.

    Order matters and is not sorted away: two runs over the same wafers in a
    different order can differ, and the hash should say so.
    """
    digest = hashlib.sha256()
    for wafer_id, label in pairs:
        digest.update(wafer_id.encode())
        digest.update(b"\x1f")
        digest.update((label.value if label is not None else "").encode())
        digest.update(b"\x1e")
    return digest.hexdigest()


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """A written artifact and its database record."""

    path: Path
    record: ModelArtifact

    @property
    def content_hash(self) -> str:
        return self.record.content_hash


class ArtifactRegistry:
    """Writes artifacts to disk under their content hash and records them."""

    def __init__(self, session: Session, registry_dir: Path) -> None:
        self._session = session
        self._dir = registry_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, content_hash: str, kind: ArtifactKind) -> Path:
        return self._dir / f"{kind.value}-{content_hash}.pt"

    def save(
        self,
        *,
        kind: ArtifactKind,
        state: dict[str, Any],
        data_hash: str,
        seed: int,
        hyperparameters: dict[str, Any],
        metrics: dict[str, Any],
        label_count: int,
        parent_id: Any | None = None,
        activate: bool = False,
    ) -> StoredArtifact:
        """Serialize, hash, write, and record one artifact.

        Writing is content-addressed, so saving the same artifact twice is a
        no-op on disk rather than a second copy under a different name.
        """
        stream = io.BytesIO()
        torch.save(state, stream)
        payload = stream.getvalue()
        content_hash = hash_bytes(payload)

        target = self.path_for(content_hash, kind)
        if not target.exists():
            target.write_bytes(payload)

        existing = self._session.execute(
            select(ModelArtifact).where(ModelArtifact.content_hash == content_hash)
        ).scalar_one_or_none()
        if existing is not None:
            logger.info("artifact_already_registered", content_hash=content_hash[:12])
            if activate:
                self.activate(existing)
            return StoredArtifact(path=target, record=existing)

        state_of_git = git_state()
        record = ModelArtifact(
            kind=kind,
            content_hash=content_hash,
            data_hash=data_hash,
            git_commit=state_of_git.commit,
            git_dirty=state_of_git.dirty,
            seed=seed,
            hyperparameters=hyperparameters,
            metrics=metrics,
            label_count=label_count,
            parent_id=parent_id,
            is_active=False,
        )
        self._session.add(record)
        self._session.flush()
        if activate:
            self.activate(record)

        logger.info(
            "artifact_registered",
            kind=kind.value,
            content_hash=content_hash[:12],
            data_hash=data_hash[:12],
            git_commit=state_of_git.commit[:12],
            git_dirty=state_of_git.dirty,
            label_count=label_count,
        )
        return StoredArtifact(path=target, record=record)

    def activate(self, record: ModelArtifact) -> None:
        """Make ``record`` the active artifact of its kind.

        Exactly one artifact per kind is active, so "the current model" is a
        property of the database rather than of whichever file a script happened
        to load.
        """
        self._session.execute(
            update(ModelArtifact)
            .where(ModelArtifact.kind == record.kind)
            .values(is_active=False)
        )
        self._session.execute(
            update(ModelArtifact).where(ModelArtifact.id == record.id).values(is_active=True)
        )
        self._session.flush()

    def active(self, kind: ArtifactKind) -> ModelArtifact | None:
        return self._session.execute(
            select(ModelArtifact).where(
                ModelArtifact.kind == kind, ModelArtifact.is_active.is_(True)
            )
        ).scalar_one_or_none()

    def load_state(self, record: ModelArtifact) -> dict[str, Any]:
        """Read an artifact back, verifying its bytes still hash as recorded."""
        path = self.path_for(record.content_hash, record.kind)
        if not path.is_file():
            raise FileNotFoundError(
                f"artifact {record.content_hash[:12]} is registered but {path} is missing"
            )
        payload = path.read_bytes()
        actual = hash_bytes(payload)
        if actual != record.content_hash:
            raise ValueError(
                f"artifact at {path} hashes to {actual[:12]} but is registered as "
                f"{record.content_hash[:12]}; the file has changed since it was written"
            )
        state: dict[str, Any] = torch.load(io.BytesIO(payload), weights_only=False)
        return state

    def history(self, kind: ArtifactKind) -> Sequence[ModelArtifact]:
        """Every artifact of a kind, oldest first. The label efficiency x-axis."""
        return (
            self._session.execute(
                select(ModelArtifact)
                .where(ModelArtifact.kind == kind)
                .order_by(ModelArtifact.label_count, ModelArtifact.created_at)
            )
            .scalars()
            .all()
        )


def describe(record: ModelArtifact) -> str:
    """One-line provenance summary, for logs and the case study."""
    return json.dumps(
        {
            "kind": record.kind.value,
            "content_hash": record.content_hash[:12],
            "data_hash": record.data_hash[:12],
            "git_commit": record.git_commit[:12],
            "git_dirty": record.git_dirty,
            "seed": record.seed,
            "label_count": record.label_count,
        },
        sort_keys=True,
    )
