"""Reading wafers out of the database and turning them into tensors.

The dataset classes here read grids straight from Postgres rather than from a
cached tensor file. Wafer grids are the ground truth the console renders and the
classifier consumes, and keeping one copy means a reviewer and the model can
never be looking at different pixels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sqlalchemy import Select, select
from sqlalchemy.orm import Session
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from yieldloop.db.enums import DefectPattern, LabelSource, SplitName
from yieldloop.db.models import Decision, Wafer
from yieldloop.models.classifier import CLASS_ORDER, WaferCNN, label_index


@dataclass(frozen=True, slots=True)
class WaferSample:
    """One wafer as the model sees it."""

    wafer_id: str
    grid: np.ndarray
    label: DefectPattern | None
    #: Where the label came from. Carried into the artifact data hash, so a model
    #: trained partly on reviewer labels is distinguishable from one trained only
    #: on the dataset's own annotations.
    source: LabelSource = LabelSource.DATASET

    @property
    def target(self) -> int:
        """Class index, or -1 when unlabeled."""
        return label_index(self.label) if self.label is not None else -1


def labeled_query(split: SplitName) -> Select[tuple[Wafer]]:
    """Wafers in ``split`` that carry a human label."""
    return select(Wafer).where(Wafer.split == split, Wafer.dataset_label.is_not(None))


def unlabeled_query(split: SplitName) -> Select[tuple[Wafer]]:
    """Wafers in ``split`` with no label. The active learning pool."""
    return select(Wafer).where(Wafer.split == split, Wafer.dataset_label.is_(None))


#: Rows fetched per round trip when streaming wafers out of the database.
FETCH_BATCH = 5_000


def reviewer_labels(session: Session, split: SplitName) -> dict[str, DefectPattern]:
    """The most recent reviewer label for each wafer in ``split``.

    This is what closes the human loop. A decision captured by the console is
    only training signal if the next round actually reads it, and reading it here
    -- rather than in a separate ingest step -- means every consumer of
    ``load_samples`` gets the corrected labels without having to know they exist.

    Scoped to one split deliberately. A reviewer decision on a holdout wafer must
    never reach training, and filtering at the query rather than trusting callers
    is what makes that hold.

    Later decisions win. A reviewer who revisits a wafer is correcting their
    earlier judgement, not casting a second vote.
    """
    # Selects the wafer identifier through the join rather than resolving it
    # from a separate map. The map version walked every wafer in the split --
    # 567,614 rows to resolve a handful of decisions -- and halved bulk load
    # throughput, which is on the critical path of every training run and round.
    latest: dict[str, DefectPattern] = {}
    for name, label in session.execute(
        select(Wafer.wafer_id, Decision.chosen_label)
        .join(Decision, Decision.wafer_id == Wafer.id)
        .where(Wafer.split == split, Decision.chosen_label.is_not(None))
        .order_by(Decision.created_at)
    ).all():
        if label is not None:
            latest[name] = label
    return latest


def load_samples(
    session: Session,
    split: SplitName,
    *,
    labeled: bool = True,
    limit: int | None = None,
    include_reviewer_labels: bool = True,
) -> list[WaferSample]:
    """Load wafers from one split, ordered deterministically by wafer id.

    Ordering explicitly rather than relying on the database's natural order keeps
    the data hash -- and therefore artifact identity -- stable across runs.

    Selects the four columns it needs rather than whole ORM entities, and streams
    them in batches. Loading 120,000 labeled wafers as mapped objects spends
    minutes constructing instances and identity-map entries that are discarded
    immediately, when only the grid bytes and the label are ever read.

    With ``include_reviewer_labels`` the human loop is closed: a reviewer label
    overrides the dataset's own annotation for that wafer, and a reviewer label
    on a previously-unlabeled wafer makes it a training example. The reviewer
    wins because theirs is the more recent human judgement, made against the same
    wafer map with the model's context available -- and because a console whose
    corrections are recorded and then discarded is not a human loop at all.

    Set it to False to reproduce a run from before any decisions existed.
    """
    corrections = reviewer_labels(session, split) if include_reviewer_labels else {}

    def _columns() -> Select[tuple[str, bytes, int, int, DefectPattern | None]]:
        return select(
            Wafer.wafer_id,
            Wafer.grid,
            Wafer.grid_height,
            Wafer.grid_width,
            Wafer.dataset_label,
        ).where(Wafer.split == split)

    # The label filter stays in the database. Dropping it to merge corrections in
    # Python meant scanning the whole split -- 567,614 rows of 4 KiB grids to
    # find 20,000 labeled ones -- and cost 15x throughput on a path that every
    # training run and every active round depends on. Corrections are applied as
    # a delta instead, so they cost in proportion to how many there are.
    base = _columns().where(
        Wafer.dataset_label.is_not(None) if labeled else Wafer.dataset_label.is_(None)
    )

    samples: list[WaferSample] = []
    for wafer_id, grid, height, width, dataset_label in session.execute(
        base.order_by(Wafer.wafer_id).execution_options(yield_per=FETCH_BATCH)
    ):
        reviewer = corrections.get(wafer_id)
        if labeled:
            label = reviewer if reviewer is not None else dataset_label
            source = LabelSource.REVIEWER if reviewer is not None else LabelSource.DATASET
        else:
            # A reviewer-labeled wafer has left the unlabeled pool and must not
            # be offered for labeling again.
            if reviewer is not None:
                continue
            label, source = None, LabelSource.DATASET

        samples.append(
            WaferSample(
                wafer_id=wafer_id,
                # Copied out of the read-only buffer: torch cannot take a
                # non-writable array without warning, and the copy is cheap.
                grid=np.frombuffer(grid, dtype=np.uint8).reshape(height, width).copy(),
                label=label,
                source=source,
            )
        )

    if labeled and corrections:
        # Wafers the console labeled that the dataset never did. These are new
        # training examples and the base query cannot see them.
        seen = {sample.wafer_id for sample in samples}
        newly = [wafer_id for wafer_id in corrections if wafer_id not in seen]
        if newly:
            for wafer_id, grid, height, width, _dataset_label in session.execute(
                _columns().where(Wafer.wafer_id.in_(newly))
            ):
                samples.append(
                    WaferSample(
                        wafer_id=wafer_id,
                        grid=np.frombuffer(grid, dtype=np.uint8).reshape(height, width).copy(),
                        label=corrections[wafer_id],
                        source=LabelSource.REVIEWER,
                    )
                )
            # Re-sorted so the ordering -- and therefore the data hash -- does
            # not depend on when a wafer happened to be labeled.
            samples.sort(key=lambda sample: sample.wafer_id)

    return samples[:limit] if limit is not None else samples


class WaferDataset(Dataset[tuple[Tensor, Tensor]]):
    """Torch dataset over in-memory wafer samples."""

    def __init__(self, samples: Sequence[WaferSample]) -> None:
        self._samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self._samples[index]
        grid = torch.from_numpy(np.ascontiguousarray(sample.grid))
        return grid, torch.tensor(sample.target, dtype=torch.long)

    @property
    def samples(self) -> list[WaferSample]:
        return self._samples

    def wafer_ids(self) -> list[str]:
        return [sample.wafer_id for sample in self._samples]

    def label_pairs(self) -> list[tuple[str, DefectPattern | None]]:
        """The ordered pairs the artifact data hash is computed over."""
        return [(sample.wafer_id, sample.label) for sample in self._samples]

    def reviewer_label_count(self) -> int:
        """How many labels in this set came from the console rather than WM811K.

        Recorded on the artifact so a model trained partly on human corrections
        is distinguishable from one trained only on the dataset.
        """
        return sum(1 for s in self._samples if s.source is LabelSource.REVIEWER)

    def class_counts(self) -> dict[DefectPattern, int]:
        counts = dict.fromkeys(CLASS_ORDER, 0)
        for sample in self._samples:
            if sample.label is not None:
                counts[sample.label] += 1
        return counts


def build_loader(
    dataset: WaferDataset, *, batch_size: int, shuffle: bool, seed: int
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Build a reproducible dataloader.

    Seeded explicitly so a training run is reproducible from its recorded seed,
    which is what makes the registry's reproducibility claim true.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        drop_last=False,
    )


@torch.no_grad()
def compute_logits(
    model: WaferCNN, loader: DataLoader[tuple[Tensor, Tensor]], device: torch.device
) -> tuple[Tensor, Tensor]:
    """Run the model over a loader, returning ``(logits, labels)``."""
    model.eval()
    logit_batches: list[Tensor] = []
    label_batches: list[Tensor] = []
    for grids, labels in loader:
        logits, _ = model(grids.to(device))
        logit_batches.append(logits.cpu())
        label_batches.append(labels)
    if not logit_batches:
        return torch.zeros(0, len(CLASS_ORDER)), torch.zeros(0, dtype=torch.long)
    return torch.cat(logit_batches), torch.cat(label_batches)


@torch.no_grad()
def compute_embeddings(
    model: WaferCNN, loader: DataLoader[tuple[Tensor, Tensor]], device: torch.device
) -> Tensor:
    """Embed every wafer in a loader. Feeds diversity sampling and retrieval."""
    model.eval()
    batches: list[Tensor] = []
    for grids, _ in loader:
        batches.append(model.embed(grids.to(device)).cpu())
    if not batches:
        return torch.zeros(0, model.config.embedding_dim)
    return torch.cat(batches)


def encode_embedding(vector: Tensor) -> bytes:
    """Serialize one embedding for the ``predictions.embedding`` column."""
    return vector.detach().cpu().numpy().astype(np.float32).tobytes(order="C")


def decode_embedding(payload: bytes, dim: int) -> np.ndarray:
    """Inverse of :func:`encode_embedding`."""
    expected = dim * 4
    if len(payload) != expected:
        raise ValueError(f"embedding payload is {len(payload)} bytes, expected {expected}")
    return np.frombuffer(payload, dtype=np.float32)


def select_device() -> torch.device:
    """Pick the best available device.

    Apple Silicon exposes MPS, which is a large speedup for this model on the
    hardware this is developed on.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
