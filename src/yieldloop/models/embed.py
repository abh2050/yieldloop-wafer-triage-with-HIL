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

from yieldloop.db.enums import DefectPattern, SplitName
from yieldloop.db.models import Wafer
from yieldloop.models.classifier import CLASS_ORDER, WaferCNN, label_index


@dataclass(frozen=True, slots=True)
class WaferSample:
    """One wafer as the model sees it."""

    wafer_id: str
    grid: np.ndarray
    label: DefectPattern | None

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


def load_samples(
    session: Session,
    split: SplitName,
    *,
    labeled: bool = True,
    limit: int | None = None,
) -> list[WaferSample]:
    """Load wafers from one split, ordered deterministically by wafer id.

    Ordering explicitly rather than relying on the database's natural order keeps
    the data hash -- and therefore artifact identity -- stable across runs.

    Selects the four columns it needs rather than whole ORM entities, and streams
    them in batches. Loading 120,000 labeled wafers as mapped objects spends
    minutes constructing instances and identity-map entries that are discarded
    immediately, when only the grid bytes and the label are ever read.
    """
    statement = (
        select(
            Wafer.wafer_id,
            Wafer.grid,
            Wafer.grid_height,
            Wafer.grid_width,
            Wafer.dataset_label,
        )
        .where(
            Wafer.split == split,
            Wafer.dataset_label.is_not(None) if labeled else Wafer.dataset_label.is_(None),
        )
        .order_by(Wafer.wafer_id)
    )
    if limit is not None:
        statement = statement.limit(limit)

    return [
        WaferSample(
            wafer_id=wafer_id,
            # Copied out of the read-only buffer: torch cannot take a
            # non-writable array without warning, and the copy is cheap at 4 KiB.
            grid=np.frombuffer(grid, dtype=np.uint8).reshape(height, width).copy(),
            label=label,
        )
        for wafer_id, grid, height, width, label in session.execute(
            statement.execution_options(yield_per=FETCH_BATCH)
        )
    ]


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
