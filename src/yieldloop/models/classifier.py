"""The wafer map defect pattern classifier.

A small convolutional network, deliberately. The defect patterns in WM811K are
spatial signatures on a 64x64 grid -- rings, arcs, scratches, clusters -- and a
few convolutional blocks capture them. A larger backbone would raise the accuracy
ceiling slightly and would make the parts of this system that actually matter
(calibration, routing, the human loop) harder to iterate on.

The network exposes an embedding alongside its logits. The same vector drives
diversity sampling and retrieval, so the notion of "similar wafer" is identical
in both places rather than being two things that happen to share a name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import torch
from torch import Tensor, nn

from yieldloop.db.enums import DefectPattern

#: Class order is fixed by the enum, so a stored probability vector can always be
#: read back against the right labels regardless of training run.
CLASS_ORDER: Final[tuple[DefectPattern, ...]] = tuple(DefectPattern)
NUM_CLASSES: Final[int] = len(CLASS_ORDER)

#: WM811K die values are categorical, not ordinal: 2 (failing) is not "twice" 1
#: (passing), and feeding the raw integers would impose an ordering the data does
#: not have. Maps are one-hot encoded into three channels instead.
INPUT_CHANNELS: Final[int] = 3


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    """Architecture and training hyperparameters, recorded with every artifact."""

    grid_height: int = 64
    grid_width: int = 64
    embedding_dim: int = 128
    channels: tuple[int, ...] = (32, 64, 128)
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 30
    early_stopping_patience: int = 5
    #: Rare classes are a fraction of a percent of the data. Without reweighting,
    #: the loss is minimized by never predicting them at all.
    class_weighting: bool = True
    seed: int = 20260913

    def as_dict(self) -> dict[str, object]:
        return {
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "embedding_dim": self.embedding_dim,
            "channels": list(self.channels),
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "class_weighting": self.class_weighting,
            "seed": self.seed,
            "architecture": "WaferCNN",
            "num_classes": NUM_CLASSES,
            "input_channels": INPUT_CHANNELS,
        }


def one_hot_maps(grids: Tensor) -> Tensor:
    """Encode ``(n, h, w)`` uint8 maps as ``(n, 3, h, w)`` float channels.

    Channel 0 is outside the wafer, 1 is passing die, 2 is failing die. Keeping
    the "outside" channel explicit matters: wafer edges differ in shape between
    lots, and the network needs to distinguish "no die here" from "a die that
    passed".
    """
    if grids.dim() != 3:
        raise ValueError(f"expected (n, h, w) grids; got shape {tuple(grids.shape)}")
    long_grids = grids.long()
    if int(long_grids.max()) >= INPUT_CHANNELS or int(long_grids.min()) < 0:
        raise ValueError("grid values must be in {0, 1, 2}")
    encoded = torch.nn.functional.one_hot(long_grids, num_classes=INPUT_CHANNELS)
    return encoded.permute(0, 3, 1, 2).float()


class WaferCNN(nn.Module):
    """Convolutional classifier returning logits and an L2-normalized embedding."""

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        super().__init__()
        self.config = config or ClassifierConfig()

        blocks: list[nn.Module] = []
        in_channels = INPUT_CHANNELS
        for out_channels in self.config.channels:
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)

        # Global pooling rather than a flatten, so the head does not depend on the
        # grid size and a change to the normalization config does not silently
        # reshape the network.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.config.dropout),
            nn.Linear(in_channels, self.config.embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(self.config.embedding_dim, NUM_CLASSES)

    def embed(self, grids: Tensor) -> Tensor:
        """Return L2-normalized embeddings for ``(n, h, w)`` grids.

        Normalized here rather than at the call site so that every consumer --
        diversity sampling, FAISS retrieval, lot centroids -- works in the same
        geometry, where inner product and cosine similarity agree.
        """
        features = self.pool(self.features(one_hot_maps(grids)))
        embedding = self.embedding_head(features)
        return torch.nn.functional.normalize(embedding, p=2.0, dim=1)

    def forward(self, grids: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(logits, embedding)``."""
        embedding = self.embed(grids)
        return self.classifier(embedding), embedding


@dataclass(slots=True)
class ClassCounts:
    """Label counts, used to build the loss weighting."""

    counts: dict[DefectPattern, int] = field(default_factory=dict)

    def weight_tensor(self, device: torch.device | None = None) -> Tensor:
        """Inverse-frequency weights, normalized to mean 1 over present classes.

        Normalizing keeps the loss on a comparable scale to the unweighted case,
        so the learning rate does not have to be retuned when weighting is
        toggled.

        Classes with no examples get a neutral weight of 1.0 and are excluded
        from the normalization. Treating an absent class as infinitely rare
        instead -- the obvious inverse-frequency reading -- makes its weight
        dominate the mean and drives every *real* class weight toward zero, which
        silently disables the weighting it was supposed to provide.
        """
        total = sum(self.counts.values())
        if total == 0:
            return torch.ones(NUM_CLASSES, device=device)

        weights = torch.ones(NUM_CLASSES, dtype=torch.float32, device=device)
        present = [
            index
            for index, cls in enumerate(CLASS_ORDER)
            if self.counts.get(cls, 0) > 0
        ]
        if not present:
            return weights

        for index in present:
            weights[index] = total / self.counts[CLASS_ORDER[index]]
        weights[present] = weights[present] / weights[present].mean()
        return weights


def label_index(pattern: DefectPattern) -> int:
    """Position of a defect pattern in the fixed class order."""
    return CLASS_ORDER.index(pattern)


def probabilities_to_dict(probabilities: Tensor) -> dict[str, float]:
    """Render a single probability vector as a class-keyed dict."""
    if probabilities.dim() != 1 or probabilities.numel() != NUM_CLASSES:
        raise ValueError(
            f"expected a {NUM_CLASSES}-element vector; got shape {tuple(probabilities.shape)}"
        )
    detached = probabilities.detach()
    return {cls.value: float(detached[index]) for index, cls in enumerate(CLASS_ORDER)}
