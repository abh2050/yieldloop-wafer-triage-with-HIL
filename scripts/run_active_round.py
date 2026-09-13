"""Run one active learning round: select wafers and queue them for review.

Loads the active classifier, scores a pool of unlabeled wafers, selects a batch
with the configured strategy, writes calibrated predictions, and creates review
tasks.

``--strategy random`` is the control arm of the label efficiency experiment. It
runs through exactly the same code path as the real strategies so the comparison
is not confounded by a difference in plumbing.

The candidate pool is capped by ``--pool``. Embedding all 638,507 unlabeled
wafers on every round would dominate the round time for no benefit: the sampler
picks tens of wafers, and a large random pool is already far more diverse than
the batch drawn from it. The cap is recorded on the round so a label efficiency
point is never compared against one drawn from a different pool size.

Usage::

    python scripts/run_active_round.py --strategy entropy_diversity --batch-size 64
    python scripts/run_active_round.py --strategy random --batch-size 64
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from yieldloop.config import Settings, get_settings
from yieldloop.db.enums import SamplingStrategy, SplitName, TaskGate
from yieldloop.db.models import ActiveRound, Prediction, Wafer
from yieldloop.db.session import build_engine
from yieldloop.guardrails.thresholds import RoutingBands, classify
from yieldloop.logging import configure_logging, get_logger
from yieldloop.models.classifier import CLASS_ORDER, probabilities_to_dict
from yieldloop.models.embed import (
    WaferDataset,
    build_loader,
    encode_embedding,
    load_samples,
    select_device,
)
from yieldloop.models.train import load_trained
from yieldloop.review.queue import create_task
from yieldloop.sampling.scheduler import SelectionRequest, select_batch
from yieldloop.sampling.uncertainty import shannon_entropy

logger = get_logger(__name__)


def run_round(
    session: Session,
    settings: Settings,
    *,
    strategy: SamplingStrategy,
    batch_size: int,
    pool_size: int,
    seed: int,
) -> int:
    """Select and queue one batch. Returns the number of tasks created."""
    loaded = load_trained(session, settings)
    if loaded is None:
        raise RuntimeError(
            "no active classifier; train one with scripts/train.py before running a round"
        )
    model, temperature = loaded
    device = select_device()
    model.to(device)

    samples = load_samples(session, SplitName.TRAIN, labeled=False, limit=pool_size)
    if not samples:
        raise RuntimeError("no unlabeled wafers in the train split")

    dataset = WaferDataset(samples)
    loader = build_loader(dataset, batch_size=256, shuffle=False, seed=seed)

    logger.info("round_scoring", pool=len(dataset), strategy=strategy.value, device=str(device))
    started = time.monotonic()

    logit_batches: list[torch.Tensor] = []
    embedding_batches: list[torch.Tensor] = []
    with torch.no_grad():
        model.eval()
        for grids, _ in loader:
            logits, embeddings = model(grids.to(device))
            logit_batches.append(logits.cpu())
            embedding_batches.append(embeddings.cpu())

    logits = torch.cat(logit_batches)
    embeddings = torch.cat(embedding_batches)
    # Calibrated, not raw: entropy over overconfident softmax ranks by the
    # network's miscalibration as much as by genuine ambiguity.
    probabilities = (logits / temperature).softmax(dim=1)
    inference_ms = (time.monotonic() - started) * 1000.0 / max(len(dataset), 1)

    probability_array = probabilities.numpy().astype(np.float64)
    entropies = shannon_entropy(probability_array)

    round_row = ActiveRound(
        round_number=_next_round_number(session, strategy),
        strategy=strategy,
        batch_size=batch_size,
        diversity_weight=settings.diversity_weight,
        seed=seed,
        labels_before=_labeled_count(session),
    )
    session.add(round_row)
    session.flush()

    selection = select_batch(
        SelectionRequest(
            wafer_ids=dataset.wafer_ids(),
            probabilities=probability_array,
            embeddings=embeddings.numpy().astype(np.float64),
            labeled_ids=frozenset(),
            strategy=strategy,
            batch_size=batch_size,
            diversity_weight=settings.diversity_weight,
            min_distance=settings.diversity_min_distance,
            seed=seed,
        )
    )

    bands = RoutingBands.from_settings(settings)
    position = {wafer_id: index for index, wafer_id in enumerate(dataset.wafer_ids())}
    artifact_id = _active_artifact_id(session)

    created = 0
    for rank, wafer_id in enumerate(selection.wafer_ids):
        index = position[wafer_id]
        wafer = session.execute(select(Wafer).where(Wafer.wafer_id == wafer_id)).scalar_one()

        row = probabilities[index]
        top = int(row.argmax())
        confidence = float(row[top])
        decision = classify(confidence, bands)

        prediction = Prediction(
            wafer_id=wafer.id,
            artifact_id=artifact_id,
            predicted_label=CLASS_ORDER[top],
            confidence=confidence,
            probabilities=probabilities_to_dict(row),
            entropy=float(entropies[index]),
            routing_band=decision.band,
            auto_commit_threshold=bands.auto_commit_threshold,
            confidence_floor=bands.confidence_floor,
            embedding=encode_embedding(embeddings[index]),
            inference_ms=inference_ms,
        )
        session.add(prediction)
        session.flush()

        create_task(
            session,
            wafer=wafer,
            gate=TaskGate.LABEL,
            prediction=prediction,
            bands=bands,
            priority=float(selection.scores[rank]) if rank < len(selection.scores) else 0.0,
            round_id=round_row.id,
        )
        created += 1

    session.commit()
    logger.info(
        "round_complete",
        strategy=strategy.value,
        selected=created,
        pool=len(dataset),
        truncated=selection.truncated,
        seconds=round(time.monotonic() - started, 1),
    )
    return created


def _next_round_number(session: Session, strategy: SamplingStrategy) -> int:
    highest = session.execute(
        select(func.max(ActiveRound.round_number)).where(ActiveRound.strategy == strategy)
    ).scalar_one_or_none()
    return int(highest or 0) + 1


def _labeled_count(session: Session) -> int:
    return int(
        session.execute(
            select(func.count())
            .select_from(Wafer)
            .where(Wafer.dataset_label.is_not(None), Wafer.split == SplitName.TRAIN)
        ).scalar_one()
    )


def _active_artifact_id(session: Session) -> object:
    from yieldloop.db.enums import ArtifactKind
    from yieldloop.db.models import ModelArtifact

    return session.execute(
        select(ModelArtifact.id).where(
            ModelArtifact.kind == ArtifactKind.CLASSIFIER, ModelArtifact.is_active.is_(True)
        )
    ).scalar_one()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in SamplingStrategy],
        default=SamplingStrategy.ENTROPY_DIVERSITY.value,
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--pool", type=int, default=20_000, help="unlabeled candidates to score")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings)

    engine = build_engine(settings)
    with Session(engine) as session:
        try:
            created = run_round(
                session,
                settings,
                strategy=SamplingStrategy(args.strategy),
                batch_size=args.batch_size or settings.round_batch_size,
                pool_size=args.pool,
                seed=args.seed if args.seed is not None else settings.partition_seed,
            )
        except RuntimeError as exc:
            print(f"{exc}", file=sys.stderr)
            return 1

    print(f"queued {created} wafers for review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
