"""One active learning round, end to end against a real database.

The round is where sampling, calibration, and routing meet, and the invariants
that matter are the ones spanning them: the batch never contains an
already-labeled wafer, every selected wafer gets a calibrated prediction and a
review task, and the task's visibility follows the routing band rather than the
reviewer's preference.

Skips when no classifier has been trained, rather than training one: a round
against an untrained model measures nothing.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import torch
from scripts.run_active_round import run_round
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from yieldloop.config import Settings
from yieldloop.db.enums import RoutingBand, SamplingStrategy, SplitName, TaskGate, TaskState
from yieldloop.db.models import ActiveRound, Prediction, ReviewTask, Wafer
from yieldloop.db.session import build_engine
from yieldloop.models.embed import decode_embedding
from yieldloop.models.train import load_trained

pytestmark = pytest.mark.integration

POOL = 2_000
BATCH = 16


@pytest.fixture(scope="module")
def live_session() -> Iterator[Session]:
    """A session against the configured database, which holds the real corpus."""
    session = Session(build_engine())
    settings = Settings()
    try:
        wafers = int(session.execute(select(func.count()).select_from(Wafer)).scalar_one())
    except Exception:
        session.close()
        pytest.skip("configured database is unreachable")
    if wafers == 0:
        session.close()
        pytest.skip("no wafers ingested; run scripts/bootstrap_db.py")
    if load_trained(session, settings) is None:
        session.close()
        pytest.skip("no active classifier; run python -m scripts.train")
    yield session
    session.close()


@pytest.fixture
def round_result(live_session: Session) -> Iterator[int]:
    """Run one real round and roll it back afterwards.

    The round commits internally, so the tasks it creates are removed explicitly
    rather than by transaction rollback -- the database is the developer's real
    one and must be left as it was found.
    """
    settings = Settings()
    created = run_round(
        live_session,
        settings,
        strategy=SamplingStrategy.ENTROPY_DIVERSITY,
        batch_size=BATCH,
        pool_size=POOL,
        seed=4242,
    )
    yield created

    round_row = live_session.execute(
        select(ActiveRound).order_by(ActiveRound.created_at.desc()).limit(1)
    ).scalar_one()
    task_ids = (
        live_session.execute(select(ReviewTask.id).where(ReviewTask.round_id == round_row.id))
        .scalars()
        .all()
    )
    prediction_ids = (
        live_session.execute(
            select(ReviewTask.prediction_id).where(ReviewTask.round_id == round_row.id)
        )
        .scalars()
        .all()
    )
    for task_id in task_ids:
        live_session.delete(live_session.get(ReviewTask, task_id))
    live_session.flush()
    for prediction_id in prediction_ids:
        if prediction_id is not None:
            row = live_session.get(Prediction, prediction_id)
            if row is not None:
                live_session.delete(row)
    live_session.delete(round_row)
    live_session.commit()


def test_a_round_selects_at_most_the_requested_batch(round_result: int) -> None:
    """At most, not exactly.

    Wafers already waiting in the queue from an earlier round are skipped rather
    than queued twice, which would have two reviewers label the same wafer
    independently -- duplicated effort, not a second opinion. A short batch is
    the correct outcome, consistent with the sampler returning fewer rather than
    padding with near-duplicates.
    """
    assert 0 < round_result <= BATCH


def test_every_selected_wafer_is_unlabeled(live_session: Session, round_result: int) -> None:
    """Re-showing a labeled wafer wastes the resource the project exists to save."""
    round_row = live_session.execute(
        select(ActiveRound).order_by(ActiveRound.created_at.desc()).limit(1)
    ).scalar_one()
    labels = (
        live_session.execute(
            select(Wafer.dataset_label)
            .join(ReviewTask, ReviewTask.wafer_id == Wafer.id)
            .where(ReviewTask.round_id == round_row.id)
        )
        .scalars()
        .all()
    )
    assert labels, "round produced no tasks"
    assert all(label is None for label in labels)


def test_selected_wafers_are_unique(live_session: Session, round_result: int) -> None:
    round_row = live_session.execute(
        select(ActiveRound).order_by(ActiveRound.created_at.desc()).limit(1)
    ).scalar_one()
    wafer_ids = (
        live_session.execute(select(ReviewTask.wafer_id).where(ReviewTask.round_id == round_row.id))
        .scalars()
        .all()
    )
    assert len(set(wafer_ids)) == len(wafer_ids)


def test_every_selection_has_a_calibrated_prediction(
    live_session: Session, round_result: int
) -> None:
    round_row = live_session.execute(
        select(ActiveRound).order_by(ActiveRound.created_at.desc()).limit(1)
    ).scalar_one()
    rows = (
        live_session.execute(
            select(Prediction)
            .join(ReviewTask, ReviewTask.prediction_id == Prediction.id)
            .where(ReviewTask.round_id == round_row.id)
        )
        .scalars()
        .all()
    )
    assert 0 < len(rows) <= BATCH
    for prediction in rows:
        assert 0.0 <= prediction.confidence <= 1.0
        total = sum(prediction.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-4)
        assert prediction.entropy >= 0.0
        # The thresholds in force are recorded alongside, so the routing decision
        # stays reconstructible after a threshold change.
        assert prediction.confidence_floor < prediction.auto_commit_threshold


def test_predictions_carry_a_usable_embedding(live_session: Session, round_result: int) -> None:
    """The same vector drives diversity sampling and retrieval, so it must be
    stored in a form both can read."""
    settings = Settings()
    round_row = live_session.execute(
        select(ActiveRound).order_by(ActiveRound.created_at.desc()).limit(1)
    ).scalar_one()
    blobs = (
        live_session.execute(
            select(Prediction.embedding)
            .join(ReviewTask, ReviewTask.prediction_id == Prediction.id)
            .where(ReviewTask.round_id == round_row.id)
        )
        .scalars()
        .all()
    )
    assert blobs and all(blob is not None for blob in blobs)
    for blob in blobs:
        assert blob is not None
        vector = decode_embedding(blob, settings.embedding_dim)
        assert vector.shape == (settings.embedding_dim,)
        assert np.isfinite(vector).all()
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-3)


def test_label_gate_tasks_never_show_the_prediction(
    live_session: Session, round_result: int
) -> None:
    """Whatever the confidence. An independent label is the point of this gate."""
    round_row = live_session.execute(
        select(ActiveRound).order_by(ActiveRound.created_at.desc()).limit(1)
    ).scalar_one()
    tasks = (
        live_session.execute(select(ReviewTask).where(ReviewTask.round_id == round_row.id))
        .scalars()
        .all()
    )
    assert tasks
    for task in tasks:
        assert task.gate is TaskGate.LABEL
        assert task.state is TaskState.PENDING
        assert not task.show_prediction


def test_the_round_records_what_it_would_take_to_repeat_it(
    live_session: Session, round_result: int
) -> None:
    """A label efficiency point is only comparable if its provenance is recorded."""
    round_row = live_session.execute(
        select(ActiveRound).order_by(ActiveRound.created_at.desc()).limit(1)
    ).scalar_one()
    assert round_row.strategy is SamplingStrategy.ENTROPY_DIVERSITY
    assert round_row.batch_size == BATCH
    assert round_row.seed == 4242
    assert round_row.labels_before > 0


def test_the_sampler_prefers_uncertain_wafers(live_session: Session, round_result: int) -> None:
    """Entropy sampling should not be picking confident wafers.

    A round dominated by auto-commit-band predictions would mean the sampler is
    spending reviewer time on cases the system could already commit itself.
    """
    round_row = live_session.execute(
        select(ActiveRound).order_by(ActiveRound.created_at.desc()).limit(1)
    ).scalar_one()
    bands = (
        live_session.execute(
            select(Prediction.routing_band)
            .join(ReviewTask, ReviewTask.prediction_id == Prediction.id)
            .where(ReviewTask.round_id == round_row.id)
        )
        .scalars()
        .all()
    )
    assert bands
    confident = sum(1 for band in bands if band is RoutingBand.AUTO_COMMIT)
    assert confident / len(bands) < 0.25


def test_a_round_is_reproducible_from_its_seed(live_session: Session) -> None:
    """Two rounds with the same seed and pool must select the same wafers."""
    settings = Settings()
    model, temperature = load_trained(live_session, settings)  # type: ignore[misc]
    device = torch.device("cpu")
    model.to(device)

    from yieldloop.models.embed import WaferDataset, build_loader, load_samples
    from yieldloop.sampling.scheduler import SelectionRequest, select_batch

    samples = load_samples(live_session, SplitName.TRAIN, labeled=False, limit=500)
    dataset = WaferDataset(samples)
    loader = build_loader(dataset, batch_size=256, shuffle=False, seed=1)

    logits_batches, embedding_batches = [], []
    with torch.no_grad():
        model.eval()
        for grids, _ in loader:
            logits, embeddings = model(grids.to(device))
            logits_batches.append(logits.cpu())
            embedding_batches.append(embeddings.cpu())
    probabilities = (torch.cat(logits_batches) / temperature).softmax(dim=1).numpy()
    embeddings_array = torch.cat(embedding_batches).numpy().astype(np.float64)

    def pick() -> tuple[str, ...]:
        return select_batch(
            SelectionRequest(
                wafer_ids=dataset.wafer_ids(),
                probabilities=probabilities.astype(np.float64),
                embeddings=embeddings_array,
                labeled_ids=frozenset(),
                strategy=SamplingStrategy.ENTROPY_DIVERSITY,
                batch_size=8,
                diversity_weight=settings.diversity_weight,
                min_distance=settings.diversity_min_distance,
                seed=99,
            )
        ).wafer_ids

    assert pick() == pick()
