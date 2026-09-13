"""SQLAlchemy 2.0 ORM models.

Provenance is carried in the schema rather than in documentation alone. Any row
that is not a direct record of a real WM811K field or a real human action carries
a ``derivation_rule`` naming the deterministic function in
:mod:`yieldloop.ingest.normalize` that produced it, and that rule is described in
``docs/data_contract.md``. Nothing in this schema can hold an observed fab event
without saying where it came from.

The ``audit_records`` table is append only. The constraint is enforced in the
migration by revoking UPDATE and DELETE from the application role, not by
application convention.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, ClassVar
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from yieldloop.db.enums import (
    ArtifactKind,
    AuditEventType,
    BreakerState,
    CauseCategory,
    DecisionAction,
    DefectPattern,
    EvidenceKind,
    GuardrailOutcome,
    GuardrailStage,
    LabelSource,
    RoutingBand,
    SamplingStrategy,
    SplitName,
    TaskGate,
    TaskState,
)


def _pg_enum(enum_type: type, name: str) -> SAEnum:
    """Build a Postgres enum column type that stores member *values*."""
    return SAEnum(
        enum_type,
        name=name,
        native_enum=True,
        values_callable=lambda e: [member.value for member in e],
        validate_strings=True,
    )


class Base(DeclarativeBase):
    """Declarative base for every yieldloop table."""

    #: SQLAlchemy reads this mapping at class construction time; it is the
    #: documented extension point, not a mutable default argument.
    type_annotation_map: ClassVar[dict[Any, Any]] = {
        dict[str, Any]: JSONB,
        list[str]: JSONB,
    }


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ---------------------------------------------------------------------------
# Real dataset entities
# ---------------------------------------------------------------------------


class Lot(Base, TimestampMixin):
    """A production lot, keyed by the real WM811K ``lotName``.

    Partitioning is lot keyed so that no lot straddles a split.
    """

    __tablename__ = "lots"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    lot_name: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    wafer_count: Mapped[int] = mapped_column(Integer, nullable=False)
    split: Mapped[SplitName] = mapped_column(
        _pg_enum(SplitName, "split_name"), nullable=False, index=True
    )
    #: Ordinal position of this lot in the deterministic lot ordering used to
    #: derive the process timeline. See docs/data_contract.md, rule LOT_ORDINAL.
    lot_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Synthetic-but-deterministic calendar date assigned to the lot so that the
    #: retention window and the process-event window have a total order.
    #: Derived, never observed. See docs/data_contract.md, rule LOT_DATE.
    derived_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    derivation_rule: Mapped[str] = mapped_column(String(64), nullable=False)

    wafers: Mapped[list[Wafer]] = relationship(back_populates="lot", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("wafer_count > 0", name="ck_lots_wafer_count_positive"),
        CheckConstraint("lot_ordinal >= 0", name="ck_lots_ordinal_nonneg"),
        Index("ix_lots_split_ordinal", "split", "lot_ordinal"),
    )


class Wafer(Base, TimestampMixin):
    """A single wafer map from WM811K, normalized to the configured grid.

    ``grid`` holds the normalized map as raveled ``uint8`` bytes with the WM811K
    encoding preserved: 0 outside the wafer, 1 passing die, 2 failing die. The
    raw pre-normalization shape is kept so normalization is auditable.
    """

    __tablename__ = "wafers"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    lot_id: Mapped[UUID] = mapped_column(
        ForeignKey("lots.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: Stable identifier: ``{lot_name}-{wafer_index}``. Unique across the dataset.
    wafer_id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True, index=True)
    #: Real WM811K ``waferIndex``.
    wafer_index: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Real WM811K ``dieSize``.
    die_size: Mapped[float] = mapped_column(Float, nullable=False)
    #: Row index of this wafer in the source pickle, for exact traceability.
    source_row: Mapped[int] = mapped_column(BigInteger, nullable=False, unique=True)

    raw_height: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_width: Mapped[int] = mapped_column(Integer, nullable=False)
    grid_height: Mapped[int] = mapped_column(Integer, nullable=False)
    grid_width: Mapped[int] = mapped_column(Integer, nullable=False)
    grid: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    #: Real die counts, computed from the raw map before normalization.
    die_total: Mapped[int] = mapped_column(Integer, nullable=False)
    die_fail: Mapped[int] = mapped_column(Integer, nullable=False)

    #: The original human label from WM811K. Null where the dataset is unlabeled,
    #: which is the majority of it and is what makes active learning meaningful.
    dataset_label: Mapped[DefectPattern | None] = mapped_column(
        _pg_enum(DefectPattern, "defect_pattern"), nullable=True, index=True
    )
    #: Real WM811K ``trainTestLabel`` where present; preserved but not used for
    #: partitioning, since yieldloop partitions by lot.
    dataset_split_label: Mapped[str | None] = mapped_column(String(16), nullable=True)

    split: Mapped[SplitName] = mapped_column(
        _pg_enum(SplitName, "split_name"), nullable=False, index=True
    )

    lot: Mapped[Lot] = relationship(back_populates="wafers")
    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="wafer", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("die_total > 0", name="ck_wafers_die_total_positive"),
        CheckConstraint("die_fail >= 0 AND die_fail <= die_total", name="ck_wafers_die_fail_range"),
        CheckConstraint("grid_height > 0 AND grid_width > 0", name="ck_wafers_grid_positive"),
        CheckConstraint(
            "octet_length(grid) = grid_height * grid_width", name="ck_wafers_grid_bytes_match"
        ),
        UniqueConstraint("lot_id", "wafer_index", name="uq_wafers_lot_index"),
        Index("ix_wafers_split_label", "split", "dataset_label"),
    )


# ---------------------------------------------------------------------------
# Model registry and predictions
# ---------------------------------------------------------------------------


class ModelArtifact(Base, TimestampMixin):
    """A content-addressed trained artifact.

    Every artifact records the hash of the data it was trained on, the git commit
    of the code that produced it, and the full hyperparameter set, so any result
    in the eval report can be traced to a reproducible training run.
    """

    __tablename__ = "model_artifacts"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    kind: Mapped[ArtifactKind] = mapped_column(
        _pg_enum(ArtifactKind, "artifact_kind"), nullable=False
    )
    #: sha256 of the serialized artifact; the registry filename is this digest.
    content_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    #: sha256 over the ordered wafer ids and labels the artifact was fit on.
    data_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    git_commit: Mapped[str] = mapped_column(String(40), nullable=False)
    git_dirty: Mapped[bool] = mapped_column(Boolean, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    hyperparameters: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    #: Number of labels available at training time. The x-axis of the label
    #: efficiency curve.
    label_count: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("model_artifacts.id", ondelete="SET NULL"), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)

    __table_args__ = (
        CheckConstraint("label_count >= 0", name="ck_artifacts_label_count_nonneg"),
        CheckConstraint("char_length(content_hash) = 64", name="ck_artifacts_content_hash_len"),
        CheckConstraint("char_length(data_hash) = 64", name="ck_artifacts_data_hash_len"),
        Index("ix_artifacts_kind_active", "kind", "is_active"),
    )


class Prediction(Base, TimestampMixin):
    """A calibrated classifier prediction for one wafer under one artifact."""

    __tablename__ = "predictions"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    wafer_id: Mapped[UUID] = mapped_column(
        ForeignKey("wafers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    artifact_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_artifacts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    predicted_label: Mapped[DefectPattern] = mapped_column(
        _pg_enum(DefectPattern, "defect_pattern"), nullable=False
    )
    #: Calibrated probability of ``predicted_label``.
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    #: Full calibrated distribution, keyed by DefectPattern value.
    probabilities: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    #: Shannon entropy of the calibrated distribution, in nats. Drives sampling.
    entropy: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    #: Which routing band this prediction fell in, under the thresholds in force
    #: at the time. The thresholds themselves are recorded alongside so the
    #: decision is reconstructible after a threshold change.
    routing_band: Mapped[RoutingBand] = mapped_column(
        _pg_enum(RoutingBand, "routing_band"), nullable=False, index=True
    )
    auto_commit_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_floor: Mapped[float] = mapped_column(Float, nullable=False)
    #: L2-normalized embedding used for diversity sampling and retrieval.
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    inference_ms: Mapped[float] = mapped_column(Float, nullable=False)

    wafer: Mapped[Wafer] = relationship(back_populates="predictions")

    __table_args__ = (
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="ck_pred_conf_range"),
        CheckConstraint("entropy >= 0.0", name="ck_pred_entropy_nonneg"),
        CheckConstraint(
            "confidence_floor < auto_commit_threshold", name="ck_pred_bands_ordered"
        ),
        UniqueConstraint("wafer_id", "artifact_id", name="uq_pred_wafer_artifact"),
    )


# ---------------------------------------------------------------------------
# Human loop
# ---------------------------------------------------------------------------


class ActiveRound(Base, TimestampMixin):
    """One round of active learning: a strategy, a batch, and its outcome."""

    __tablename__ = "active_rounds"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    strategy: Mapped[SamplingStrategy] = mapped_column(
        _pg_enum(SamplingStrategy, "sampling_strategy"), nullable=False
    )
    artifact_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("model_artifacts.id", ondelete="SET NULL"), nullable=True
    )
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    diversity_weight: Mapped[float] = mapped_column(Float, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Labels available when the round was planned. Paired with ``strategy``,
    #: this is what the label efficiency curve is plotted against.
    labels_before: Mapped[int] = mapped_column(Integer, nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    tasks: Mapped[list[ReviewTask]] = relationship(back_populates="round")

    __table_args__ = (
        UniqueConstraint("round_number", "strategy", name="uq_round_number_strategy"),
        CheckConstraint("batch_size > 0", name="ck_round_batch_positive"),
    )


class ReviewTask(Base, TimestampMixin):
    """One unit of human work, produced by one of the three gates."""

    __tablename__ = "review_tasks"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    wafer_id: Mapped[UUID] = mapped_column(
        ForeignKey("wafers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    round_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("active_rounds.id", ondelete="SET NULL"), nullable=True, index=True
    )
    prediction_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("predictions.id", ondelete="SET NULL"), nullable=True
    )
    gate: Mapped[TaskGate] = mapped_column(
        _pg_enum(TaskGate, "task_gate"), nullable=False, index=True
    )
    state: Mapped[TaskState] = mapped_column(
        _pg_enum(TaskState, "task_state"), nullable=False, default=TaskState.PENDING, index=True
    )
    #: Selection score from the sampler; higher means more informative.
    priority: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    #: Whether the console is permitted to show the model prediction. False below
    #: the confidence floor, so the reviewer is not anchored.
    show_prediction: Mapped[bool] = mapped_column(Boolean, nullable=False)
    assigned_to: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    assigned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    round: Mapped[ActiveRound | None] = relationship(back_populates="tasks")
    decision: Mapped[Decision | None] = relationship(back_populates="task", uselist=False)

    __table_args__ = (
        UniqueConstraint("wafer_id", "gate", "round_id", name="uq_task_wafer_gate_round"),
        Index("ix_tasks_queue_order", "state", "gate", "priority"),
    )


class ReasonCode(Base, TimestampMixin):
    """Controlled vocabulary a reviewer must pick from when editing or rejecting."""

    __tablename__ = "reason_codes"

    code: Mapped[str] = mapped_column(String(48), primary_key=True)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    #: Which actions this code is valid for, as DecisionAction values.
    applies_to: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    #: Whether the code applies to classifier decisions, hypothesis decisions, or both.
    applies_to_gates: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class Decision(Base, TimestampMixin):
    """A human decision. This is the training signal the next round consumes."""

    __tablename__ = "decisions"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    task_id: Mapped[UUID] = mapped_column(
        ForeignKey("review_tasks.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    wafer_id: Mapped[UUID] = mapped_column(
        ForeignKey("wafers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    reviewer_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    action: Mapped[DecisionAction] = mapped_column(
        _pg_enum(DecisionAction, "decision_action"), nullable=False, index=True
    )
    #: The label the reviewer settled on. Null only for a rejected hypothesis,
    #: where the decision is about the agent output rather than the wafer class.
    chosen_label: Mapped[DefectPattern | None] = mapped_column(
        _pg_enum(DefectPattern, "defect_pattern"), nullable=True, index=True
    )
    #: What the model had predicted, denormalized so agreement and override rate
    #: remain computable after an artifact is retired.
    model_label: Mapped[DefectPattern | None] = mapped_column(
        _pg_enum(DefectPattern, "defect_pattern"), nullable=True
    )
    model_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: True when the reviewer's label differs from what the model predicted.
    is_override: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    #: Whether the prediction was visible to the reviewer, so anchoring effects
    #: can be measured rather than assumed away.
    prediction_was_shown: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reason_code: Mapped[str | None] = mapped_column(
        ForeignKey("reason_codes.code", ondelete="RESTRICT"), nullable=True, index=True
    )
    #: Reviewer free text. Bounded by the input filter and treated as untrusted
    #: data everywhere downstream.
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[LabelSource] = mapped_column(
        _pg_enum(LabelSource, "label_source"), nullable=False, default=LabelSource.REVIEWER
    )
    #: Wall time from task presentation to submit. The reviewer throughput claim
    #: is measured from this column, not estimated.
    decision_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Set when the decision concerns an agent hypothesis rather than a label.
    hypothesis_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("hypotheses.id", ondelete="SET NULL"), nullable=True, index=True
    )

    task: Mapped[ReviewTask] = relationship(back_populates="decision")

    __table_args__ = (
        CheckConstraint("decision_ms >= 0", name="ck_decision_ms_nonneg"),
        CheckConstraint(
            "model_confidence IS NULL OR (model_confidence >= 0.0 AND model_confidence <= 1.0)",
            name="ck_decision_conf_range",
        ),
        CheckConstraint(
            "action = 'accept' OR reason_code IS NOT NULL",
            name="ck_decision_reason_required_on_change",
        ),
        Index("ix_decisions_reviewer_time", "reviewer_id", "created_at"),
    )


# ---------------------------------------------------------------------------
# Retrieval corpus
# ---------------------------------------------------------------------------


class ProcessEvent(Base, TimestampMixin):
    """A process event in the window around a lot.

    These rows are derived, not observed. Each carries the deterministic rule
    from ``docs/data_contract.md`` that produced it from real WM811K lot and
    wafer index structure. ``is_derived`` is non-nullable and always true here,
    so no consumer can mistake this for measured fab telemetry.
    """

    __tablename__ = "process_events"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    #: Stable evidence identifier cited by the agent, e.g. ``pe:LOT123:0``.
    evidence_id: Mapped[str] = mapped_column(String(96), nullable=False, unique=True, index=True)
    lot_id: Mapped[UUID] = mapped_column(
        ForeignKey("lots.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    #: Constrained to the CauseCategory taxonomy so the agent cannot cite an
    #: event class that has no place in its output schema.
    category: Mapped[CauseCategory] = mapped_column(
        _pg_enum(CauseCategory, "cause_category"), nullable=False
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    is_derived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    derivation_rule: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        CheckConstraint("is_derived IS TRUE", name="ck_process_events_always_derived"),
    )


class HistoricalExcursion(Base, TimestampMixin):
    """A past lot with a resolved root cause, retrieved as precedent.

    The resolution is grounded in real WM811K labels: the lot's dominant human
    defect label determines the recorded cause category via a documented mapping.
    """

    __tablename__ = "historical_excursions"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    evidence_id: Mapped[str] = mapped_column(String(96), nullable=False, unique=True, index=True)
    lot_id: Mapped[UUID] = mapped_column(
        ForeignKey("lots.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: The real human defect label that dominates this lot.
    observed_pattern: Mapped[DefectPattern] = mapped_column(
        _pg_enum(DefectPattern, "defect_pattern"), nullable=False, index=True
    )
    #: Fraction of labeled wafers in the lot carrying ``observed_pattern``.
    pattern_share: Mapped[float] = mapped_column(Float, nullable=False)
    resolved_cause: Mapped[CauseCategory] = mapped_column(
        _pg_enum(CauseCategory, "cause_category"), nullable=False
    )
    #: Free text describing the resolution. Untrusted input to the agent: it is
    #: structurally isolated and scanned for instruction-shaped content.
    resolution_text: Mapped[str] = mapped_column(Text, nullable=False)
    #: Mean embedding over the lot's wafers, used for similarity retrieval.
    centroid: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    is_derived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    derivation_rule: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        CheckConstraint(
            "pattern_share > 0.0 AND pattern_share <= 1.0", name="ck_excursion_share_range"
        ),
    )


# ---------------------------------------------------------------------------
# Agent output
# ---------------------------------------------------------------------------


class HypothesisRequest(Base, TimestampMixin):
    """One guarded agent invocation, recorded whether or not it produced output."""

    __tablename__ = "hypothesis_requests"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    lot_id: Mapped[UUID] = mapped_column(
        ForeignKey("lots.id", ondelete="CASCADE"), nullable=False, index=True
    )
    requested_by: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    #: sha256 over the serialized context bundle. Grounding is checked against
    #: the bundle with this hash, so a later audit can prove what was available.
    context_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    #: Every evidence_id present in the bundle, which is exactly the set the
    #: agent was permitted to cite.
    evidence_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    abstained: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    abstention_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    injection_suspected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    context_gaps: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    #: Hypotheses the model returned before the grounding gate.
    hypotheses_returned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Hypotheses that survived grounding. The gap between the two is the
    #: grounding rejection rate the circuit breaker watches.
    hypotheses_grounded: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    #: True when the breaker was open and the console fell back to classifier only.
    degraded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    hypotheses: Mapped[list[Hypothesis]] = relationship(
        back_populates="request", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "hypotheses_grounded <= hypotheses_returned", name="ck_hypreq_grounded_le_returned"
        ),
        CheckConstraint("cost_usd >= 0.0", name="ck_hypreq_cost_nonneg"),
        CheckConstraint(
            "abstained IS FALSE OR abstention_reason IS NOT NULL",
            name="ck_hypreq_abstention_reason_required",
        ),
    )


class Hypothesis(Base, TimestampMixin):
    """A single ranked hypothesis that passed the grounding gate."""

    __tablename__ = "hypotheses"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    request_id: Mapped[UUID] = mapped_column(
        ForeignKey("hypothesis_requests.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    cause_category: Mapped[CauseCategory] = mapped_column(
        _pg_enum(CauseCategory, "cause_category"), nullable=False, index=True
    )
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    supporting_signal: Mapped[str] = mapped_column(Text, nullable=False)
    contradicting_signal: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    confirming_query: Mapped[str] = mapped_column(Text, nullable=False)
    eliminating_query: Mapped[str] = mapped_column(Text, nullable=False)

    request: Mapped[HypothesisRequest] = relationship(back_populates="hypotheses")
    citations: Mapped[list[HypothesisCitation]] = relationship(
        back_populates="hypothesis", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("rank >= 1", name="ck_hypothesis_rank_positive"),
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="ck_hypothesis_conf_range"),
        UniqueConstraint("request_id", "rank", name="uq_hypothesis_request_rank"),
    )


class HypothesisCitation(Base, TimestampMixin):
    """An evidence reference on a hypothesis.

    A row exists here only if the evidence ID was present in the context bundle
    for the parent request. The grounding gate drops the claim otherwise, so an
    unresolvable citation is never persisted and never rendered.
    """

    __tablename__ = "hypothesis_citations"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    hypothesis_id: Mapped[UUID] = mapped_column(
        ForeignKey("hypotheses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    evidence_id: Mapped[str] = mapped_column(String(96), nullable=False, index=True)
    evidence_kind: Mapped[EvidenceKind] = mapped_column(
        _pg_enum(EvidenceKind, "evidence_kind"), nullable=False
    )

    hypothesis: Mapped[Hypothesis] = relationship(back_populates="citations")

    __table_args__ = (
        UniqueConstraint("hypothesis_id", "evidence_id", name="uq_citation_hyp_evidence"),
    )


# ---------------------------------------------------------------------------
# Guardrail state and the audit log
# ---------------------------------------------------------------------------


class AuditRecord(Base):
    """Append-only audit log.

    UPDATE and DELETE are revoked from the application role in the migration, so
    immutability is a database property. There is no ``updated_at`` because a row
    here is never modified.
    """

    __tablename__ = "audit_records"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    #: Monotonic sequence, so gaps and reordering are detectable. GENERATED
    #: ALWAYS means the application cannot supply or override the value.
    sequence: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), nullable=False, unique=True, index=True
    )
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    event_type: Mapped[AuditEventType] = mapped_column(
        _pg_enum(AuditEventType, "audit_event_type"), nullable=False, index=True
    )
    actor: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    #: Table name and primary key of the subject row, as free-form strings so the
    #: audit log survives the deletion of what it describes.
    subject_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    subject_id: Mapped[str] = mapped_column(String(96), nullable=False, index=True)
    #: Correlation id linking every record produced by one request.
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    #: sha256 of the previous record's digest concatenated with this payload,
    #: making silent tampering detectable even by a superuser.
    prev_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    digest: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    __table_args__ = (
        CheckConstraint("char_length(digest) = 64", name="ck_audit_digest_len"),
        Index("ix_audit_type_time", "event_type", "occurred_at"),
    )


class GuardrailAction(Base, TimestampMixin):
    """A guardrail intervention, recorded for the adversarial test suite and the
    guardrail dashboards. Every row is mirrored into ``audit_records``."""

    __tablename__ = "guardrail_actions"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    stage: Mapped[GuardrailStage] = mapped_column(
        _pg_enum(GuardrailStage, "guardrail_stage"), nullable=False, index=True
    )
    outcome: Mapped[GuardrailOutcome] = mapped_column(
        _pg_enum(GuardrailOutcome, "guardrail_outcome"), nullable=False, index=True
    )
    reason: Mapped[str] = mapped_column(String(128), nullable=False)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (Index("ix_guardrail_stage_outcome", "stage", "outcome"),)


class CostLedgerEntry(Base, TimestampMixin):
    """Per-call spend, the source of truth for session and daily ceilings."""

    __tablename__ = "cost_ledger"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    spend_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        CheckConstraint("cost_usd >= 0.0", name="ck_ledger_cost_nonneg"),
        CheckConstraint("prompt_tokens >= 0", name="ck_ledger_prompt_nonneg"),
        CheckConstraint("completion_tokens >= 0", name="ck_ledger_completion_nonneg"),
        Index("ix_ledger_session_date", "session_id", "spend_date"),
    )


class BreakerEvent(Base, TimestampMixin):
    """A circuit breaker state transition."""

    __tablename__ = "breaker_events"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    from_state: Mapped[BreakerState] = mapped_column(
        _pg_enum(BreakerState, "breaker_state"), nullable=False
    )
    to_state: Mapped[BreakerState] = mapped_column(
        _pg_enum(BreakerState, "breaker_state"), nullable=False
    )
    trigger: Mapped[str] = mapped_column(String(64), nullable=False)
    observed: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    __table_args__ = (CheckConstraint("from_state <> to_state", name="ck_breaker_state_changes"),)


class ThresholdChange(Base, TimestampMixin):
    """A change to the routing bands, recorded with who made it and why."""

    __tablename__ = "threshold_changes"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    changed_by: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    old_value: Mapped[float] = mapped_column(Float, nullable=False)
    new_value: Mapped[float] = mapped_column(Float, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        CheckConstraint("old_value <> new_value", name="ck_threshold_actually_changed"),
    )


class DriftSnapshot(Base, TimestampMixin):
    """A scheduled telemetry rollup: override rate, calibration drift, input drift."""

    __tablename__ = "drift_snapshots"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    artifact_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("model_artifacts.id", ondelete="SET NULL"), nullable=True, index=True
    )
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decision_count: Mapped[int] = mapped_column(Integer, nullable=False)
    override_rate: Mapped[float] = mapped_column(Float, nullable=False)
    expected_calibration_error: Mapped[float] = mapped_column(Float, nullable=False)
    #: Population stability index of the input embedding distribution against the
    #: training distribution.
    input_psi: Mapped[float] = mapped_column(Float, nullable=False)
    per_class: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    __table_args__ = (
        CheckConstraint("window_end > window_start", name="ck_drift_window_ordered"),
        CheckConstraint(
            "override_rate >= 0.0 AND override_rate <= 1.0", name="ck_drift_override_range"
        ),
        CheckConstraint("decision_count >= 0", name="ck_drift_count_nonneg"),
    )
