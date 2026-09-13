"""Enumerations shared by the database schema and the API boundary.

The defect pattern values are exactly the nine ``failureType`` values present in
the real WM811K dataset. They are not invented and must not be extended without
a corresponding change to the data contract.
"""

from __future__ import annotations

from enum import StrEnum


class DefectPattern(StrEnum):
    """The nine real WM811K ``failureType`` labels, normalized to snake case."""

    CENTER = "center"
    DONUT = "donut"
    EDGE_LOC = "edge_loc"
    EDGE_RING = "edge_ring"
    LOC = "loc"
    NEAR_FULL = "near_full"
    RANDOM = "random"
    SCRATCH = "scratch"
    NONE = "none"

    @classmethod
    def from_dataset_label(cls, raw: str) -> DefectPattern:
        """Map a raw WM811K ``failureType`` string onto a member.

        The dataset stores labels such as ``"Edge-Loc"`` and ``"Near-full"``.
        The mapping is a pure normalization: lowercase, and hyphens and spaces
        collapsed to underscores. Unknown values raise rather than silently
        becoming ``none``, so a dataset change surfaces as an ingest failure.
        """
        normalized = raw.strip().lower().replace("-", "_").replace(" ", "_")
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(
                f"unrecognized WM811K failureType {raw!r} (normalized to {normalized!r}); "
                "the dataset contract in docs/data_contract.md lists the nine expected values"
            ) from exc


class SplitName(StrEnum):
    """Deterministic lot-keyed partition assignment."""

    TRAIN = "train"
    VAL = "val"
    HOLDOUT = "holdout"


class LabelSource(StrEnum):
    """Where a label on a wafer came from."""

    #: The original human label shipped with WM811K.
    DATASET = "dataset"
    #: A decision captured by this console from a reviewer.
    REVIEWER = "reviewer"


class RoutingBand(StrEnum):
    """Which side of the configured routing bands a prediction fell on."""

    #: At or above the auto-commit threshold: committed without a human.
    AUTO_COMMIT = "auto_commit"
    #: Between the confidence floor and auto-commit: human sees the prediction.
    UNCERTAINTY_BAND = "uncertainty_band"
    #: Below the floor: human reviews with the prediction withheld, to avoid anchoring.
    BELOW_FLOOR = "below_floor"


class TaskState(StrEnum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    EXPIRED = "expired"


class TaskGate(StrEnum):
    """Which of the three human gates produced this task."""

    #: Most-informative unlabeled wafers surfaced for labeling.
    LABEL = "label"
    #: Uncertainty-band predictions surfaced for confirmation.
    CONFIRM = "confirm"
    #: Flagged lots surfaced with ranked root cause hypotheses.
    ESCALATION = "escalation"


class DecisionAction(StrEnum):
    ACCEPT = "accept"
    EDIT = "edit"
    REJECT = "reject"


class SamplingStrategy(StrEnum):
    """How a round selected its batch. ``RANDOM`` is the label-efficiency control."""

    ENTROPY = "entropy"
    DIVERSITY = "diversity"
    ENTROPY_DIVERSITY = "entropy_diversity"
    RANDOM = "random"


class ArtifactKind(StrEnum):
    CLASSIFIER = "classifier"
    CALIBRATOR = "calibrator"
    EMBEDDER = "embedder"
    FAISS_INDEX = "faiss_index"


class EvidenceKind(StrEnum):
    """The kinds of evidence that may appear in an agent context bundle."""

    CLASSIFIER_PREDICTION = "classifier_prediction"
    DIE_STATISTICS = "die_statistics"
    SIMILAR_LOT = "similar_lot"
    PROCESS_EVENT = "process_event"


class CauseCategory(StrEnum):
    """Root cause taxonomy the agent must choose from."""

    TOOL_DRIFT = "tool_drift"
    CHAMBER_CONDITION = "chamber_condition"
    RECIPE_CHANGE = "recipe_change"
    MATERIAL_LOT = "material_lot"
    HANDLING_MECHANICAL = "handling_mechanical"
    METROLOGY_ARTIFACT = "metrology_artifact"
    UPSTREAM_PROCESS = "upstream_process"
    UNKNOWN = "unknown"


class AuditEventType(StrEnum):
    """Every category of event the audit log is required to capture."""

    MODEL_OUTPUT = "model_output"
    HUMAN_DECISION = "human_decision"
    GUARDRAIL_ACTION = "guardrail_action"
    THRESHOLD_CHANGE = "threshold_change"


class GuardrailStage(StrEnum):
    """Which guardrail produced an action record."""

    INPUT_FILTER = "input_filter"
    INJECTION = "injection"
    SCHEMA_VALIDATOR = "schema_validator"
    GROUNDING = "grounding"
    THRESHOLDS = "thresholds"
    BUDGET = "budget"
    CIRCUIT_BREAKER = "circuit_breaker"


class GuardrailOutcome(StrEnum):
    PASSED = "passed"
    #: Part of the payload was removed or isolated, and processing continued.
    MODIFIED = "modified"
    #: The guardrail failed closed and nothing reached the caller.
    BLOCKED = "blocked"


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
