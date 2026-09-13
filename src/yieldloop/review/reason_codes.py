"""The controlled vocabulary a reviewer picks from when editing or rejecting.

Free-text-only disagreement is unusable as signal. "Wrong" tells the next
training round nothing, whereas "the pattern is present but the model called the
wrong class" and "the wafer is unlabelable from this map" point at different
fixes. The schema enforces a code on every non-accept decision; this module
defines what the codes mean.

Every code is real vocabulary about wafer maps and model behaviour. None of them
encode a disposition -- scrap, hold, rework -- because that decision belongs to
the engineer and this console does not take it.
"""

from __future__ import annotations

from dataclasses import dataclass

from yieldloop.db.enums import DecisionAction, TaskGate


@dataclass(frozen=True, slots=True)
class ReasonCodeSpec:
    code: str
    label: str
    description: str
    applies_to: tuple[DecisionAction, ...]
    applies_to_gates: tuple[TaskGate, ...]
    sort_order: int


_CLASSIFIER_GATES = (TaskGate.LABEL, TaskGate.CONFIRM)
_EDIT_OR_REJECT = (DecisionAction.EDIT, DecisionAction.REJECT)

REASON_CODES: tuple[ReasonCodeSpec, ...] = (
    ReasonCodeSpec(
        code="wrong_class",
        label="Wrong class",
        description=(
            "A defect pattern is present but the model named the wrong one. The most "
            "informative correction: the signal was there and was misread."
        ),
        applies_to=_EDIT_OR_REJECT,
        applies_to_gates=_CLASSIFIER_GATES,
        sort_order=10,
    ),
    ReasonCodeSpec(
        code="no_pattern_present",
        label="No pattern present",
        description="The model claimed a pattern on a wafer that shows none.",
        applies_to=_EDIT_OR_REJECT,
        applies_to_gates=_CLASSIFIER_GATES,
        sort_order=20,
    ),
    ReasonCodeSpec(
        code="pattern_missed",
        label="Pattern missed",
        description="The model called this clean but a pattern is present.",
        applies_to=_EDIT_OR_REJECT,
        applies_to_gates=_CLASSIFIER_GATES,
        sort_order=30,
    ),
    ReasonCodeSpec(
        code="mixed_pattern",
        label="Mixed pattern",
        description=(
            "More than one pattern is present. The single-label schema cannot express "
            "it, and the chosen label is the dominant one."
        ),
        applies_to=_EDIT_OR_REJECT,
        applies_to_gates=_CLASSIFIER_GATES,
        sort_order=40,
    ),
    ReasonCodeSpec(
        code="boundary_case",
        label="Boundary case",
        description=(
            "The pattern sits between two classes and either label is defensible. "
            "Genuine ambiguity, not a model error."
        ),
        applies_to=_EDIT_OR_REJECT,
        applies_to_gates=_CLASSIFIER_GATES,
        sort_order=50,
    ),
    ReasonCodeSpec(
        code="map_quality",
        label="Map not readable",
        description=(
            "Too few die, or the map is too sparse to judge. The wafer is not "
            "labelable rather than being labeled wrongly."
        ),
        applies_to=_EDIT_OR_REJECT,
        applies_to_gates=_CLASSIFIER_GATES,
        sort_order=60,
    ),
    ReasonCodeSpec(
        code="hypothesis_unsupported",
        label="Evidence does not support it",
        description=(
            "The cited evidence is real but does not support the claim made from it."
        ),
        applies_to=_EDIT_OR_REJECT,
        applies_to_gates=(TaskGate.ESCALATION,),
        sort_order=70,
    ),
    ReasonCodeSpec(
        code="hypothesis_implausible",
        label="Mechanism implausible",
        description=(
            "The evidence supports the claim but the proposed physical mechanism does "
            "not fit the signature."
        ),
        applies_to=_EDIT_OR_REJECT,
        applies_to_gates=(TaskGate.ESCALATION,),
        sort_order=80,
    ),
    ReasonCodeSpec(
        code="hypothesis_known_cause",
        label="Cause already known",
        description="The real cause is already established and differs from this.",
        applies_to=_EDIT_OR_REJECT,
        applies_to_gates=(TaskGate.ESCALATION,),
        sort_order=90,
    ),
    ReasonCodeSpec(
        code="insufficient_context",
        label="Not enough context",
        description=(
            "The retrieved evidence is too thin to judge. Points at retrieval rather "
            "than at the agent."
        ),
        applies_to=_EDIT_OR_REJECT,
        applies_to_gates=(TaskGate.ESCALATION,),
        sort_order=100,
    ),
)

REASON_CODES_BY_CODE: dict[str, ReasonCodeSpec] = {spec.code: spec for spec in REASON_CODES}


def codes_for(gate: TaskGate, action: DecisionAction) -> list[ReasonCodeSpec]:
    """Codes valid for a gate and action, in display order."""
    return sorted(
        (
            spec
            for spec in REASON_CODES
            if gate in spec.applies_to_gates and action in spec.applies_to
        ),
        key=lambda spec: spec.sort_order,
    )


def is_valid(code: str, gate: TaskGate, action: DecisionAction) -> bool:
    """Whether ``code`` may be used for this gate and action."""
    spec = REASON_CODES_BY_CODE.get(code)
    if spec is None:
        return False
    return gate in spec.applies_to_gates and action in spec.applies_to
