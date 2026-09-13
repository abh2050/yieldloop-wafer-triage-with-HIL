"""Markdown report generation. This file is the case study artifact.

Written to be read by someone who was not in the room: every table says what the
number means and what it should be compared against. In particular accuracy is
never printed without the majority-class baseline beside it, because 85% of
labeled WM811K wafers are `none` and an unqualified accuracy figure here is
actively misleading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No data._\n"
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return f"{head}\n{rule}\n{body}\n"


def _classifier_section(result: dict[str, Any]) -> str:
    holdout = result["holdout"]
    baseline = result["majority_class_baseline"]
    calibration = result["calibration"]
    escalation = result["escalation"]

    curve = result["routing_curve"]
    floor = curve[0]["confidence_floor"] if curve else "n/a"

    lines = [
        "## Classifier",
        "",
        f"Evaluated on {holdout['samples']:,} labeled wafers from the **holdout** split. ",
        "Holdout is read only here; training fits on train and the temperature fits on ",
        "validation, and because partitioning is lot-keyed these are lots the model has ",
        "never seen.",
        "",
        _table(
            ["Metric", "Model", "Majority-class baseline", "Reading"],
            [
                [
                    "Accuracy",
                    _percent(holdout["accuracy"]),
                    _percent(baseline["accuracy"]),
                    "Misleading alone: 85% of labels are `none`",
                ],
                [
                    "Macro F1",
                    f"{holdout['macro_f1']:.4f}",
                    f"{baseline['macro_f1']:.4f}",
                    "The number that separates a useful model from a constant one",
                ],
                [
                    "Balanced accuracy",
                    _percent(holdout["balanced_accuracy"]),
                    _percent(baseline["balanced_accuracy"]),
                    "Macro recall; equal weight per class",
                ],
            ],
        ),
        "",
        "### Per-class recall",
        "",
        "Reported separately because the class distribution spans three orders of ",
        "magnitude. A model can score well on accuracy while never predicting ",
        "`near_full` at all.",
        "",
        _table(
            ["Class", "Precision", "Recall", "F1", "Support"],
            [
                [
                    label,
                    f"{m['precision']:.3f}",
                    f"{m['recall']:.3f}",
                    f"{m['f1']:.3f}",
                    f"{m['support']:,}",
                ]
                for label, m in sorted(
                    holdout["per_class"].items(), key=lambda kv: -kv[1]["support"]
                )
            ],
        ),
        "",
        "### Calibration",
        "",
        "Every routing decision compares a confidence against a threshold, so those ",
        "confidences have to mean something. Temperature scaling cannot change which ",
        "class is predicted, so accuracy above is unaffected and the improvement below ",
        "is real.",
        "",
        "The last row is the one that bears on whether automation is safe: it measures ",
        "only the band auto-commit governs, rather than averaging across a range most ",
        "predictions never reach. Max calibration error ignores bins holding fewer than ",
        "30 predictions, since a two-sample bin admits accuracies of only 0, 0.5 or 1 ",
        "and its apparent gap is noise.",
        "",
        _table(
            ["Metric", "Value"],
            [
                ["Fitted temperature", f"{calibration['temperature']:.4f}"],
                ["ECE before calibration", f"{calibration['ece_uncalibrated']:.4f}"],
                ["ECE after calibration", f"{calibration['ece_calibrated']:.4f}"],
                [
                    "Max calibration error",
                    f"{calibration['mce_calibrated']:.4f} "
                    "(worst bin holding at least 30 predictions)",
                ],
                [
                    "Gap in the auto-commit band",
                    f"{calibration['auto_commit_calibration_error']:.4f} over "
                    f"{calibration['auto_commit_samples']:,} predictions",
                ],
            ],
        ),
        "",
        "### Routing tradeoff",
        "",
        "**Chart 1 of 2.** Sweeping the auto-commit threshold with the confidence floor ",
        f"held at {floor}. ",
        "`escaped errors` are wrong predictions committed without a human ever seeing ",
        "them, which is the column a fab actually cares about.",
        "",
        _table(
            ["Auto-commit", "Automated", "Accuracy when automated", "Escaped errors", "Human load"],
            [
                [
                    f"{point['auto_commit_threshold']:.2f}",
                    _percent(point["automation_rate"]),
                    _percent(point["auto_commit_accuracy"]),
                    _percent(point["escaped_error_rate"]),
                    f"{point['human_load']:,}",
                ]
                for point in result["routing_curve"][::2]
            ],
        ),
        "",
        "### Current configuration",
        "",
        _table(
            ["Metric", "Value"],
            [
                ["Automation rate", _percent(escalation["automation_rate"])],
                ["Escalation rate", _percent(escalation["escalation_rate"])],
                [
                    "Blind review share",
                    f"{_percent(escalation['blind_review_share'])} of human work has the "
                    "prediction withheld",
                ],
                ["Escaped error rate", _percent(escalation["escaped_error_rate"])],
            ],
        ),
    ]
    return "\n".join(lines)


def _agent_section(result: dict[str, Any]) -> str:
    grounding = result["grounding"]
    rows = [
        ["Calls", f"{grounding['calls']:,}"],
        ["Hypotheses proposed", f"{grounding['hypotheses_returned']:,}"],
        ["Hypotheses grounded", f"{grounding['hypotheses_grounded']:,}"],
        ["Grounding rate", _percent(grounding["grounding_rate"])],
        ["Abstention rate", _percent(grounding["abstention_rate"])],
        ["Fabricated citations", f"{grounding['fabricated_citations']:,}"],
        ["Median latency", f"{grounding['median_latency_ms']:.0f} ms"],
        ["p95 latency", f"{grounding['p95_latency_ms']:.0f} ms"],
        ["Cost", f"${result['total_cost_usd']:.4f}"],
    ]
    if result.get("precision_at_1") is not None:
        rows.append(["Precision@1", _percent(result["precision_at_1"])])
        rows.append(["Precision@3", _percent(result["precision_at_3"])])

    note = result.get("precision_note")
    lines = [
        "## Root cause agent",
        "",
        "Grounding rate is the fraction of proposed hypotheses whose citations all ",
        "resolved to evidence in the context bundle. A rate below 100% does not mean a ",
        "reviewer saw something wrong -- the gate dropped those claims -- it means the ",
        "model attempted a fabrication.",
        "",
        "Abstention rate is **not** minimized. Abstaining on a thin bundle is correct; a ",
        "rate of zero against sparse evidence would mean the model is inventing support.",
        "",
        _table(["Metric", "Value"], rows),
    ]
    if note:
        lines += ["", f"> {note}"]
    return "\n".join(lines)


def _review_section(result: dict[str, Any]) -> str:
    rows = [
        ["Decisions captured", f"{result['decisions']:,}"],
        ["With a model prediction to compare", f"{result['with_model_label']:,}"],
        ["Override rate (all)", _percent(result["override_rate"])],
        ["Override rate (prediction shown)", _percent(result["override_rate_shown"])],
        ["Override rate (prediction withheld)", _percent(result["override_rate_blind"])],
        [
            "Anchoring delta",
            f"{result['anchoring_delta'] * 100:+.2f} pp"
            if result.get("anchoring_measurable")
            else "not measurable — one regime has no decisions",
        ],
        ["Median decision time", f"{result['median_decision_seconds']:.2f} s"],
        ["p95 decision time", f"{result['p95_decision_ms'] / 1000:.2f} s"],
    ]
    lines = [
        "## Reviewer agreement",
        "",
        "Override rate alone is ambiguous: a low rate can mean the model is good, or ",
        "that reviewers are deferring to it. Splitting by whether the prediction was ",
        "visible separates the two, which is why every decision records what the ",
        "reviewer could see.",
        "",
        "A large positive anchoring delta -- reviewers disagreeing far more often when ",
        "they could not see the prediction -- means visible predictions are buying ",
        "agreement rather than earning it, and the confidence floor should rise.",
        "",
        _table(["Metric", "Value"], rows),
    ]
    if result.get("thin_sample"):
        lines += [
            "",
            f"> Fewer than {result['min_decisions_for_confidence']} decisions. These "
            "rates are reported for completeness but are not yet a measurement, and the "
            "eval gate does not hold them to a floor.",
        ]
    if result.get("reason_code_counts"):
        lines += [
            "",
            "### Why reviewers disagreed",
            "",
            _table(
                ["Reason", "Count"],
                [
                    [code, str(count)]
                    for code, count in sorted(
                        result["reason_code_counts"].items(), key=lambda kv: -kv[1]
                    )
                ],
            ),
        ]
    return "\n".join(lines)


def _guardrail_section(result: dict[str, Any]) -> str:
    status = "**PASS**" if result["passed"] else "**FAIL**"
    lines = [
        "## Guardrails",
        "",
        f"{result['checks']} adversarial checks: {status}",
        "",
        "The suite asserts a property, not a rate: for every hostile input, the layer ",
        "must fail closed. A single pass-through is a failure regardless of how many ",
        "other cases were handled. It runs without a database or an API key, so it gates ",
        "every pull request.",
        "",
    ]
    if result["failures"]:
        lines += ["### Failures", ""]
        lines += [f"- {failure}" for failure in result["failures"]]
    return "\n".join(lines)


def render(payload: dict[str, Any]) -> str:
    """Render the full markdown report."""
    results = payload["results"]
    sections = [
        "# yieldloop evaluation report",
        "",
        "All metrics are computed from the real WM811K dataset and from decisions real ",
        "reviewers made in this console. There is no synthetic ground truth anywhere in ",
        "this report.",
        "",
    ]

    status = "PASS" if payload["passed"] else "FAIL"
    sections += [
        f"**Gate: {status}** · {payload['seconds']:.1f}s",
        "",
    ]

    if payload["regressions"]:
        sections += ["## Regressions", ""]
        sections += [
            f"- `{r['metric']}`: observed {r['observed']:.4f} against committed floor "
            f"{r['baseline']:.4f} (tolerance {r['tolerance']:.4f})"
            for r in payload["regressions"]
        ]
        sections += [""]

    if payload["skipped"]:
        sections += ["## Not measured", ""]
        sections += [f"- {item}" for item in payload["skipped"]]
        sections += [
            "",
            "_Reported as gaps rather than as zeros: no data is not the same as poor performance._",
            "",
        ]

    if "classifier" in results:
        sections += [_classifier_section(results["classifier"]), ""]
    if "agent" in results:
        sections += [_agent_section(results["agent"]), ""]
    if "review" in results:
        sections += [_review_section(results["review"]), ""]
    if "guardrail" in results:
        sections += [_guardrail_section(results["guardrail"]), ""]

    return "\n".join(sections)


def write_report(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(payload))
