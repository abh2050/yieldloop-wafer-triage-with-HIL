"""Markdown report generation. This file produces the case study artifact.

The report addresses a reader who was not in the room, so every table states
what the number means and what it should be compared against. Accuracy never
appears without the majority-class baseline beside it, because 85% of labeled
WM811K wafers are `none` and an unqualified accuracy figure misleads the reader.
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

    efficiency = result.get("label_efficiency")
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
                    "Misleads on its own, because 85% of labels are `none`",
                ],
                [
                    "Macro F1",
                    f"{holdout['macro_f1']:.4f}",
                    f"{baseline['macro_f1']:.4f}",
                    "Weights every class equally, so rare classes count",
                ],
                [
                    "Balanced accuracy",
                    _percent(holdout["balanced_accuracy"]),
                    _percent(baseline["balanced_accuracy"]),
                    "Averages recall across classes with equal weight",
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
        "The last row bears on whether automation is safe. It measures only the band ",
        "that auto-commit governs rather than averaging across a range most predictions ",
        "never reach. Max calibration error ignores bins holding fewer than 30 ",
        "predictions, because a two-sample bin admits accuracies of only 0, 0.5 or 1 ",
        "and its apparent gap carries noise.",
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
        "",
        _label_efficiency_section(efficiency),
    ]
    return "\n".join(lines)


def _label_efficiency_section(efficiency: dict[str, Any] | None) -> str:
    """Chart 2 of 2: active learning against random, at matched label counts."""
    if efficiency is None:
        return (
            "### Label efficiency\n\n"
            "**Chart 2 of 2. Not yet run.** The curve requires training both arms at "
            "several budgets, so it is produced by `python -m scripts.run_label_efficiency` "
            "rather than recomputed here. Reported as absent rather than approximated: a "
            "cheaper estimate would not be the quantity the curve claims to measure.\n"
        )

    comparison = efficiency["comparison"]
    active = {p["label_count"]: p for p in efficiency["active"]}
    random_arm = {p["label_count"]: p for p in efficiency["random"]}

    rare_measurable = efficiency.get("rare_class_recall_measurable", False)
    rows = []
    for budget in sorted(set(active) & set(random_arm)):
        a, r = active[budget], random_arm[budget]
        delta = a["macro_f1"] - r["macro_f1"]
        row = [f"{budget:,}", f"{a['macro_f1']:.4f}", f"{r['macro_f1']:.4f}", f"{delta:+.4f}"]
        if rare_measurable:
            row += [f"{a['rare_class_recall']:.3f}", f"{r['rare_class_recall']:.3f}"]
        rows.append(row)

    headers = ["Labels", "Macro F1 (active)", "Macro F1 (random)", "Delta"]
    if rare_measurable:
        headers += ["Rare recall (active)", "Rare recall (random)"]

    saving = comparison.get("label_saving_ratio")
    reached = (
        f"{comparison['strategy_labels_to_target']:,}"
        if comparison["strategy_labels_to_target"]
        else "never"
    )
    baseline_reached = (
        f"{comparison['baseline_labels_to_target']:,}"
        if comparison["baseline_labels_to_target"]
        else "never"
    )

    lines = [
        "### Label efficiency",
        "",
        "**Chart 2 of 2.** Entropy-plus-diversity selection against random, compared ",
        "only at matched label counts. Both arms use the same splits, hyperparameters ",
        "and seed and differ only in which wafers were chosen.",
        "",
        "The simulation runs the active arm iteratively. The selector at each step ",
        "trains only on what it has acquired so far. Selecting in one shot with a model ",
        "that had seen the whole dataset would leak the dataset into the selection, and ",
        "that leak is how this experiment usually gets reported wrongly.",
        "",
        _table(headers, rows),
        "",
        _table(
            ["Summary", "Value"],
            [
                ["Mean macro F1 delta", f"{comparison['mean_delta']:+.4f}"],
                [f"Labels to reach macro F1 {comparison['target']}", f"active {reached}"],
                ["", f"random {baseline_reached}"],
                [
                    "Label saving",
                    f"{saving:.2f}x" if saving else "not reached by both arms",
                ],
            ],
        ),
    ]
    if not rare_measurable:
        support = efficiency.get("rare_class_support", 0)
        minimum = efficiency.get("min_support_for_recall", 30)
        lines += [
            "",
            f"> This report **omits** rare-class recall and says so here. The smallest "
            f"rare class has {support} wafers in this evaluation set, below the {minimum} "
            "a recall figure needs to carry meaning. At that size recall takes only a few "
            "discrete values, so movement between the arms reflects quantization rather "
            "than signal, and an average is only as trustworthy as its weakest term. "
            "The table above reports per-class recall on the full holdout.",
        ]

    first = min(set(active) & set(random_arm), default=None)
    if first is not None and abs(active[first]["macro_f1"] - random_arm[first]["macro_f1"]) < 1e-9:
        lines += [
            "",
            f"> The two arms are **identical at {first:,} labels** by construction, not by "
            "coincidence: active learning has no model to select with until it has labels, "
            "so it starts from a random seed set of that size. The matching scores confirm "
            "the harness is comparing what it claims. That point contributes a zero to the "
            "mean delta, so the mean understates the effect; the comparison begins at the "
            "second budget.",
        ]

    if comparison["strategy_labels_to_target"] and not comparison["baseline_labels_to_target"]:
        largest = max(set(active) & set(random_arm))
        lines += [
            "",
            f"> Random never reached macro F1 {comparison['target']} within "
            f"{largest:,} labels, so an exact saving ratio cannot be computed. The "
            f"measurable statement is a lower bound: active reached it at "
            f"{comparison['strategy_labels_to_target']:,}, so the saving is **at least "
            f"{largest / comparison['strategy_labels_to_target']:.1f}x**, and the true "
            "figure requires extending the random arm.",
        ]

    if comparison["mean_delta"] <= 0:
        lines += [
            "",
            "> The active arm did **not** beat random at these budgets. Reported as "
            "measured. Active learning commonly underperforms at small budgets, where "
            "the selector is trained on too little to rank informativeness well -- the "
            "cold-start problem. The honest reading is that the benefit, if any, appears "
            "at larger budgets than these.",
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
        "Grounding rate reports the fraction of proposed hypotheses whose citations all ",
        "resolved to evidence in the context bundle. A rate below 100% means the model ",
        "attempted a fabrication and the gate dropped those claims before any reviewer ",
        "saw them.",
        "",
        "Abstention rate stays deliberately **unminimized**. Abstaining on a thin bundle ",
        "is the correct answer, and a rate of zero against sparse evidence would mean the ",
        "model is inventing support.",
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
            else "not measurable, because one regime has no decisions",
        ],
        ["Median decision time", f"{result['median_decision_seconds']:.2f} s"],
        ["p95 decision time", f"{result['p95_decision_ms'] / 1000:.2f} s"],
    ]
    lines = [
        "## Reviewer agreement",
        "",
        "Override rate alone reads two ways. A low rate can mean the model is good, or ",
        "it can mean reviewers are deferring to it. Splitting the rate by whether the ",
        "console showed the prediction separates the two, so every decision records what ",
        "the reviewer could see.",
        "",
        "A large positive anchoring delta means reviewers disagreed far more often when ",
        "they could not see the prediction. Visible predictions are then buying agreement ",
        "rather than earning it, and the confidence floor should rise.",
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
        "The suite asserts a property rather than a rate. The layer must fail closed on ",
        "every hostile input, so a single pass-through fails the suite regardless of how ",
        "many other cases it handled. The suite runs without a database or an API key, so ",
        "it gates every pull request.",
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
        "# Yieldloop evaluation report",
        "",
        "Every metric here comes from the real WM811K dataset and from decisions real ",
        "reviewers made in this console. This report contains no synthetic ground truth.",
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
