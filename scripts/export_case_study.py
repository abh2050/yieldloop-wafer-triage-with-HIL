"""Export the case study: metrics, provenance, and the two curves.

Produces a single markdown document that can be read by someone who has not seen
the repository. It draws only on what the system actually recorded -- artifact
provenance from the registry, metrics from the eval harness, decision counts from
the database -- so it cannot claim anything the system cannot substantiate.

Where a number is unavailable it says so. A case study that quietly omits the
metrics that did not land is a sales document, not an evaluation.

Usage::

    python -m scripts.export_case_study --output case_study.md

Run as a module, not as a path: it imports the eval package from the repository
root, which is not on the path when the file is executed directly.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.harness import run as run_harness
from eval.report import render as render_eval
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from yieldloop.config import Settings, get_settings
from yieldloop.db.enums import ArtifactKind
from yieldloop.db.models import (
    AuditRecord,
    Decision,
    GuardrailAction,
    HistoricalExcursion,
    HypothesisRequest,
    Lot,
    ModelArtifact,
    ProcessEvent,
    Wafer,
)
from yieldloop.db.session import build_engine
from yieldloop.guardrails.audit import AuditLog
from yieldloop.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _counts(session: Session) -> dict[str, int]:
    def count(model: Any) -> int:
        return int(session.execute(select(func.count()).select_from(model)).scalar_one())

    return {
        "lots": count(Lot),
        "wafers": count(Wafer),
        "labeled_wafers": int(
            session.execute(
                select(func.count()).select_from(Wafer).where(Wafer.dataset_label.is_not(None))
            ).scalar_one()
        ),
        "decisions": count(Decision),
        "hypothesis_requests": count(HypothesisRequest),
        "guardrail_actions": count(GuardrailAction),
        "audit_records": count(AuditRecord),
        "process_events": count(ProcessEvent),
        "historical_excursions": count(HistoricalExcursion),
    }


def _artifacts(session: Session) -> list[dict[str, Any]]:
    rows = (
        session.execute(
            select(ModelArtifact)
            .where(ModelArtifact.kind == ArtifactKind.CLASSIFIER)
            .order_by(ModelArtifact.label_count)
        )
        .scalars()
        .all()
    )
    return [
        {
            "content_hash": row.content_hash[:12],
            "data_hash": row.data_hash[:12],
            "git_commit": row.git_commit[:12],
            "git_dirty": row.git_dirty,
            "seed": row.seed,
            "label_count": row.label_count,
            "is_active": row.is_active,
            "val_macro_f1": row.metrics.get("val_macro_f1"),
            "val_accuracy": row.metrics.get("val_accuracy"),
        }
        for row in rows
    ]


def _guardrail_activity(session: Session) -> list[dict[str, Any]]:
    rows = session.execute(
        select(GuardrailAction.stage, GuardrailAction.outcome, func.count())
        .group_by(GuardrailAction.stage, GuardrailAction.outcome)
        .order_by(func.count().desc())
    ).all()
    return [
        {"stage": stage.value, "outcome": outcome.value, "count": int(count)}
        for stage, outcome, count in rows
    ]


def _number(value: Any) -> str:
    """Render a metric, or an em dash when it was never recorded."""
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def render(
    counts: dict[str, int],
    artifacts: list[dict[str, Any]],
    guardrails: list[dict[str, Any]],
    chain_intact: bool,
    chain_checked: int,
    settings: Settings,
    eval_markdown: str,
) -> str:
    lines = [
        "# yieldloop case study",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}_",
        "",
        "A wafer map defect triage console with a human decision loop, active ",
        "learning, and a root cause agent that cannot cite evidence it was not given.",
        "",
        "Every figure below is read from what the system recorded. Nothing here is ",
        "estimated, and where a number is unavailable it says so.",
        "",
        "## What is in the system",
        "",
        "| | Count |",
        "| --- | --- |",
        f"| Lots ingested | {counts['lots']:,} |",
        f"| Wafers ingested | {counts['wafers']:,} |",
        f"| Wafers with a human label | {counts['labeled_wafers']:,} |",
        f"| Reviewer decisions captured | {counts['decisions']:,} |",
        f"| Hypothesis requests | {counts['hypothesis_requests']:,} |",
        f"| Guardrail interventions | {counts['guardrail_actions']:,} |",
        f"| Audit records | {counts['audit_records']:,} |",
        f"| Derived process events | {counts['process_events']:,} |",
        f"| Historical excursions indexed | {counts['historical_excursions']:,} |",
        "",
        "All wafer maps and defect labels are real WM811K data. Derived rows carry ",
        "the rule that produced them; see `docs/data_contract.md`.",
        "",
        "## Configuration in force",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Confidence floor | {settings.confidence_floor} |",
        f"| Auto-commit threshold | {settings.auto_commit_threshold} |",
        f"| Grid | {settings.grid_height}x{settings.grid_width} |",
        f"| Partition | {settings.partition_train_fraction:.2f} / "
        f"{settings.partition_val_fraction:.2f} / "
        f"{settings.partition_holdout_fraction:.2f}, lot-keyed |",
        "",
        "The routing bands are configuration rather than constants, which is what ",
        "lets the harness sweep them to produce the tradeoff curve.",
        "",
        "## Model provenance",
        "",
        "Every artifact records the hash of the ordered wafer/label pairs it was fit ",
        "on, the commit that produced it, and whether the tree was dirty. A number in ",
        "this document traces to a run that can be repeated.",
        "",
    ]

    if artifacts:
        lines += [
            "| Labels | Macro F1 | Accuracy | Content | Data | Commit | Clean | Active |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for artifact in artifacts:
            cells = [
                f"{artifact['label_count']:,}",
                _number(artifact["val_macro_f1"]),
                _number(artifact["val_accuracy"]),
                f"`{artifact['content_hash']}`",
                f"`{artifact['data_hash']}`",
                f"`{artifact['git_commit']}`",
                "no" if artifact["git_dirty"] else "yes",
                "yes" if artifact["is_active"] else "",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    else:
        lines += ["_No classifier has been trained yet._", ""]

    lines += [
        "## Guardrail activity",
        "",
        "What the guardrails actually did, as recorded. `blocked` means the layer ",
        "failed closed and nothing reached the reviewer.",
        "",
    ]
    if guardrails:
        lines += ["| Stage | Outcome | Count |", "| --- | --- | --- |"]
        lines += [f"| {row['stage']} | {row['outcome']} | {row['count']:,} |" for row in guardrails]
    else:
        lines.append("_No guardrail interventions recorded yet._")
    lines += [""]

    lines += [
        "## Audit integrity",
        "",
        f"Hash chain over {chain_checked:,} records: **{'intact' if chain_intact else 'BROKEN'}**.",
        "",
        "`audit_records` has UPDATE, DELETE, and TRUNCATE revoked and a trigger that ",
        "raises regardless of privilege. The chain makes tampering detectable even by ",
        "something that bypasses the application entirely.",
        "",
        "---",
        "",
        eval_markdown,
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=Path("case_study.md"))
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="omit the agent suite, which spends tokens",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings)

    suites = (
        {"classifier", "guardrail"}
        if args.skip_agent
        else {
            "classifier",
            "agent",
            "guardrail",
        }
    )
    harness = run_harness(settings, suites=suites, holdout_limit=None, agent_lots=10)

    with Session(build_engine(settings)) as session:
        counts = _counts(session)
        artifacts = _artifacts(session)
        guardrails = _guardrail_activity(session)
        verification = AuditLog(session).verify_chain()

    document = render(
        counts=counts,
        artifacts=artifacts,
        guardrails=guardrails,
        chain_intact=verification.intact,
        chain_checked=verification.checked,
        settings=settings,
        eval_markdown=render_eval(harness.as_dict()),
    )
    args.output.write_text(document)
    print(f"case study written to {args.output}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "counts": counts,
                    "artifacts": artifacts,
                    "guardrails": guardrails,
                    "audit_chain": {
                        "checked": verification.checked,
                        "intact": verification.intact,
                    },
                    "eval": harness.as_dict(),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"json written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
