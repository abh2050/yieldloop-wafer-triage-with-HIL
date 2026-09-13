"""The evaluation harness. A runnable program, not a notebook.

Runs the suites, compares against the committed metric floors in
``baselines.json``, and exits non-zero on a regression. That exit code is the
whole point: a metric that is only ever looked at by a human is a metric that
drifts.

Usage::

    python -m eval.harness                    # everything available
    python -m eval.harness --suite guardrail  # no database or key needed
    python -m eval.harness --update-baselines # after a deliberate improvement
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from eval.report import write_report
from eval.suites import agent_suite, classifier_suite, guardrail_suite, review_suite
from yieldloop.config import Settings, get_settings
from yieldloop.db.session import build_engine
from yieldloop.logging import configure_logging, get_logger

logger = get_logger(__name__)

BASELINES_PATH = Path(__file__).parent / "baselines.json"

#: Metrics where a *lower* value is better, so the comparison flips.
LOWER_IS_BETTER = frozenset(
    {
        "classifier.ece_calibrated",
        "classifier.mce_calibrated",
        "classifier.escaped_error_rate",
        "agent.fabrication_rate",
    }
)


@dataclass(slots=True)
class Regression:
    metric: str
    baseline: float
    observed: float
    tolerance: float

    def describe(self) -> str:
        direction = "below" if self.metric not in LOWER_IS_BETTER else "above"
        return (
            f"{self.metric}: {self.observed:.4f} is {direction} the committed floor "
            f"{self.baseline:.4f} by more than the {self.tolerance:.4f} tolerance"
        )


@dataclass(slots=True)
class HarnessResult:
    results: dict[str, Any] = field(default_factory=dict)
    regressions: list[Regression] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.regressions

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": self.results,
            "regressions": [
                {
                    "metric": r.metric,
                    "baseline": r.baseline,
                    "observed": r.observed,
                    "tolerance": r.tolerance,
                }
                for r in self.regressions
            ],
            "skipped": self.skipped,
            "passed": self.passed,
            "seconds": self.seconds,
        }


def load_baselines() -> dict[str, Any]:
    if not BASELINES_PATH.is_file():
        return {"metrics": {}, "tolerance": 0.02}
    payload: dict[str, Any] = json.loads(BASELINES_PATH.read_text())
    return payload


def extract_metrics(results: dict[str, Any]) -> dict[str, float]:
    """Flatten the metrics the gate compares."""
    flat: dict[str, float] = {}

    classifier = results.get("classifier")
    if classifier:
        holdout = classifier["holdout"]
        flat["classifier.accuracy"] = float(holdout["accuracy"])
        flat["classifier.macro_f1"] = float(holdout["macro_f1"])
        flat["classifier.balanced_accuracy"] = float(holdout["balanced_accuracy"])
        flat["classifier.ece_calibrated"] = float(classifier["calibration"]["ece_calibrated"])
        flat["classifier.mce_calibrated"] = float(classifier["calibration"]["mce_calibrated"])
        flat["classifier.escaped_error_rate"] = float(
            classifier["escalation"]["escaped_error_rate"]
        )
        for label, metrics in holdout["per_class"].items():
            flat[f"classifier.recall.{label}"] = float(metrics["recall"])

    agent = results.get("agent")
    if agent:
        flat["agent.grounding_rate"] = float(agent["grounding"]["grounding_rate"])
        flat["agent.fabrication_rate"] = float(agent["grounding"]["fabrication_rate"])
        if agent.get("precision_at_1") is not None:
            flat["agent.precision_at_1"] = float(agent["precision_at_1"])
        if agent.get("precision_at_3") is not None:
            flat["agent.precision_at_3"] = float(agent["precision_at_3"])

    review = results.get("review")
    if review and not review.get("thin_sample", True):
        flat["review.override_rate"] = float(review["override_rate"])
        flat["review.median_decision_seconds"] = float(review["median_decision_seconds"])
        if review.get("anchoring_measurable"):
            flat["review.anchoring_delta"] = float(review["anchoring_delta"])

    guardrail = results.get("guardrail")
    if guardrail:
        # A property, not a rate: any failure is a regression.
        flat["guardrail.passed"] = 1.0 if guardrail["passed"] else 0.0

    return flat


def compare_to_baselines(observed: dict[str, float], baselines: dict[str, Any]) -> list[Regression]:
    """Find metrics that regressed beyond tolerance."""
    tolerance = float(baselines.get("tolerance", 0.02))
    floors: dict[str, float] = baselines.get("metrics", {})
    regressions: list[Regression] = []

    for metric, floor in floors.items():
        if metric not in observed:
            # A metric that could not be measured this run is not a regression.
            continue
        value = observed[metric]
        if metric in LOWER_IS_BETTER:
            if value > float(floor) + tolerance:
                regressions.append(Regression(metric, float(floor), value, tolerance))
        elif value < float(floor) - tolerance:
            regressions.append(Regression(metric, float(floor), value, tolerance))
    return regressions


def run(
    settings: Settings, *, suites: set[str], holdout_limit: int | None, agent_lots: int
) -> HarnessResult:
    outcome = HarnessResult()
    started = time.monotonic()

    if "guardrail" in suites:
        guardrail = guardrail_suite.run()
        outcome.results["guardrail"] = guardrail.as_dict()

    needs_database = bool({"classifier", "agent", "review"} & suites)
    if needs_database:
        engine = build_engine(settings)
        with Session(engine) as session:
            if "classifier" in suites:
                result = classifier_suite.run(session, settings, limit=holdout_limit)
                if result is None:
                    outcome.skipped.append(
                        "classifier: no active classifier or no labeled holdout wafers"
                    )
                else:
                    outcome.results["classifier"] = result.as_dict()

            if "review" in suites:
                review = review_suite.run(session)
                if review is None:
                    outcome.skipped.append(
                        "review: no reviewer decisions captured by the console yet"
                    )
                else:
                    outcome.results["review"] = review.as_dict()

            if "agent" in suites:
                agent = agent_suite.run(session, settings, max_lots=agent_lots)
                if agent is None:
                    outcome.skipped.append("agent: OPENAI_API_KEY is not set")
                else:
                    outcome.results["agent"] = agent.as_dict()

    observed = extract_metrics(outcome.results)
    outcome.regressions = compare_to_baselines(observed, load_baselines())
    outcome.seconds = time.monotonic() - started
    return outcome


def update_baselines(observed: dict[str, float], tolerance: float) -> None:
    """Rewrite the committed floors from the current run."""
    BASELINES_PATH.write_text(
        json.dumps(
            {
                "tolerance": tolerance,
                "note": (
                    "Committed metric floors. The eval gate fails a pull request when a "
                    "metric regresses beyond the tolerance. Update deliberately, with the "
                    "reason in the commit message."
                ),
                "metrics": {k: round(v, 6) for k, v in sorted(observed.items())},
            },
            indent=2,
        )
        + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=["classifier", "agent", "guardrail", "review"],
        help="run only these suites (repeatable); default is all",
    )
    parser.add_argument("--holdout-limit", type=int, default=None)
    parser.add_argument("--agent-lots", type=int, default=10)
    parser.add_argument("--report", type=Path, default=Path("eval_report.md"))
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--update-baselines", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings)

    suites = set(args.suite) if args.suite else {"classifier", "agent", "guardrail", "review"}
    outcome = run(
        settings,
        suites=suites,
        holdout_limit=args.holdout_limit,
        agent_lots=args.agent_lots,
    )

    if args.update_baselines:
        update_baselines(extract_metrics(outcome.results), load_baselines().get("tolerance", 0.02))
        print(f"baselines updated at {BASELINES_PATH}")

    write_report(outcome.as_dict(), args.report)
    print(f"report written to {args.report}")

    if args.json:
        args.json.write_text(json.dumps(outcome.as_dict(), indent=2) + "\n")
        print(f"json written to {args.json}")

    for skipped in outcome.skipped:
        print(f"skipped: {skipped}", file=sys.stderr)

    if outcome.regressions:
        print("\nREGRESSIONS:", file=sys.stderr)
        for regression in outcome.regressions:
            print(f"  - {regression.describe()}", file=sys.stderr)
        return 1

    print("no regressions against committed baselines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
