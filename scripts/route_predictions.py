"""Route existing predictions to the confirm and escalation gates.

The production path, distinct from an active learning round. A round asks *which
unlabeled wafers should a human annotate*; this asks *which already-predicted
wafers should a human check before the system acts on them*.

Auto-commit is a decision not to ask, so it happens here by omission: anything at
or above the threshold produces no task. Everything below becomes a confirm-gate
task whose visibility follows the routing band, so predictions below the floor
reach the reviewer with the model's guess withheld.

Lots with several confident non-`none` predictions are escalated for root cause
review, because one odd wafer is noise while a pattern across a lot is a
lot-level cause worth asking about.

Usage::

    python scripts/route_predictions.py
    python scripts/route_predictions.py --limit 500 --no-escalate
"""

from __future__ import annotations

import argparse

from sqlalchemy.orm import Session

from yieldloop.config import get_settings
from yieldloop.db.session import build_engine
from yieldloop.guardrails.thresholds import RoutingBands
from yieldloop.logging import configure_logging, get_logger
from yieldloop.review.queue import enqueue_confirmations, enqueue_escalations

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, default=200, help="predictions to consider")
    parser.add_argument("--escalation-limit", type=int, default=25)
    parser.add_argument(
        "--min-wafers",
        type=int,
        default=2,
        help="flagged wafers before a lot is escalated",
    )
    parser.add_argument("--no-escalate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings)
    bands = RoutingBands.from_settings(settings)

    with Session(build_engine(settings)) as session:
        outcome = enqueue_confirmations(session, bands=bands, limit=args.limit)
        escalated = (
            0
            if args.no_escalate
            else enqueue_escalations(
                session,
                bands=bands,
                limit=args.escalation_limit,
                min_wafers=args.min_wafers,
            )
        )
        session.commit()

    logger.info(
        "routing_complete",
        considered=outcome.considered,
        auto_committed=outcome.auto_committed,
        queued=outcome.queued_for_confirmation,
        already_queued=outcome.already_queued,
        escalated=escalated,
        automation_rate=round(outcome.automation_rate, 4),
    )
    print(
        f"considered {outcome.considered:,} predictions: "
        f"{outcome.auto_committed:,} auto-committed ({outcome.automation_rate:.1%}), "
        f"{outcome.queued_for_confirmation:,} queued for confirmation, "
        f"{escalated:,} lots escalated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
