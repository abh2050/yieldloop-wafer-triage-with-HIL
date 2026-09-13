"""Train, calibrate, and register a classifier.

Reads train and validation only; holdout is touched exclusively by the eval
harness. ``--train-limit`` caps the labeled training set, which is the x-axis of
the label efficiency curve: the same call at different limits produces the
comparable points on it.

Usage::

    python -m scripts.train
    python -m scripts.train --train-limit 4000 --max-epochs 15
"""

from __future__ import annotations

import argparse
import json
import sys

from sqlalchemy.orm import Session

from yieldloop.config import get_settings
from yieldloop.db.session import build_engine
from yieldloop.logging import configure_logging
from yieldloop.models.classifier import ClassifierConfig
from yieldloop.models.registry import describe
from yieldloop.models.train import train_classifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="cap labeled training wafers (a label efficiency curve point)",
    )
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-activate",
        action="store_true",
        help="register the artifact without making it the active one",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings)

    config = ClassifierConfig(
        grid_height=settings.grid_height,
        grid_width=settings.grid_width,
        embedding_dim=settings.embedding_dim,
        seed=args.seed if args.seed is not None else settings.train_seed,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
    )

    with Session(build_engine(settings)) as session:
        try:
            result = train_classifier(
                session,
                settings,
                config=config,
                train_limit=args.train_limit,
                activate=not args.no_activate,
            )
        except ValueError as exc:
            print(f"{exc}", file=sys.stderr)
            return 1
        session.commit()
        provenance = describe(result.artifact.record)

    print(json.dumps(result.metrics, indent=2, sort_keys=True))
    print(f"\nartifact: {provenance}")

    if result.calibration.at_bound:
        print(
            "\nWARNING: temperature scaling hit its bound. The confidences from this "
            "artifact are not trustworthy and the routing thresholds must not be tuned "
            "against it.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
