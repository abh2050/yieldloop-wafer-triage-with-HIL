"""Fetch the real WM811K wafer map dataset.

yieldloop trains and evaluates on real wafer maps only. This script is the sole
sanctioned way the dataset enters the repository. It downloads the archive from
Kaggle, extracts ``LSWMD.pkl``, and records a sha256 that every subsequent load
verifies, so a silently swapped or truncated file surfaces as a hard failure
rather than as a quiet change in the metrics.

Credentials: the Kaggle API reads ``~/.kaggle/kaggle.json``. Create one at
Kaggle -> Settings -> API -> Create New API Token. Nothing here falls back to a
generated dataset if credentials are missing; it fails and says what to do.

Usage::

    python scripts/fetch_dataset.py            # download if absent, then verify
    python scripts/fetch_dataset.py --force    # re-download even if present
    python scripts/fetch_dataset.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

from yieldloop.config import Settings, get_settings
from yieldloop.logging import configure_logging, get_logger

logger = get_logger(__name__)

_READ_CHUNK = 1024 * 1024

KAGGLE_CREDENTIAL_HELP = """
Kaggle credentials were not found.

  1. Sign in at https://www.kaggle.com and open Settings -> API.
  2. Click "Create New API Token". A kaggle.json file downloads.
  3. Move it into place and restrict its permissions:

       mkdir -p ~/.kaggle
       mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
       chmod 600 ~/.kaggle/kaggle.json

  4. Accept the dataset terms once, in a browser, at
     https://www.kaggle.com/datasets/{dataset}

Alternatively set KAGGLE_USERNAME and KAGGLE_KEY in the environment.
""".strip()


class DatasetFetchError(RuntimeError):
    """Raised when the real dataset cannot be obtained or verified."""


def sha256_of(path: Path) -> str:
    """Return the hex sha256 of ``path``, streamed so large files are safe."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def credentials_present() -> bool:
    """Return True when the Kaggle API has credentials available to it."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    candidate = (
        Path(config_dir) / "kaggle.json" if config_dir else Path.home() / ".kaggle" / "kaggle.json"
    )
    return candidate.is_file()


def download_archive(settings: Settings, destination: Path) -> None:
    """Download and unpack the Kaggle dataset into ``destination``.

    The Kaggle client authenticates at import time, so it is imported here rather
    than at module scope; that keeps ``--verify-only`` usable without credentials.
    """
    if not credentials_present():
        raise DatasetFetchError(KAGGLE_CREDENTIAL_HELP.format(dataset=settings.kaggle_dataset))

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    logger.info("dataset_download_start", dataset=settings.kaggle_dataset, dest=str(destination))
    api.dataset_download_files(
        settings.kaggle_dataset, path=str(destination), unzip=True, quiet=False
    )

    target = destination / settings.wm811k_filename
    if target.is_file():
        return

    # Some Kaggle mirrors nest the payload; unpack any archive that contains it.
    for archive in sorted(destination.glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            if settings.wm811k_filename in names:
                zf.extract(settings.wm811k_filename, path=destination)
                break
    if not target.is_file():
        found = sorted(p.name for p in destination.iterdir())
        raise DatasetFetchError(
            f"{settings.wm811k_filename} was not present after downloading "
            f"{settings.kaggle_dataset}. Files in {destination}: {found}"
        )


def verify(settings: Settings, path: Path) -> str:
    """Verify ``path`` against the configured sha256 and return the digest.

    When no expected digest is configured, the computed one is returned so the
    operator can pin it. When one is configured and does not match, this raises:
    a dataset that has changed underneath a set of published metrics is a
    correctness problem, not a warning.
    """
    if not path.is_file():
        raise DatasetFetchError(
            f"{path} does not exist. Run `python scripts/fetch_dataset.py` first."
        )
    actual = sha256_of(path)
    expected = settings.wm811k_sha256.strip().lower()
    if expected and actual != expected:
        raise DatasetFetchError(
            f"sha256 mismatch for {path}.\n"
            f"  expected (YIELDLOOP_WM811K_SHA256): {expected}\n"
            f"  actual:                             {actual}\n"
            "Refusing to proceed: metrics computed from a different file are not "
            "comparable to the committed baselines."
        )
    return actual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the file is present"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="verify the local file without downloading"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings)

    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    target = settings.wm811k_path

    try:
        if args.verify_only:
            pass
        elif args.force or not target.is_file():
            download_archive(settings, data_dir)
        else:
            logger.info("dataset_present", path=str(target))
        digest = verify(settings, target)
    except DatasetFetchError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    size_mb = target.stat().st_size / (1024 * 1024)
    logger.info("dataset_ready", path=str(target), size_mb=round(size_mb, 1), sha256=digest)

    if not settings.wm811k_sha256.strip():
        print(
            "\nRecord this digest so future loads are verified:\n"
            f"  YIELDLOOP_WM811K_SHA256={digest}\n",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
