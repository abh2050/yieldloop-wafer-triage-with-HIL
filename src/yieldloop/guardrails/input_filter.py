"""Input validation, applied before a single token is spent.

Everything here runs ahead of retrieval and ahead of the model call. The ordering
is the point: an oversized note or an out-of-window lot should cost nothing, and
validating after assembling a context bundle would mean paying for embeddings and
prompt tokens to discover something a regex could have told us.

Rejections are typed and carry a stable reason code, so the API can translate
them and the audit log can aggregate them without parsing English.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final

from yieldloop.config import Settings
from yieldloop.guardrails import InputRejectedError

#: Wafer identifiers are ``{lot_name}-{wafer_index}``; lot names in WM811K are
#: alphanumeric with occasional separators. Anchored, bounded, and deliberately
#: narrower than the column width so a value that reaches the database has
#: already been shown to be well formed.
WAFER_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.]{1,64}-\d{1,6}$")
LOT_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.]{1,64}$")
REVIEWER_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._@-]{1,128}$")

#: Control characters are disallowed in free text. Tab, newline, and carriage
#: return are kept: engineers write multi-line notes, and stripping newlines
#: would mangle legitimate content to no security benefit, since isolation rather
#: than sanitization is what protects the prompt.
_ALLOWED_CONTROL: Final[frozenset[str]] = frozenset({"\t", "\n", "\r"})


@dataclass(frozen=True, slots=True)
class GridBounds:
    """Permitted normalized grid dimensions."""

    height: int
    width: int


def validate_wafer_id(value: str) -> str:
    """Validate a wafer identifier."""
    candidate = value.strip()
    if not WAFER_ID_PATTERN.match(candidate):
        raise InputRejectedError(
            f"malformed wafer id {value!r}; expected {{lot_name}}-{{wafer_index}}",
            detail={"field": "wafer_id", "code": "malformed_wafer_id"},
        )
    return candidate


def validate_lot_name(value: str) -> str:
    """Validate a lot name."""
    candidate = value.strip()
    if not LOT_NAME_PATTERN.match(candidate):
        raise InputRejectedError(
            f"malformed lot name {value!r}",
            detail={"field": "lot_name", "code": "malformed_lot_name"},
        )
    return candidate


def validate_reviewer_id(value: str) -> str:
    """Validate a reviewer identifier."""
    candidate = value.strip()
    if not REVIEWER_ID_PATTERN.match(candidate):
        raise InputRejectedError(
            f"malformed reviewer id {value!r}",
            detail={"field": "reviewer_id", "code": "malformed_reviewer_id"},
        )
    return candidate


def validate_grid_shape(height: int, width: int, bounds: GridBounds) -> tuple[int, int]:
    """Reject a wafer map whose normalized grid is not the expected shape.

    An unexpected shape means the array did not come through the documented
    normalization path, and feeding it to the classifier would produce a
    confident prediction over the wrong pixels.
    """
    if height != bounds.height or width != bounds.width:
        raise InputRejectedError(
            f"grid is {height}x{width} but the configured normalization is "
            f"{bounds.height}x{bounds.width}",
            detail={
                "field": "grid",
                "code": "grid_shape_mismatch",
                "expected": [bounds.height, bounds.width],
                "actual": [height, width],
            },
        )
    return height, width


def validate_retention_window(
    lot_date: date, *, today: date, retention_days: int
) -> date:
    """Reject a lot outside the retention window.

    A lot older than the window has aged out, and one dated in the future is a
    data error. Both are refused rather than clamped.
    """
    if retention_days < 1:
        raise ValueError(f"retention_days must be positive; got {retention_days}")

    oldest = today - timedelta(days=retention_days)
    if lot_date > today:
        raise InputRejectedError(
            f"lot date {lot_date.isoformat()} is in the future",
            detail={"field": "lot_date", "code": "lot_date_in_future"},
        )
    if lot_date < oldest:
        raise InputRejectedError(
            f"lot date {lot_date.isoformat()} is outside the {retention_days}-day "
            f"retention window starting {oldest.isoformat()}",
            detail={
                "field": "lot_date",
                "code": "lot_outside_retention",
                "oldest_permitted": oldest.isoformat(),
            },
        )
    return lot_date


def validate_free_text(value: str | None, *, field: str, max_chars: int) -> str | None:
    """Validate a reviewer-authored free text field.

    Length is checked on the NFKC-normalized string. Checking the raw string
    would let a payload pass the limit and then expand once normalized, and the
    limit exists to bound prompt cost, which is paid post-normalization.

    The text is **not** modified. Neutralizing it is the job of
    :mod:`yieldloop.guardrails.injection`, which isolates rather than strips.
    """
    if value is None:
        return None
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive; got {max_chars}")

    normalized = unicodedata.normalize("NFKC", value)
    if len(normalized) > max_chars:
        raise InputRejectedError(
            f"{field} is {len(normalized)} characters after normalization, over the "
            f"{max_chars} limit",
            detail={
                "field": field,
                "code": "free_text_too_long",
                "length": len(normalized),
                "limit": max_chars,
            },
        )

    offending = {
        char
        for char in normalized
        if unicodedata.category(char) in {"Cc", "Cf", "Co", "Cs"}
        and char not in _ALLOWED_CONTROL
    }
    if offending:
        raise InputRejectedError(
            f"{field} contains disallowed control characters: "
            f"{sorted(hex(ord(c)) for c in offending)}",
            detail={
                "field": field,
                "code": "disallowed_characters",
                "code_points": sorted(hex(ord(c)) for c in offending),
            },
        )
    return value


@dataclass(frozen=True, slots=True)
class InputFilter:
    """Bundles the settings-derived limits so callers do not re-read config."""

    grid_bounds: GridBounds
    max_free_text_chars: int
    retention_days: int

    @classmethod
    def from_settings(cls, settings: Settings) -> InputFilter:
        return cls(
            grid_bounds=GridBounds(height=settings.grid_height, width=settings.grid_width),
            max_free_text_chars=settings.max_free_text_chars,
            retention_days=settings.lot_retention_days,
        )

    def check_note(self, value: str | None, *, field: str = "note") -> str | None:
        return validate_free_text(value, field=field, max_chars=self.max_free_text_chars)

    def check_grid(self, height: int, width: int) -> tuple[int, int]:
        return validate_grid_shape(height, width, self.grid_bounds)

    def check_lot_date(self, lot_date: date, *, today: date) -> date:
        return validate_retention_window(
            lot_date, today=today, retention_days=self.retention_days
        )
