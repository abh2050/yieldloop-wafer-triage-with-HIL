"""Instruction-shaped content in untrusted text.

Two kinds of text reach a prompt and neither is authored by a trusted party:
reviewer free-text notes, and the resolution text on retrieved historical lots.
Either could contain something that reads as an instruction.

The defence here is **structural isolation, not removal**. Stripping matched
substrings is the wrong approach for three reasons: it is trivially evaded by
obfuscation, it silently corrupts legitimate engineering prose (a reviewer
writing "ignore the previous edge measurement" means it literally), and it
destroys evidence that would otherwise show up in the audit log. So the text is
passed through *unmodified* and wrapped in a delimited block that is declared as
data, with any delimiter occurring inside the payload neutralized so the block
cannot be closed early.

Detection is separate from isolation. A detection does not block the request; it
raises ``injection_suspected`` on the response and writes an audit record. The
isolation is what actually protects the prompt, and it applies to every piece of
untrusted text whether or not anything was detected.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

#: Zero-width and bidirectional control characters. These carry no meaning in
#: engineering prose and are a standard way to hide instruction text from a
#: human reviewer while leaving it visible to the model.
_INVISIBLE_CHARS: Final[re.Pattern[str]] = re.compile(r"[­​-‏‪-‮⁠-⁤⁦-⁩﻿]")

#: Patterns that indicate an attempt to address the model rather than describe a
#: wafer. Deliberately conservative: these raise a flag and an audit record, they
#: do not block, so a false positive costs a log line rather than a lost review.
_INSTRUCTION_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "override_instruction",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?"
            r"\b(previous|prior|above|earlier|all|any|your|the)\b[^.\n]{0,40}?"
            r"\b(instruction|prompt|rule|constraint|direction|guideline|context)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_reassignment",
        re.compile(
            r"\b(you are now|act as|pretend to be|roleplay as|from now on,? you|"
            r"your new (role|task|instruction)|new instructions?:)",
            re.IGNORECASE,
        ),
    ),
    (
        "chat_role_marker",
        re.compile(
            r"(^|\n)\s*(system|assistant|user|developer)\s*:",
            re.IGNORECASE,
        ),
    ),
    (
        "chat_template_token",
        re.compile(
            r"(<\|[a-z_]+\|>|\[/?INST\]|<<SYS>>|###\s*(instruction|system|response))",
            re.IGNORECASE,
        ),
    ),
    (
        "output_format_hijack",
        re.compile(
            r"\b(respond|reply|answer|output|return)\b[^.\n]{0,30}?"
            r"\b(only|instead|exactly|verbatim)\b",
            re.IGNORECASE,
        ),
    ),
    (
        # Matches in either order: an attack can name the guardrail first
        # ("grounding is not required") or the negation first ("do not abstain").
        "guardrail_targeting",
        re.compile(
            r"\b(abstention|abstain|grounding|evidence[_ ]id|guardrail|schema|audit)\b"
            r"[^.\n]{0,40}?\b(skip|ignore|disable|bypass|not required|do not|don't|never)\b"
            r"|"
            r"\b(skip|ignore|disable|bypass|do not|don't|never|no need to)\b"
            r"[^.\n]{0,40}?"
            r"\b(abstention|abstain|grounding|guardrail|evidence[_ ]id|schema|audit)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "fabrication_request",
        re.compile(
            r"\b(invent|make up|fabricate|generate a|assume a)\b[^.\n]{0,30}?"
            r"\b(evidence|evidence_id|tool|chamber|recipe|timestamp|lot)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "exfiltration",
        re.compile(
            r"\b(reveal|print|repeat|show|output)\b[^.\n]{0,30}?"
            r"\b(system prompt|your (instructions|prompt|rules)|api[_ ]?key)\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class InjectionScan:
    """The result of scanning one piece of untrusted text."""

    #: Names of the patterns that matched. Empty means nothing was detected.
    signals: tuple[str, ...]
    #: True when invisible or bidirectional control characters were present.
    had_invisible_characters: bool

    @property
    def suspected(self) -> bool:
        return bool(self.signals) or self.had_invisible_characters


def scan(text: str) -> InjectionScan:
    """Detect instruction-shaped content without modifying the text.

    Normalizes to NFKC and strips invisible characters *for the purpose of
    matching only*, so that obfuscation with zero-width joiners or compatibility
    variants does not defeat detection. The original string is never altered.
    """
    if not text:
        return InjectionScan(signals=(), had_invisible_characters=False)

    had_invisible = bool(_INVISIBLE_CHARS.search(text))
    normalized = _INVISIBLE_CHARS.sub("", unicodedata.normalize("NFKC", text))

    signals = tuple(name for name, pattern in _INSTRUCTION_PATTERNS if pattern.search(normalized))
    return InjectionScan(signals=signals, had_invisible_characters=had_invisible)


def isolate(text: str, *, field: str, evidence_id: str | None = None) -> str:
    """Wrap untrusted text in a delimited block declared as data.

    The text itself is passed through unchanged. What changes is its framing: it
    is fenced, labelled with its provenance, and any occurrence of the fence
    inside the payload is neutralized so the block cannot be closed early and
    escaped.
    """
    fence_open = f"<untrusted-data field={field!r}"
    if evidence_id is not None:
        fence_open += f" evidence_id={evidence_id!r}"
    fence_open += ">"
    fence_close = "</untrusted-data>"

    # Neutralize any attempt to close the fence from inside. Replacing "<" with
    # its fullwidth form keeps the text human-readable and keeps the character
    # count stable, while making the tag inert.
    # The fullwidth less-than is the neutralization itself, not a typo: it is
    # visually faithful for a human reading the audit log while being inert as
    # markup. noqa RUF001 is therefore deliberate.
    safe = text.replace("</untrusted-data", "＜/untrusted-data").replace(  # noqa: RUF001
        "<untrusted-data",
        "＜untrusted-data",  # noqa: RUF001
    )
    return f"{fence_open}\n{safe}\n{fence_close}"


def scan_and_isolate(
    text: str, *, field: str, evidence_id: str | None = None
) -> tuple[str, InjectionScan]:
    """Isolate ``text`` and report what was detected in it.

    Isolation happens regardless of the scan result. Detection informs the audit
    log and the ``injection_suspected`` flag; it is not what provides the
    protection.
    """
    return isolate(text, field=field, evidence_id=evidence_id), scan(text)
