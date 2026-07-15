"""Redact credentials and bound notebook text before model use."""

from __future__ import annotations

import hashlib
import re


_REDACTED = "[REDACTED]"

_NAMED_PATTERNS = (
    (
        "pem_private_key",
        re.compile(
            r"-----BEGIN (?P<key_type>(?:[A-Z0-9]+ )*PRIVATE KEY)-----"
            r".*?"
            r"-----END (?P=key_type)-----",
            re.DOTALL,
        ),
    ),
    (
        "openai_api_key",
        re.compile(
            r"(?<![A-Za-z0-9_-])"
            r"sk-(?:(?:proj|svcacct)-)?[A-Za-z0-9_-]{20,}"
            r"(?![A-Za-z0-9_-])"
        ),
    ),
    (
        "github_token",
        re.compile(
            r"(?<![A-Za-z0-9_])"
            r"(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{22,255})"
            r"(?![A-Za-z0-9_])"
        ),
    ),
)

_SENSITIVE_ASSIGNMENT = re.compile(
    r"""
    (?P<prefix>
        (?P<key_quote>["']?)
        (?P<name>
            [A-Za-z0-9_.-]*
            (?:api[_-]?key|token|password|secret)
            [A-Za-z0-9_.-]*
        )
        (?P=key_quote)
        \s*(?:=|:)\s*
    )
    (?:
        "(?P<double_value>(?:\\.|[^"\\\r\n])*)"
        |
        '(?P<single_value>(?:\\.|[^'\\\r\n])*)'
        |
        (?P<bare_value>\[REDACTED\]|[^\s,}\]\[]+)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _replace_pattern(text: str, pattern: re.Pattern[str]) -> tuple[str, bool]:
    cleaned, count = pattern.subn(_REDACTED, text)
    return cleaned, count > 0


def _is_sensitive_assignment(name: str, value: str) -> bool:
    stripped_value = value.strip()
    if not stripped_value or stripped_value.casefold() in {
        _REDACTED.casefold(),
        "none",
        "null",
        "true",
        "false",
    }:
        return False

    normalized_name = re.sub(r"[.-]", "_", name.casefold())
    if normalized_name.endswith("_count") and re.fullmatch(
        r"[+-]?\d+(?:\.\d+)?", stripped_value
    ):
        return False

    return True


def _redact_sensitive_assignments(text: str) -> tuple[str, bool]:
    replaced = False

    def replacement(match: re.Match[str]) -> str:
        nonlocal replaced

        value = next(
            candidate
            for candidate in (
                match.group("double_value"),
                match.group("single_value"),
                match.group("bare_value"),
            )
            if candidate is not None
        )
        if not _is_sensitive_assignment(match.group("name"), value):
            return match.group(0)

        replaced = True
        if match.group("double_value") is not None:
            return f'{match.group("prefix")}"{_REDACTED}"'
        if match.group("single_value") is not None:
            return f"{match.group('prefix')}'{_REDACTED}'"
        return f"{match.group('prefix')}{_REDACTED}"

    return _SENSITIVE_ASSIGNMENT.sub(replacement, text), replaced


def redact_text(text: str) -> tuple[str, list[str]]:
    """Replace recognized secrets with a marker and return safe pattern labels."""

    cleaned = text
    labels: list[str] = []

    for label, pattern in _NAMED_PATTERNS:
        cleaned, matched = _replace_pattern(cleaned, pattern)
        if matched:
            labels.append(label)

    cleaned, matched_assignment = _redact_sensitive_assignments(cleaned)
    if matched_assignment:
        labels.append("sensitive_assignment")

    return cleaned, labels


def summarize_text(text: str, max_chars: int = 4000) -> dict:
    """Return a redacted, character-bounded summary with stable metadata."""

    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError("max_chars must be an integer")
    if max_chars < 0:
        raise ValueError("max_chars must be non-negative")

    cleaned, _ = redact_text(text)
    return {
        "text": cleaned[:max_chars],
        "original_chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "truncated": len(cleaned) > max_chars,
    }
