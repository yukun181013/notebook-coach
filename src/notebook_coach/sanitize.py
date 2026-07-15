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

_SENSITIVE_NAME_PATTERN = (
    r"[A-Za-z0-9_.-]*"
    r"(?:api[_-]?key|token|password|secret)"
    r"[A-Za-z0-9_.-]*"
)

_SENSITIVE_ASSIGNMENT = re.compile(
    rf"""
    (?P<target>
        (?:
            (?P<key_quote>["']?)
            (?P<simple_name>{_SENSITIVE_NAME_PATTERN})
            (?P=key_quote)
        )
        |
        (?:
            [A-Za-z_][A-Za-z0-9_.]*[ \t]*\[[ \t]*
            (?P<subscript_quote>["'])
            (?P<subscript_name>{_SENSITIVE_NAME_PATTERN})
            (?P=subscript_quote)
            [ \t]*\]
        )
    )
    (?:
        (?P<typed_separator>
            [ \t]*:[ \t]*
            [A-Za-z_][A-Za-z0-9_.]*(?:\[[^=\r\n]*\])?
            (?:[ \t]*\|[ \t]*[A-Za-z_][A-Za-z0-9_.]*)*
            [ \t]+=[ \t]*
        )
        (?:
            "(?P<typed_double>(?:\\.|[^"\\\r\n])*)"
            |
            '(?P<typed_single>(?:\\.|[^'\\\r\n])*)'
            |
            (?P<typed_bare>\[REDACTED\]|[^\s,}}\]\[]+)
        )
        |
        (?P<equals_separator>[ \t]*=[ \t]*)
        (?:
            "(?P<equals_double>(?:\\.|[^"\\\r\n])*)"
            |
            '(?P<equals_single>(?:\\.|[^'\\\r\n])*)'
            |
            (?P<equals_bare>\[REDACTED\]|[^\s,}}\]\[]+)
        )
        |
        (?P<colon_separator>[ \t]*:[ \t]*)
        (?:
            "(?P<colon_double>(?:\\.|[^"\\\r\n])*)"
            |
            '(?P<colon_single>(?:\\.|[^'\\\r\n])*)'
            |
            (?P<colon_bare>[^\r\n,]*?[^\s,\r\n])
        )
        (?P<colon_suffix>[ \t]*)(?=,|[\x7d\]]|\r?$)
    )
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
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

        name = match.group("simple_name") or match.group("subscript_name")
        for style in ("typed", "equals", "colon"):
            separator = match.group(f"{style}_separator")
            if separator is None:
                continue

            double_value = match.group(f"{style}_double")
            single_value = match.group(f"{style}_single")
            bare_value = match.group(f"{style}_bare")
            value = next(
                candidate
                for candidate in (double_value, single_value, bare_value)
                if candidate is not None
            )
            if not _is_sensitive_assignment(name, value):
                return match.group(0)

            replaced = True
            quote = '"' if double_value is not None else "'" if single_value is not None else ""
            suffix = match.group("colon_suffix") or ""
            return f"{match.group('target')}{separator}{quote}{_REDACTED}{quote}{suffix}"

        return match.group(0)

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
