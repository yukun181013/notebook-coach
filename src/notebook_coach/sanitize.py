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

_SENSITIVE_TARGET = re.compile(
    rf"""
    (?:
        (?P<subscript>
            [A-Za-z_][A-Za-z0-9_.]*[ \t]*\[[ \t]*
            (?P<subscript_quote>["'])
            (?P<subscript_name>{_SENSITIVE_NAME_PATTERN})
            (?P=subscript_quote)
            [ \t]*\]
        )
        |
        (?P<simple>
            (?P<key_quote>["']?)
            (?P<simple_name>{_SENSITIVE_NAME_PATTERN})
            (?P=key_quote)
        )
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PYTHON_TYPE_ATOM_PATTERN = (
    r"(?:"
    r"str|bytes|int|float|bool|complex|object|None|Any"
    r"|list|dict|tuple|set|frozenset"
    r"|(?:typing\.)?[A-Z][A-Za-z0-9_.]*"
    r")"
    r"(?:[ \t]*\[[^=\r\n]*\])?"
)

_PYTHON_ANNOTATION = re.compile(
    rf"""
    [ \t]*
    {_PYTHON_TYPE_ATOM_PATTERN}
    (?:[ \t]*\|[ \t]*{_PYTHON_TYPE_ATOM_PATTERN})*
    [ \t]*
    (?P<terminator>=|[,):#]|$)
    """,
    re.VERBOSE,
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


def _skip_horizontal_space(line: str, start: int) -> int:
    while start < len(line) and line[start] in " \t":
        start += 1
    return start


def _assignment_value_start(line: str, target_end: int) -> int | None:
    separator = _skip_horizontal_space(line, target_end)
    if separator >= len(line):
        return None

    if line[separator] == "=":
        return _skip_horizontal_space(line, separator + 1)
    if line[separator] != ":":
        return None

    annotation = _PYTHON_ANNOTATION.match(line, separator + 1)
    if annotation is None:
        return _skip_horizontal_space(line, separator + 1)
    if annotation.group("terminator") != "=":
        return None
    return _skip_horizontal_space(line, annotation.end())


def _scan_quoted_value(line: str, start: int) -> tuple[int, int, str]:
    quote = line[start]
    cursor = start + 1
    while cursor < len(line):
        if line[cursor] == "\\" and cursor + 1 < len(line):
            cursor += 2
            continue
        if line[cursor] == quote:
            return start + 1, cursor, line[start + 1 : cursor]
        cursor += 1
    return start + 1, len(line), line[start + 1 :]


def _scan_container_value(line: str, start: int) -> tuple[int, int, str]:
    closing_for = {"[": "]", "{": "}", "(": ")"}
    stack = [closing_for[line[start]]]
    cursor = start + 1
    quote: str | None = None

    while cursor < len(line):
        character = line[cursor]
        if quote is not None:
            if character == "\\" and cursor + 1 < len(line):
                cursor += 2
                continue
            if character == quote:
                quote = None
        elif character in "\"'":
            quote = character
        elif character in closing_for:
            stack.append(closing_for[character])
        elif character == stack[-1]:
            stack.pop()
            if not stack:
                return start, cursor + 1, line[start : cursor + 1]
        cursor += 1

    return start, len(line), line[start:]


def _scan_bare_value(line: str, start: int) -> tuple[int, int, str] | None:
    cursor = start
    while cursor < len(line):
        character = line[cursor]
        if character in ",)}]":
            break
        if character == ";" and (
            cursor + 1 == len(line) or line[cursor + 1].isspace()
        ):
            break
        cursor += 1

    end = cursor
    while end > start and line[end - 1] in " \t":
        end -= 1
    if end == start:
        return None
    return start, end, line[start:end]


def _scan_assignment_value(line: str, start: int) -> tuple[int, int, str] | None:
    if start >= len(line):
        return None
    if line[start] in "\"'":
        return _scan_quoted_value(line, start)
    if line[start] in "[{(":
        return _scan_container_value(line, start)
    return _scan_bare_value(line, start)


def _redact_assignment_line(line: str) -> tuple[str, bool]:
    output: list[str] = []
    copied_until = 0
    search_from = 0
    replaced = False

    while target := _SENSITIVE_TARGET.search(line, search_from):
        value_start = _assignment_value_start(line, target.end())
        if value_start is None:
            search_from = target.end()
            continue

        value = _scan_assignment_value(line, value_start)
        if value is None:
            search_from = max(target.end(), value_start + 1)
            continue

        replace_start, replace_end, original_value = value
        name = target.group("simple_name") or target.group("subscript_name")
        if not _is_sensitive_assignment(name, original_value):
            search_from = max(target.end(), replace_end)
            continue

        output.append(line[copied_until:replace_start])
        output.append(_REDACTED)
        copied_until = replace_end
        search_from = replace_end
        replaced = True

    output.append(line[copied_until:])
    return "".join(output), replaced


def _redact_sensitive_assignments(text: str) -> tuple[str, bool]:
    cleaned_lines: list[str] = []
    replaced = False

    for line in text.splitlines(keepends=True):
        if line.endswith("\r\n"):
            body, ending = line[:-2], "\r\n"
        elif line.endswith(("\r", "\n")):
            body, ending = line[:-1], line[-1]
        else:
            body, ending = line, ""

        cleaned, line_replaced = _redact_assignment_line(body)
        cleaned_lines.append(cleaned + ending)
        replaced = replaced or line_replaced

    return "".join(cleaned_lines), replaced


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
