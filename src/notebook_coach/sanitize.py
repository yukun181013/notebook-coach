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

_ASSIGNMENT_NAME_PATTERN = r"[A-Za-z_][A-Za-z0-9_.-]*"

_ASSIGNMENT_TARGET = re.compile(
    rf"""
    (?:
        (?P<subscript>
            [A-Za-z_][A-Za-z0-9_.]*[ \t]*\[[ \t]*
            (?P<subscript_quote>["'])
            (?P<subscript_name>{_ASSIGNMENT_NAME_PATTERN})
            (?P=subscript_quote)
            [ \t]*\]
        )
        |
        (?P<quoted>
            (?P<key_quote>["'])
            (?P<quoted_name>{_ASSIGNMENT_NAME_PATTERN})
            (?P=key_quote)
        )
        |
        (?P<simple>
            (?<![A-Za-z0-9_.-])
            (?P<simple_name>{_ASSIGNMENT_NAME_PATTERN})
        )
    )
    """,
    re.VERBOSE,
)

_PYTHON_ANNOTATION_ATOM_PATTERN = (
    r"(?:"
    r'"(?:\\.|[^"\\\r\n])*"'
    r"|'(?:\\.|[^'\\\r\n])*'"
    r"|[A-Za-z_][A-Za-z0-9_.]*(?:[ \t]*\[[^=\r\n]*\])?"
    r")"
)

_PYTHON_ANNOTATION = re.compile(
    rf"""
    [ \t]*
    (?P<annotation>
        {_PYTHON_ANNOTATION_ATOM_PATTERN}
        (?:[ \t]*\|[ \t]*{_PYTHON_ANNOTATION_ATOM_PATTERN})*
    )
    [ \t]*
    (?P<terminator>=|[,):#]|\r?\n|$)
    """,
    re.VERBOSE,
)

_TOKEN_METADATA_COMPONENTS = {
    "budget",
    "budgets",
    "count",
    "counts",
    "id",
    "ids",
    "length",
    "limit",
    "limits",
    "type",
    "usage",
    "used",
}
_TOKEN_METADATA_PREFIXES = {"max", "min", "num", "total"}
_BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}


def _replace_pattern(text: str, pattern: re.Pattern[str]) -> tuple[str, bool]:
    cleaned, count = pattern.subn(_REDACTED, text)
    return cleaned, count > 0


def _target_name(target: re.Match[str]) -> str:
    return next(
        name
        for name in (
            target.group("subscript_name"),
            target.group("quoted_name"),
            target.group("simple_name"),
        )
        if name is not None
    )


def _is_sensitive_name(name: str) -> bool:
    components = [
        component
        for component in re.split(r"[_.-]+", name.casefold())
        if component
    ]
    if any(component in {"apikey", "password", "secret"} for component in components):
        return True
    if any(
        components[index : index + 2] == ["api", "key"]
        for index in range(len(components) - 1)
    ):
        return True

    for index, component in enumerate(components):
        if component != "token":
            continue
        previous = components[index - 1] if index else None
        following = components[index + 1] if index + 1 < len(components) else None
        if previous in _TOKEN_METADATA_PREFIXES:
            continue
        if following in _TOKEN_METADATA_COMPONENTS:
            continue
        return True
    return False


def _is_sensitive_assignment(name: str, value: str) -> bool:
    if not _is_sensitive_name(name):
        return False
    stripped_value = value.strip()
    return bool(stripped_value) and stripped_value.casefold() not in {
        _REDACTED.casefold(),
        "none",
        "null",
        "true",
        "false",
    }


def _skip_horizontal_space(text: str, start: int) -> int:
    while start < len(text) and text[start] in " \t":
        start += 1
    return start


def _looks_like_python_annotation(annotation: str) -> bool:
    annotation = annotation.strip()
    simple_annotations = {
        "Any",
        "None",
        "bool",
        "bytes",
        "complex",
        "float",
        "int",
        "object",
        "str",
    }

    if annotation[:1] in {"\"", "'"} and annotation[-1:] == annotation[:1]:
        forwarded = annotation[1:-1]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*(?:\[[^\r\n]*\])?", forwarded):
            return False
        final_name = forwarded.rsplit(".", 1)[-1].split("[", 1)[0]
        return (
            "[" in forwarded
            or final_name in simple_annotations
            or final_name[:1].isupper()
        )

    first_name = annotation.split("[", 1)[0].split("|", 1)[0].strip()
    return (
        "." in annotation
        or "[" in annotation
        or "|" in annotation
        or first_name in simple_annotations
        or first_name[:1].isupper()
    )


def _assignment_value_start(
    text: str, target: re.Match[str]
) -> tuple[int, str] | None:
    separator = _skip_horizontal_space(text, target.end())
    if separator >= len(text):
        return None

    if text[separator] == "=":
        if text.startswith("==", separator):
            return None
        return _skip_horizontal_space(text, separator + 1), "equals"
    if text.startswith(":=", separator):
        return _skip_horizontal_space(text, separator + 2), "equals"
    if text[separator] != ":":
        return None

    if target.group("simple_name") is not None:
        annotation = _PYTHON_ANNOTATION.match(text, separator + 1)
        if annotation is not None and _looks_like_python_annotation(
            annotation.group("annotation")
        ):
            if annotation.group("terminator") != "=":
                return None
            return _skip_horizontal_space(text, annotation.end()), "equals"

    return _skip_horizontal_space(text, separator + 1), "colon"


def _line_start(text: str, position: int) -> int:
    return max(text.rfind("\n", 0, position), text.rfind("\r", 0, position)) + 1


def _line_end(text: str, position: int) -> int:
    newline = text.find("\n", position)
    carriage_return = text.find("\r", position)
    candidates = [index for index in (newline, carriage_return) if index != -1]
    return min(candidates) if candidates else len(text)


def _after_line_break(text: str, position: int) -> int:
    if position >= len(text):
        return position
    if text.startswith("\r\n", position):
        return position + 2
    return position + 1


def _indent_width(text: str, line_start: int) -> int:
    width = 0
    while line_start < len(text) and text[line_start] in " \t":
        width += 4 if text[line_start] == "\t" else 1
        line_start += 1
    return width


def _semicolon_starts_statement(text: str, semicolon: int) -> bool:
    cursor = _skip_horizontal_space(text, semicolon + 1)
    if cursor >= len(text) or text[cursor] in "\r\n#":
        return True

    statement = re.match(r"[A-Za-z_][A-Za-z0-9_.]*", text[cursor:])
    if statement is None:
        return False
    word = statement.group(0)
    if word in {
        "assert",
        "break",
        "continue",
        "del",
        "import",
        "pass",
        "raise",
        "return",
    }:
        return True
    after_name = _skip_horizontal_space(text, cursor + statement.end())
    return after_name < len(text) and text[after_name] in "(=:"


def _scan_quoted_token(text: str, start: int, delimiter: str) -> int | None:
    cursor = start + len(delimiter)
    while cursor < len(text):
        if text[cursor] == "\\" and cursor + 1 < len(text):
            cursor += 2
            continue
        if text.startswith(delimiter, cursor):
            return cursor + len(delimiter)
        if len(delimiter) == 1 and text[cursor] in "\r\n":
            return None
        cursor += 1
    return None


def _is_rhs_boundary(text: str, position: int) -> bool:
    position = _skip_horizontal_space(text, position)
    if position >= len(text) or text[position] in "\r\n,)}]#":
        return True
    return text[position] == ";" and _semicolon_starts_statement(text, position)


def _is_yaml_block_header(text: str, start: int) -> bool:
    header = text[start : _line_end(text, start)]
    return bool(
        re.fullmatch(r"[|>](?:[1-9][+-]?|[+-][1-9]?)?[ \t]*(?:#.*)?", header)
    )


def _scan_yaml_block(
    text: str, start: int, target_start: int
) -> tuple[int, int, str, int]:
    header_end = _line_end(text, start)
    block_end = header_end
    base_indent = _indent_width(text, _line_start(text, target_start))
    line_start = _after_line_break(text, header_end)

    while line_start < len(text):
        line_end = _line_end(text, line_start)
        line = text[line_start:line_end]
        if line.strip() and _indent_width(text, line_start) <= base_indent:
            break
        block_end = line_end
        if line_end >= len(text):
            break
        line_start = _after_line_break(text, line_end)

    return start, block_end, text[start:block_end], block_end


def _scan_expression_value(
    text: str, start: int
) -> tuple[int, int, str, int] | None:
    stack: list[str] = []
    quote: str | None = None
    in_comment = False
    cursor = start

    while cursor < len(text):
        character = text[cursor]

        if in_comment:
            if character in "\r\n":
                if not stack:
                    break
                in_comment = False
                cursor = _after_line_break(text, cursor)
                continue
            cursor += 1
            continue

        if quote is not None:
            if character == "\\" and cursor + 1 < len(text):
                cursor += 2
                continue
            if text.startswith(quote, cursor):
                cursor += len(quote)
                quote = None
                continue
            if len(quote) == 1 and character in "\r\n":
                break
            cursor += 1
            continue

        if text.startswith(('"""', "'''"), cursor):
            quote = text[cursor : cursor + 3]
            cursor += 3
            continue
        if character in "\"'":
            quote = character
            cursor += 1
            continue
        if character == "#":
            in_comment = True
            cursor += 1
            continue
        if character in _BRACKET_PAIRS:
            stack.append(_BRACKET_PAIRS[character])
            cursor += 1
            continue
        if character in ")]}":
            if not stack:
                break
            if character == stack[-1]:
                stack.pop()
            cursor += 1
            continue
        if not stack and character == ",":
            break
        if (
            not stack
            and character == ";"
            and _semicolon_starts_statement(text, cursor)
        ):
            break
        if character in "\r\n" and not stack:
            break
        if character in "\r\n":
            cursor = _after_line_break(text, cursor)
            continue
        cursor += 1

    end = cursor
    while end > start and text[end - 1] in " \t":
        end -= 1
    if end == start:
        return None
    return start, end, text[start:end], cursor


def _scan_assignment_value(
    text: str, start: int, style: str, target_start: int
) -> tuple[int, int, str, int] | None:
    if start >= len(text):
        return None

    if style == "colon" and text[start] in "|>" and _is_yaml_block_header(text, start):
        return _scan_yaml_block(text, start, target_start)

    delimiter = None
    if text.startswith(('"""', "'''"), start):
        delimiter = text[start : start + 3]
    elif text[start] in "\"'":
        delimiter = text[start]

    if delimiter is not None:
        quoted_end = _scan_quoted_token(text, start, delimiter)
        if quoted_end is not None and _is_rhs_boundary(text, quoted_end):
            content_start = start + len(delimiter)
            content_end = quoted_end - len(delimiter)
            return (
                content_start,
                content_end,
                text[content_start:content_end],
                quoted_end,
            )

    return _scan_expression_value(text, start)


def _redact_sensitive_assignments(text: str) -> tuple[str, bool]:
    output: list[str] = []
    copied_until = 0
    search_from = 0
    replaced = False

    while target := _ASSIGNMENT_TARGET.search(text, search_from):
        name = _target_name(target)
        if not _is_sensitive_name(name):
            search_from = target.end()
            continue

        assignment = _assignment_value_start(text, target)
        if assignment is None:
            search_from = target.end()
            continue
        value_start, style = assignment

        value = _scan_assignment_value(text, value_start, style, target.start())
        if value is None:
            search_from = max(target.end(), value_start + 1)
            continue
        replace_start, replace_end, original_value, resume_at = value

        if not _is_sensitive_assignment(name, original_value):
            search_from = max(target.end(), resume_at)
            continue

        output.append(text[copied_until:replace_start])
        output.append(_REDACTED)
        copied_until = replace_end
        search_from = max(replace_end, resume_at)
        replaced = True

    output.append(text[copied_until:])
    return "".join(output), replaced


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
