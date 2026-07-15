"""Build deterministic, non-executing snapshots of Jupyter notebooks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import nbformat

from notebook_coach import SCHEMA_VERSION
from notebook_coach.sanitize import redact_text, summarize_text


MAX_SELECTED_CELLS = 200
MAX_SOURCE_CHARS = 150_000
MAX_OUTPUT_CHARS = 4_000
MAX_DESCRIPTOR_CHARS = 200

_CELL_SELECTION_HELP = "Use --cells 1-20 with 1-based cell numbers."
_TEXT_MIME_TYPES = {
    "application/javascript",
    "application/json",
    "application/xml",
}


class NotebookInputError(ValueError):
    """Raised when a notebook path, file, structure, or selection is invalid."""


class SnapshotLimitError(NotebookInputError):
    """Raised when the selected notebook content exceeds snapshot limits."""


class _SanitizationState:
    def __init__(self) -> None:
        self.labels: set[str] = set()
        self.redacted_fields = 0
        self.truncated_fields = 0
        self.omitted_binary_fields = 0

    def summarize(self, text: str, *, max_chars: int) -> dict[str, Any]:
        _, labels = redact_text(text)
        summary = summarize_text(text, max_chars=max_chars)
        if labels:
            self.labels.update(labels)
            self.redacted_fields += 1
        if summary["truncated"]:
            self.truncated_fields += 1
        return summary

    def descriptor(self, text: str) -> str:
        cleaned, labels = redact_text(text)
        if labels:
            self.labels.update(labels)
            self.redacted_fields += 1
        if len(cleaned) > MAX_DESCRIPTOR_CHARS:
            self.truncated_fields += 1
        return cleaned[:MAX_DESCRIPTOR_CHARS]

    def omitted_binary(self, mime_type: str, body: Any) -> dict[str, Any]:
        text = _stable_body_text(body)
        self.omitted_binary_fields += 1
        return {
            "mime_type": self.descriptor(mime_type),
            "original_chars": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "omitted": True,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "labels": sorted(self.labels),
            "redacted_fields": self.redacted_fields,
            "truncated_fields": self.truncated_fields,
            "omitted_binary_fields": self.omitted_binary_fields,
        }


def _selection_error(message: str) -> NotebookInputError:
    return NotebookInputError(f"{message} {_CELL_SELECTION_HELP}")


def parse_cell_selection(selection: str, total_cells: int) -> list[int]:
    """Parse user-facing 1-based cell numbers into ordered zero-based indexes."""

    if (
        isinstance(total_cells, bool)
        or not isinstance(total_cells, int)
        or total_cells < 0
    ):
        raise _selection_error("The notebook cell count is invalid.")
    if not isinstance(selection, str) or not selection.strip():
        raise _selection_error("Cell selection cannot be empty.")

    selected: set[int] = set()
    for raw_token in selection.split(","):
        token = raw_token.strip()
        if not token:
            raise _selection_error("Cell selection contains an empty item.")

        single = re.fullmatch(r"[0-9]+", token)
        interval = re.fullmatch(r"([0-9]+)-([0-9]+)", token)
        if single:
            start = end = int(token)
        elif interval:
            start, end = (int(value) for value in interval.groups())
        else:
            raise _selection_error(f"Invalid cell selection item {token!r}.")

        if start < 1 or end < 1:
            raise _selection_error("Cell numbers start at 1, not 0.")
        if start > end:
            raise _selection_error(f"Cell range {token!r} is reversed.")
        if end > total_cells:
            raise _selection_error(
                f"Cell {end} is outside this notebook's {total_cells} cells."
            )

        selected.update(range(start - 1, end))

    return sorted(selected)


def _read_notebook(path: str | Path) -> tuple[Path, str, Any]:
    try:
        notebook_path = Path(path)
    except (TypeError, ValueError, OSError):
        raise NotebookInputError("Notebook path is invalid.") from None

    if notebook_path.suffix.casefold() != ".ipynb":
        raise NotebookInputError("Notebook input must use the .ipynb extension.")

    try:
        raw = notebook_path.read_bytes()
    except OSError:
        raise NotebookInputError("Notebook file does not exist or is not readable.") from None

    raw_sha256 = hashlib.sha256(raw).hexdigest()

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise NotebookInputError("Notebook file must be valid UTF-8.") from None

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        raise NotebookInputError("The notebook file does not contain valid JSON.") from None

    if not isinstance(payload, dict):
        raise NotebookInputError("The notebook JSON must be a top-level object.")
    if payload.get("nbformat") != 4:
        raise NotebookInputError("Notebook uses an unsupported notebook format.")
    if not isinstance(payload.get("nbformat_minor"), int) or isinstance(
        payload.get("nbformat_minor"), bool
    ):
        raise NotebookInputError("Notebook has an invalid notebook format version.")
    if not isinstance(payload.get("metadata"), dict) or not isinstance(
        payload.get("cells"), list
    ):
        raise NotebookInputError("The notebook has an invalid top-level structure.")

    try:
        notebook = nbformat.reads(text, as_version=nbformat.NO_CONVERT)
        nbformat.validate(notebook)
    except Exception:
        raise NotebookInputError("Notebook structure is invalid or unsupported.") from None

    return notebook_path, raw_sha256, notebook


def _normalize_selection(
    selected_cells: Iterable[int] | None, total_cells: int
) -> list[int]:
    if selected_cells is None:
        return list(range(total_cells))
    if isinstance(selected_cells, (str, bytes)):
        raise NotebookInputError(
            "selected_cells must contain zero-based integer cell indexes."
        )

    try:
        indexes = list(selected_cells)
    except TypeError:
        raise NotebookInputError(
            "selected_cells must contain zero-based integer cell indexes."
        ) from None

    normalized: set[int] = set()
    for index in indexes:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= total_cells
        ):
            raise NotebookInputError(
                "selected_cells contains an invalid zero-based cell index."
            )
        normalized.add(index)
    return sorted(normalized)


def _notebook_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return "".join(value)
    raise NotebookInputError("Notebook contains a text field with an invalid structure.")


def _stable_body_text(body: Any) -> str:
    if isinstance(body, str):
        return body
    if isinstance(body, list) and all(isinstance(part, str) for part in body):
        return "".join(body)
    try:
        return json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, RecursionError):
        raise NotebookInputError("Notebook output contains unsupported data.") from None


def _is_text_mime(mime_type: str) -> bool:
    normalized = mime_type.casefold()
    if normalized.startswith("image/"):
        return False
    return (
        normalized.startswith("text/")
        or normalized in _TEXT_MIME_TYPES
        or normalized.endswith("+json")
        or normalized.endswith("+xml")
    )


def _safe_mime_bundle(
    data: Mapping[str, Any], state: _SanitizationState
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for mime_type in sorted(data):
        body = data[mime_type]
        if _is_text_mime(mime_type):
            items.append(
                {
                    "mime_type": state.descriptor(mime_type),
                    "text": state.summarize(
                        _stable_body_text(body), max_chars=MAX_OUTPUT_CHARS
                    ),
                }
            )
        else:
            items.append(state.omitted_binary(mime_type, body))
    return items


def _safe_outputs(outputs: Iterable[Any], state: _SanitizationState) -> list[dict]:
    safe_outputs: list[dict] = []
    for output in outputs:
        output_type = output["output_type"]
        if output_type == "stream":
            safe_outputs.append(
                {
                    "output_type": "stream",
                    "name": state.descriptor(output["name"]),
                    "text": state.summarize(
                        _notebook_text(output["text"]), max_chars=MAX_OUTPUT_CHARS
                    ),
                }
            )
        elif output_type == "error":
            safe_outputs.append(
                {
                    "output_type": "error",
                    "ename": state.descriptor(output["ename"]),
                    "evalue": state.summarize(
                        _notebook_text(output["evalue"]), max_chars=MAX_OUTPUT_CHARS
                    ),
                    "traceback": [
                        state.summarize(
                            _notebook_text(line), max_chars=MAX_OUTPUT_CHARS
                        )
                        for line in output["traceback"]
                    ],
                }
            )
        elif output_type in {"display_data", "execute_result"}:
            safe_output = {
                "output_type": output_type,
                "data": _safe_mime_bundle(output["data"], state),
            }
            if output_type == "execute_result":
                safe_output["execution_count"] = output["execution_count"]
            safe_outputs.append(safe_output)
        else:
            raise NotebookInputError("Notebook contains an unsupported saved output.")
    return safe_outputs


def _safe_attachments(
    attachments: Mapping[str, Mapping[str, Any]], state: _SanitizationState
) -> list[dict[str, Any]]:
    safe_attachments: list[dict[str, Any]] = []
    for attachment_name in sorted(attachments):
        bundle = attachments[attachment_name]
        for mime_type in sorted(bundle):
            safe_attachments.append(state.omitted_binary(mime_type, bundle[mime_type]))
    return safe_attachments


def _safe_optional_descriptor(value: Any, state: _SanitizationState) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise NotebookInputError("Notebook metadata has an invalid structure.")
    return state.descriptor(value)


def build_snapshot(
    path: str | Path, selected_cells: Iterable[int] | None = None
) -> dict[str, Any]:
    """Read and validate a notebook, then return a safe static snapshot."""

    _, raw_sha256, notebook = _read_notebook(path)

    indexes = _normalize_selection(selected_cells, len(notebook.cells))
    if len(indexes) > MAX_SELECTED_CELLS:
        raise SnapshotLimitError(
            f"Snapshot supports at most {MAX_SELECTED_CELLS} selected cells. "
            f"{_CELL_SELECTION_HELP}"
        )

    prepared_sources: dict[int, tuple[str, str]] = {}
    total_source_chars = 0
    for index in indexes:
        source = _notebook_text(notebook.cells[index]["source"])
        cleaned, _ = redact_text(source)
        prepared_sources[index] = (source, cleaned)
        total_source_chars += len(cleaned)

    if total_source_chars > MAX_SOURCE_CHARS:
        raise SnapshotLimitError(
            f"Selected cell source exceeds {MAX_SOURCE_CHARS:,} characters after "
            f"redaction. {_CELL_SELECTION_HELP}"
        )

    state = _SanitizationState()
    metadata = notebook.metadata
    kernelspec = metadata.get("kernelspec", {})
    language_info = metadata.get("language_info", {})
    kernel_name = kernelspec.get("name") if isinstance(kernelspec, Mapping) else None
    language = (
        language_info.get("name") if isinstance(language_info, Mapping) else None
    )

    cells: list[dict[str, Any]] = []
    for index in indexes:
        cell = notebook.cells[index]
        source, cleaned_source = prepared_sources[index]
        safe_source = state.summarize(source, max_chars=MAX_SOURCE_CHARS)
        if safe_source["text"] != cleaned_source:
            raise NotebookInputError("Notebook source could not be safely summarized.")

        safe_cell: dict[str, Any] = {
            "index": index,
            "cell_type": cell["cell_type"],
            "source": safe_source,
        }
        if cell["cell_type"] == "code":
            safe_cell["execution_count"] = cell.get("execution_count")
            safe_cell["outputs"] = _safe_outputs(cell.get("outputs", ()), state)
        elif cell["cell_type"] == "markdown":
            safe_cell["attachments"] = _safe_attachments(
                cell.get("attachments", {}), state
            )
        elif cell["cell_type"] != "raw":
            raise NotebookInputError("Notebook contains an unsupported cell type.")
        cells.append(safe_cell)

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "sha256": raw_sha256,
            "cell_count": len(notebook.cells),
            "kernel_name": _safe_optional_descriptor(kernel_name, state),
            "language": _safe_optional_descriptor(language, state),
            "nbformat": notebook.nbformat,
            "nbformat_minor": notebook.nbformat_minor,
        },
        "cells": cells,
        "sanitization": state.as_dict(),
    }
