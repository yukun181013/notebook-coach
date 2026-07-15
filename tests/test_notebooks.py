from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat
import pytest

import notebook_coach.notebooks as notebooks_module
from notebook_coach.notebooks import (
    NotebookInputError,
    SnapshotLimitError,
    build_snapshot,
    parse_cell_selection,
)


SECRET = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"


def _visible_summary_chars(value):
    if isinstance(value, dict):
        if {"text", "original_chars", "sha256", "truncated"} <= set(value):
            return len(value["text"])
        return sum(_visible_summary_chars(item) for item in value.values())
    if isinstance(value, list):
        return sum(_visible_summary_chars(item) for item in value)
    return 0


def test_snapshot_is_redacted_bounded_and_hashes_original(notebook_factory):
    path = notebook_factory(code="OPENAI_API_KEY='sk-proj-abcdefghijklmnopqrstuvwxyz123456'")
    before = path.read_bytes()
    snapshot = build_snapshot(path)
    assert snapshot["source"]["sha256"]
    assert snapshot["source"]["cell_count"] == 1
    assert "sk-proj" not in str(snapshot)
    assert snapshot["cells"][0]["index"] == 0
    assert path.read_bytes() == before


def test_cell_selection_is_one_based_for_users():
    assert parse_cell_selection("1-3,5", total_cells=6) == [0, 1, 2, 4]


def test_snapshot_is_deterministic_and_json_serializable(notebook_factory):
    path = notebook_factory(code="answer = 42")

    first = build_snapshot(path)
    second = build_snapshot(path)

    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True)) == first
    assert set(first) == {"schema_version", "source", "cells", "sanitization"}
    assert first["schema_version"] == "1.0"


def test_snapshot_hashes_raw_bytes_and_does_not_rewrite_notebook(notebook_factory):
    path = notebook_factory(code="value = 1")
    path.write_bytes(path.read_bytes() + b"\n")
    before = path.read_bytes()

    snapshot = build_snapshot(path)

    assert snapshot["source"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before


def test_raw_bytes_are_hashed_before_notebook_parsing(notebook_factory, monkeypatch):
    path = notebook_factory(code="value = 1")
    events = []
    real_sha256 = notebooks_module.hashlib.sha256
    real_reads = notebooks_module.nbformat.reads

    def recording_sha256(value):
        events.append("hash")
        return real_sha256(value)

    def recording_reads(*args, **kwargs):
        events.append("parse")
        return real_reads(*args, **kwargs)

    monkeypatch.setattr(notebooks_module.hashlib, "sha256", recording_sha256)
    monkeypatch.setattr(notebooks_module.nbformat, "reads", recording_reads)

    build_snapshot(path)

    assert events[0:2] == ["hash", "parse"]


def test_selected_cells_keep_original_zero_based_indexes(notebook_factory):
    cells = [
        nbformat.v4.new_code_cell("first"),
        nbformat.v4.new_markdown_cell("second"),
        nbformat.v4.new_code_cell("third"),
    ]
    path = notebook_factory(cells=cells)

    snapshot = build_snapshot(path, selected_cells=[2, 0, 2])

    assert [cell["index"] for cell in snapshot["cells"]] == [0, 2]
    assert [cell["source"]["text"] for cell in snapshot["cells"]] == [
        "first",
        "third",
    ]


def test_build_snapshot_never_executes_notebook_code(notebook_factory, tmp_path):
    marker = tmp_path / "executed.txt"
    path = notebook_factory(code=f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')")

    build_snapshot(path)

    assert not marker.exists()


def test_rejects_non_ipynb_path(notebook_factory, tmp_path):
    valid_path = notebook_factory(code="pass")
    wrong_suffix = tmp_path / "notebook.json"
    wrong_suffix.write_bytes(valid_path.read_bytes())

    with pytest.raises(NotebookInputError, match=r"\.ipynb"):
        build_snapshot(wrong_suffix)


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_wraps_missing_or_unreadable_inputs_as_public_error(tmp_path, kind):
    path = tmp_path / f"{kind}.ipynb"
    if kind == "directory":
        path.mkdir()

    with pytest.raises(NotebookInputError) as error:
        build_snapshot(path)

    assert type(error.value) is NotebookInputError
    assert error.value.__cause__ is None


def test_wraps_non_utf8_input_as_public_error(tmp_path):
    path = tmp_path / "invalid-encoding.ipynb"
    path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(NotebookInputError, match="UTF-8") as error:
        build_snapshot(path)

    assert error.value.__cause__ is None


def test_wraps_invalid_json_as_public_error(tmp_path):
    path = tmp_path / "invalid-json.ipynb"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(NotebookInputError, match="notebook") as error:
        build_snapshot(path)

    assert "JSONDecodeError" not in str(error.value)
    assert error.value.__cause__ is None


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": {}},
        {"nbformat": 999, "nbformat_minor": 0, "metadata": {}, "cells": []},
    ],
)
def test_rejects_invalid_or_unsupported_notebook_structures(tmp_path, payload):
    path = tmp_path / "invalid-structure.ipynb"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(NotebookInputError, match="notebook") as error:
        build_snapshot(path)

    assert error.value.__cause__ is None


@pytest.mark.parametrize(
    "selection",
    ["", "0", "7", "3-2", "one", "1,,2", "1-", "-2"],
)
def test_cell_selection_errors_are_public_and_actionable(selection):
    with pytest.raises(NotebookInputError) as error:
        parse_cell_selection(selection, total_cells=6)

    assert "--cells" in str(error.value)
    assert error.value.__cause__ is None


@pytest.mark.parametrize(
    "selection",
    ["9" * 5_000, "1-" + "9" * 5_000],
    ids=["single", "range"],
)
def test_oversized_numeric_cell_selection_is_a_short_public_error(selection):
    with pytest.raises(NotebookInputError) as error:
        parse_cell_selection(selection, total_cells=6)

    assert "--cells" in str(error.value)
    assert len(str(error.value)) < 300
    assert selection not in str(error.value)
    assert error.value.__cause__ is None


def test_nonnumeric_cell_selection_is_not_reported_as_oversized():
    with pytest.raises(NotebookInputError, match="Invalid cell selection item"):
        parse_cell_selection("one", total_cells=6)


def test_cell_selection_deduplicates_and_preserves_notebook_order():
    assert parse_cell_selection("5,2,5,1-2", total_cells=5) == [0, 1, 4]


def test_build_snapshot_rejects_invalid_zero_based_selection(notebook_factory):
    path = notebook_factory(code="pass")

    with pytest.raises(NotebookInputError, match="selected_cells"):
        build_snapshot(path, selected_cells=[1])


def test_selected_cells_stops_after_the_201st_raw_entry(notebook_factory):
    path = notebook_factory(code="pass")
    requests = []

    def guarded_selection():
        for request in range(1, 203):
            requests.append(request)
            if request == 202:
                raise AssertionError("selected_cells requested a 202nd item")
            yield 0

    with pytest.raises(SnapshotLimitError, match=r"--cells 1-20") as error:
        build_snapshot(path, selected_cells=guarded_selection())

    assert requests == list(range(1, 202))
    assert error.value.__cause__ is None


def test_selected_cells_allows_200_raw_entries(notebook_factory):
    path = notebook_factory(code="pass")

    snapshot = build_snapshot(path, selected_cells=(0 for _ in range(200)))

    assert [cell["index"] for cell in snapshot["cells"]] == [0]


def test_huge_range_selection_is_not_materialized(monkeypatch):
    def forbidden_list(_value):
        raise AssertionError("selected_cells range was materialized")

    monkeypatch.setattr(notebooks_module, "list", forbidden_list, raising=False)

    with pytest.raises(NotebookInputError, match="selected_cells") as error:
        notebooks_module._normalize_selection(range(10**9), total_cells=1)

    assert error.value.__cause__ is None


def test_default_selection_rejects_cell_limit_before_building_range(monkeypatch):
    def forbidden_range(_total):
        raise AssertionError("default cell range was materialized")

    monkeypatch.setattr(notebooks_module, "range", forbidden_range, raising=False)

    with pytest.raises(SnapshotLimitError, match=r"--cells 1-20") as error:
        notebooks_module._normalize_selection(None, total_cells=201)

    assert error.value.__cause__ is None


def test_default_cell_limit_is_checked_after_selection(notebook_factory):
    cells = [nbformat.v4.new_code_cell(f"cell_{index}") for index in range(201)]
    path = notebook_factory(cells=cells)

    with pytest.raises(SnapshotLimitError, match=r"--cells 1-20"):
        build_snapshot(path)

    snapshot = build_snapshot(path, selected_cells=range(200))
    assert snapshot["source"]["cell_count"] == 201
    assert len(snapshot["cells"]) == 200
    assert snapshot["cells"][-1]["index"] == 199


def test_source_limit_checks_complete_redacted_source_before_storage(notebook_factory):
    path = notebook_factory(code="x" * 150_001)

    with pytest.raises(SnapshotLimitError, match=r"--cells 1-20"):
        build_snapshot(path)


def test_source_limit_is_checked_after_selection(notebook_factory):
    cells = [
        nbformat.v4.new_code_cell("x" * 150_001),
        nbformat.v4.new_code_cell("small = True"),
    ]
    path = notebook_factory(cells=cells)

    snapshot = build_snapshot(path, selected_cells=[1])

    assert len(snapshot["cells"]) == 1
    assert snapshot["cells"][0]["index"] == 1
    assert snapshot["cells"][0]["source"]["text"] == "small = True"


def test_source_limit_uses_full_redacted_length(notebook_factory):
    long_secret = "sk-proj-" + "a" * 150_001
    path = notebook_factory(code=f"OPENAI_API_KEY='{long_secret}'")

    snapshot = build_snapshot(path)

    source = snapshot["cells"][0]["source"]
    assert source["text"] == "OPENAI_API_KEY='[REDACTED]'"
    assert source["truncated"] is False
    assert long_secret not in str(snapshot)


def test_source_summary_keeps_all_text_that_passed_the_limit(notebook_factory):
    source_text = "x" * 150_000
    path = notebook_factory(code=source_text)

    source = build_snapshot(path)["cells"][0]["source"]

    assert source["text"] == source_text
    assert source["original_chars"] == 150_000
    assert source["sha256"] == hashlib.sha256(source_text.encode()).hexdigest()
    assert source["truncated"] is False


def test_snapshot_keeps_only_stable_whitelisted_metadata(notebook_factory):
    output = nbformat.v4.new_output(
        "display_data",
        data={"text/plain": "safe result"},
        metadata={"output_secret": SECRET},
    )
    cell = nbformat.v4.new_code_cell(
        "print('safe')",
        execution_count=7,
        outputs=[output],
        metadata={"cell_secret": SECRET, "tags": ["private"]},
    )
    path = notebook_factory(
        cells=[cell],
        metadata={
            "kernelspec": {
                "name": "python3",
                "display_name": SECRET,
                "language": "python",
                "extra": SECRET,
            },
            "language_info": {"name": "python", "version": SECRET},
            "widgets": {"state": SECRET},
            "notebook_secret": SECRET,
        },
    )

    snapshot = build_snapshot(path)

    assert set(snapshot) == {"schema_version", "source", "cells", "sanitization"}
    assert set(snapshot["source"]) == {
        "sha256",
        "cell_count",
        "kernel_name",
        "language",
        "nbformat",
        "nbformat_minor",
    }
    assert snapshot["source"]["kernel_name"] == "python3"
    assert snapshot["source"]["language"] == "python"
    assert set(snapshot["cells"][0]) == {
        "index",
        "cell_type",
        "source",
        "risk",
        "execution_count",
        "outputs",
    }
    assert snapshot["cells"][0]["risk"] == {
        "source_sha256": snapshot["cells"][0]["source"]["sha256"],
        "categories": [],
    }
    assert set(snapshot["cells"][0]["outputs"][0]) == {"output_type", "data"}
    assert SECRET not in str(snapshot)
    assert "widgets" not in str(snapshot)
    assert "metadata" not in str(snapshot)


def test_saved_module_not_found_error_is_static_sanitized_evidence(notebook_factory):
    output = nbformat.v4.new_output(
        "error",
        ename="ModuleNotFoundError",
        evalue=f"No module named 'missing'; api_key={SECRET}",
        traceback=[f"Traceback: api_key={SECRET}", "ModuleNotFoundError: missing"],
    )
    path = notebook_factory(code="import missing", outputs=[output])

    error = build_snapshot(path)["cells"][0]["outputs"][0]

    assert error["output_type"] == "error"
    assert error["ename"] == "ModuleNotFoundError"
    assert isinstance(error["traceback"], dict)
    assert set(error["traceback"]) == {
        "text",
        "original_chars",
        "sha256",
        "truncated",
    }
    assert "Traceback:" in error["traceback"]["text"]
    assert "ModuleNotFoundError: missing" in error["traceback"]["text"]
    assert SECRET not in str(error)
    assert "[REDACTED]" in str(error)


def test_error_traceback_is_one_bounded_summary(notebook_factory):
    traceback = [f"frame-{index:04d} " + "x" * 24 for index in range(1_000)]
    traceback[0] = f"frame-0000 api_key={SECRET}"
    output = nbformat.v4.new_output(
        "error",
        ename="ModuleNotFoundError",
        evalue="No module named 'missing'",
        traceback=traceback,
    )
    path = notebook_factory(code="import missing", outputs=[output])

    snapshot = build_snapshot(path)
    saved_traceback = snapshot["cells"][0]["outputs"][0]["traceback"]

    assert isinstance(saved_traceback, dict)
    assert set(saved_traceback) == {
        "text",
        "original_chars",
        "sha256",
        "truncated",
    }
    assert len(saved_traceback["text"]) <= 4_000
    assert saved_traceback["truncated"] is True
    assert "frame-0000" in saved_traceback["text"]
    assert snapshot["sanitization"]["truncated_fields"] == 1
    assert SECRET not in str(snapshot)
    assert len(json.dumps(snapshot)) < 20_000


def test_text_outputs_are_redacted_and_bounded(notebook_factory):
    long_stream = f"token={SECRET}\n" + "x" * 20_000
    outputs = [
        nbformat.v4.new_output("stream", name="stdout", text=long_stream),
        nbformat.v4.new_output(
            "display_data",
            data={"text/plain": f"result={SECRET}"},
            metadata={"secret": SECRET},
        ),
    ]
    path = notebook_factory(code="pass", outputs=outputs)

    saved_outputs = build_snapshot(path)["cells"][0]["outputs"]

    stream = saved_outputs[0]["text"]
    displayed = saved_outputs[1]["data"][0]["text"]
    assert stream["truncated"] is True
    assert len(stream["text"]) <= 4_000
    assert displayed["truncated"] is False
    assert SECRET not in str(saved_outputs)
    assert "[REDACTED]" in str(saved_outputs)


def test_saved_output_text_shares_one_snapshot_budget(notebook_factory):
    outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="s" * 4_000)
        for _ in range(10)
    ]
    outputs.extend(
        [
            nbformat.v4.new_output(
                "display_data",
                data={"text/plain": "d" * 4_000},
            ),
            nbformat.v4.new_output(
                "error",
                ename="RuntimeError",
                evalue="e" * 4_000,
                traceback=["t" * 4_000],
            ),
            nbformat.v4.new_output(
                "stream",
                name="stderr",
                text=f"last={SECRET}",
            ),
        ]
    )
    path = notebook_factory(code="q" * 60_000, outputs=outputs)

    snapshot = build_snapshot(path)
    saved_outputs = snapshot["cells"][0]["outputs"]

    assert len(snapshot["cells"][0]["source"]["text"]) == 60_000
    assert _visible_summary_chars(saved_outputs) == 50_000
    assert len(saved_outputs[-2]["traceback"]["text"]) == 2_000
    assert saved_outputs[-2]["traceback"]["truncated"] is True
    assert saved_outputs[-1]["text"]["text"] == ""
    assert saved_outputs[-1]["text"]["truncated"] is True
    assert snapshot["sanitization"]["truncated_fields"] == 2
    assert SECRET not in str(snapshot)


def test_saved_evidence_item_limit_counts_output_entries(notebook_factory):
    outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="x")
        for _ in range(501)
    ]
    path = notebook_factory(code="pass", outputs=outputs)

    with pytest.raises(SnapshotLimitError, match=r"--cells 1-20"):
        build_snapshot(path)


def test_saved_evidence_item_limit_counts_mime_items(notebook_factory):
    data = {f"application/x-item-{index}": "x" for index in range(500)}
    output = nbformat.v4.new_output("display_data", data=data)
    path = notebook_factory(code="display(result)", outputs=[output])

    with pytest.raises(SnapshotLimitError, match=r"--cells 1-20"):
        build_snapshot(path)


def test_saved_evidence_item_limit_counts_attachment_mime_items(notebook_factory):
    attachments = {
        f"file-{index}.bin": {"application/octet-stream": "x"}
        for index in range(501)
    }
    path = notebook_factory(markdown="attachments", attachments=attachments)

    with pytest.raises(SnapshotLimitError, match=r"--cells 1-20"):
        build_snapshot(path)


def test_saved_evidence_limit_ignores_unselected_cells(notebook_factory):
    oversized = nbformat.v4.new_code_cell(
        "unselected",
        outputs=[
            nbformat.v4.new_output("stream", name="stdout", text="x")
            for _ in range(501)
        ],
    )
    selected = nbformat.v4.new_code_cell("selected")
    path = notebook_factory(cells=[oversized, selected])

    snapshot = build_snapshot(path, selected_cells=[1])

    assert [cell["index"] for cell in snapshot["cells"]] == [1]


def test_binary_output_body_is_omitted_with_stable_metadata(notebook_factory):
    body = "iVBORw0KGgo" + SECRET + "A" * 100
    output = nbformat.v4.new_output(
        "display_data",
        data={"image/png": body, "text/plain": "<image>"},
    )
    path = notebook_factory(code="display(image)", outputs=[output])

    data = build_snapshot(path)["cells"][0]["outputs"][0]["data"]
    binary = next(item for item in data if item["mime_type"] == "image/png")

    assert binary == {
        "mime_type": "image/png",
        "original_chars": len(body),
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "omitted": True,
    }
    assert body not in str(data)
    assert SECRET not in str(data)


def test_markdown_attachment_bodies_are_always_omitted(notebook_factory):
    png_body = "png-body-" + SECRET
    text_body = "attachment text " + SECRET
    path = notebook_factory(
        markdown="![plot](attachment:plot.png)",
        attachments={
            "plot.png": {"image/png": png_body},
            "notes.txt": {"text/plain": text_body},
        },
    )

    cell = build_snapshot(path)["cells"][0]

    assert cell["cell_type"] == "markdown"
    assert len(cell["attachments"]) == 2
    assert all(
        set(item) == {"mime_type", "original_chars", "sha256", "omitted"}
        for item in cell["attachments"]
    )
    assert all(item["omitted"] is True for item in cell["attachments"])
    assert png_body not in str(cell)
    assert text_body not in str(cell)
    assert SECRET not in str(cell)


def test_sanitization_summary_has_stable_labels_and_exact_counts(notebook_factory):
    stream_text = f"token={SECRET}\n" + "x" * 20_000
    png_body = "binary-" + SECRET
    attachment_body = "attachment-" + SECRET
    cell = nbformat.v4.new_code_cell(
        f"api_key='{SECRET}'",
        outputs=[
            nbformat.v4.new_output("stream", name="stdout", text=stream_text),
            nbformat.v4.new_output(
                "display_data",
                data={"image/png": png_body},
            ),
        ],
    )
    markdown = nbformat.v4.new_markdown_cell("attachment below")
    markdown["attachments"] = {"file.bin": {"application/octet-stream": attachment_body}}
    path = notebook_factory(cells=[cell, markdown])

    first = build_snapshot(path)
    second = build_snapshot(path)

    assert first["sanitization"] == second["sanitization"]
    assert first["sanitization"] == {
        "labels": ["openai_api_key"],
        "redacted_fields": 2,
        "truncated_fields": 1,
        "omitted_binary_fields": 2,
    }
    assert SECRET not in str(first)
