from __future__ import annotations

import json
from pathlib import Path

import nbformat

from notebook_coach.notebooks import build_snapshot
from notebook_coach.risk import scan_snapshot
from notebook_coach.workflows import validate_run


ROOT = Path(__file__).parents[1]
SAMPLE = ROOT / "samples/transformer_attention_buggy.ipynb"
EXAMPLE = ROOT / "examples/transformer_attention_buggy"


def test_transformer_sample_is_safe_small_and_has_three_teachable_signals():
    notebook = nbformat.read(SAMPLE, as_version=4)
    assert len(notebook.cells) < 12
    snapshot = build_snapshot(SAMPLE)
    assert scan_snapshot(snapshot) == {"blocked": False, "findings": []}

    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    assert "normalized_columns" in code
    assert "zip(*scores)" in code
    assert "Each key distributes attention across queries" in markdown
    assert "random.random" in code
    assert "random.seed" not in code
    for forbidden in (
        "requests",
        "urllib",
        "socket",
        "subprocess",
        "os.system",
        "open(",
        "write_text",
        "write_bytes",
        "!pip",
    ):
        assert forbidden not in code


def test_checked_in_example_is_answer_free_and_validates():
    assert validate_run(EXAMPLE) == EXAMPLE.resolve()
    challenge = nbformat.read(EXAMPLE / "challenge.ipynb", as_version=4)
    editable = [
        cell
        for cell in challenge.cells
        if cell.metadata.get("notebook_coach", {}).get("challenge_id")
    ]
    assert [
        cell.metadata.notebook_coach.challenge_id for cell in editable
    ] == ["C-CODE", "C-CONCEPT"]
    assert all("TODO" in cell.source for cell in editable)
    assert all("solution" not in cell.source.lower() for cell in editable)

    ledger = json.loads(
        (EXAMPLE / ".notebook-coach/execution-ledger.json").read_text("utf-8")
    )
    assert ledger["entries"] == []
    verification = json.loads(
        (EXAMPLE / ".notebook-coach/verification-state.json").read_text("utf-8")
    )
    statuses = {
        item["challenge_id"]: item["status"]
        for item in verification["assessment"]["challenge_results"]
    }
    assert statuses == {"C-CODE": "needs_work", "C-CONCEPT": "needs_work"}


def test_example_evaluation_is_honest_and_contains_no_private_data():
    evaluation = (EXAMPLE / "evaluation.md").read_text("utf-8")
    assert "GPT-5.6" in evaluation
    assert "code bug" in evaluation.lower()
    assert "misconception" in evaluation.lower()
    assert "reproducibility" in evaluation.lower()
    assert "challenge" in evaluation.lower()
    assert "source score" in evaluation.lower()
    assert "limitation" in evaluation.lower()
    assert "sk-" not in evaluation

