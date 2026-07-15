from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat
import pytest

from notebook_coach.contracts import ContractError
from notebook_coach.workflows import finalize_diagnosis, prepare_diagnosis


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_diagnosis_creates_only_staging_inputs(
    notebook_path: Path, tmp_path: Path
):
    output_root = tmp_path / "runs"

    prepared = prepare_diagnosis(notebook_path, output_root)

    assert prepared.stage.stage_dir.parent == output_root.resolve() / ".staging"
    assert (prepared.stage.stage_dir / "snapshot.json").is_file()
    assert (prepared.stage.stage_dir / "risk.json").is_file()
    assert prepared.assessment_path.read_bytes() == b""
    assert prepared.risk == {"blocked": False, "findings": []}
    assert not prepared.stage.final_dir.exists()


def test_static_flow_preserves_source_hash_with_checked_in_assessment(
    notebook_path: Path, tmp_path: Path
):
    before = _source_hash(notebook_path)
    prepared = prepare_diagnosis(notebook_path, tmp_path / "runs")
    fixture_path = Path(__file__).parent / "fixtures/diagnosis-assessment.json"
    assessment = json.loads(fixture_path.read_text("utf-8"))
    assessment["run_id"] = prepared.stage.run_id
    prepared.assessment_path.write_text(json.dumps(assessment), encoding="utf-8")

    run_dir = finalize_diagnosis(prepared.stage.stage_dir)

    assert _source_hash(notebook_path) == before
    assert run_dir == prepared.stage.final_dir
    assert not prepared.stage.stage_dir.exists()
    required = {
        "baseline.json",
        "report.md",
        "challenge.ipynb",
        ".notebook-coach/report-state.json",
        ".notebook-coach/execution-ledger.json",
    }
    assert all((run_dir / relative).is_file() for relative in required)
    baseline = json.loads((run_dir / "baseline.json").read_text("utf-8"))
    assert baseline["run_id"] == prepared.stage.run_id
    assert baseline["source"]["sha256"] == before
    assert baseline["source"]["path"] == str(notebook_path.resolve())
    assert baseline["immutable"] is True
    assert nbformat.read(run_dir / "challenge.ipynb", as_version=4).metadata[
        "notebook_coach"
    ]["run_id"] == prepared.stage.run_id
    ledger = json.loads(
        (run_dir / ".notebook-coach/execution-ledger.json").read_text("utf-8")
    )
    assert ledger["next_attempt"] == 1
    assert ledger["entries"] == []


def test_invalid_assessment_leaves_stage_and_no_final_run(
    notebook_path: Path, tmp_path: Path
):
    prepared = prepare_diagnosis(notebook_path, tmp_path / "runs")
    prepared.assessment_path.write_text(
        json.dumps({"schema_version": "1.0", "run_id": prepared.stage.run_id}),
        encoding="utf-8",
    )

    with pytest.raises(ContractError):
        finalize_diagnosis(prepared.stage.stage_dir)

    assert prepared.stage.stage_dir.is_dir()
    assert not prepared.stage.final_dir.exists()

