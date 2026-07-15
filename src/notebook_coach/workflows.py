"""Two-phase, non-executing diagnosis workflows and run validation."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nbformat

from notebook_coach import SCHEMA_VERSION
from notebook_coach.contracts import ContractError, validate_diagnosis
from notebook_coach.notebooks import (
    NotebookInputError,
    build_snapshot,
    parse_cell_selection,
)
from notebook_coach.render import (
    build_baseline,
    build_challenge_notebook,
    build_report_state,
    render_report,
)
from notebook_coach.risk import scan_snapshot
from notebook_coach.runs import RunStore, Stage
from notebook_coach.scoring import score_issues


class WorkflowError(ValueError):
    """Stable workflow error suitable for machine-readable CLI output."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class DiagnosisPreparation:
    stage: Stage
    assessment_path: Path
    risk: dict[str, Any]


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _read_json_object(path: Path, *, code: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise WorkflowError(code, f"{label} is missing or invalid JSON.") from None
    if not isinstance(value, dict):
        raise WorkflowError(code, f"{label} must be a JSON object.")
    return value


def _selected_indexes(source: str | Path, cells: str | None) -> list[int] | None:
    if cells is None:
        return None
    try:
        payload = json.loads(Path(source).read_text("utf-8"))
        notebook_cells = payload["cells"]
        if not isinstance(notebook_cells, list):
            raise TypeError
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
        # Reuse the notebook reader's stable, non-sensitive validation error.
        build_snapshot(source)
        raise NotebookInputError("Notebook structure is invalid or unsupported.")
    return parse_cell_selection(cells, len(notebook_cells))


def prepare_diagnosis(
    source: str | Path,
    output_root: str | Path = "notebook-coach-output",
    *,
    cells: str | None = None,
) -> DiagnosisPreparation:
    """Create static diagnosis inputs without executing or changing the source."""

    snapshot = build_snapshot(source, _selected_indexes(source, cells))
    risk = scan_snapshot(snapshot)
    source_metadata = snapshot.get("source")
    source_sha256 = (
        source_metadata.get("sha256") if isinstance(source_metadata, dict) else None
    )
    if not isinstance(source_sha256, str):
        raise WorkflowError("snapshot_invalid", "Snapshot source hash is invalid.")

    store = RunStore(output_root)
    stage = store.create_stage(source, source_sha256)
    assessment_path = stage.stage_dir / "diagnosis-assessment.json"
    try:
        (stage.stage_dir / "snapshot.json").write_bytes(_json_bytes(snapshot))
        (stage.stage_dir / "risk.json").write_bytes(_json_bytes(risk))
        assessment_path.touch(exist_ok=False)
    except Exception:
        shutil.rmtree(stage.stage_dir, ignore_errors=True)
        raise
    return DiagnosisPreparation(
        stage=stage,
        assessment_path=assessment_path.resolve(),
        risk=risk,
    )


def finalize_diagnosis(stage_dir: str | Path) -> Path:
    """Validate a staged assessment and atomically publish final artifacts."""

    directory = Path(stage_dir).expanduser().resolve(strict=False)
    if directory.parent.name != ".staging":
        raise WorkflowError("invalid_stage", "Diagnosis stage path is invalid.")
    store = RunStore(directory.parent.parent)
    stage = store.load_stage(directory)
    snapshot = _read_json_object(
        directory / "snapshot.json", code="snapshot_invalid", label="Snapshot"
    )
    _read_json_object(directory / "risk.json", code="risk_invalid", label="Risk scan")
    try:
        assessment_value = json.loads(
            (directory / "diagnosis-assessment.json").read_text("utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise ContractError(
            "diagnosis_json", "Diagnosis assessment is missing or invalid JSON."
        ) from None

    source = snapshot.get("source")
    if not isinstance(source, dict) or source.get("sha256") != stage.source_sha256:
        raise WorkflowError(
            "source_hash_mismatch", "Stage and snapshot source hashes do not match."
        )
    cell_count = source.get("cell_count")
    assessment = validate_diagnosis(
        assessment_value,
        run_id=stage.run_id,
        cell_count=cell_count,
    )
    baseline = build_baseline(snapshot, assessment, source_path=stage.source_path)
    report_state = build_report_state(baseline)
    report = render_report(baseline, report_state)
    challenge = build_challenge_notebook(assessment)
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "next_attempt": 1,
        "entries": [],
    }
    files = {
        "baseline.json": _json_bytes(baseline),
        "report.md": report.encode("utf-8"),
        "challenge.ipynb": nbformat.writes(challenge, version=4).encode("utf-8"),
        ".notebook-coach/report-state.json": _json_bytes(report_state),
        ".notebook-coach/execution-ledger.json": _json_bytes(ledger),
    }
    return store.finalize_stage(stage, files).resolve()


def _analysis_id(baseline: dict[str, Any]) -> str:
    unsigned = copy.deepcopy(baseline)
    unsigned.pop("analysis_id", None)
    body = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def validate_run(run_dir: str | Path) -> Path:
    """Verify deterministic artifacts and immutable identity for one run."""

    directory = Path(run_dir).expanduser().resolve(strict=False)
    required = (
        "baseline.json",
        "report.md",
        "challenge.ipynb",
        ".notebook-coach/report-state.json",
        ".notebook-coach/execution-ledger.json",
    )
    if not directory.is_dir() or any(
        not (directory / relative).is_file() for relative in required
    ):
        raise WorkflowError(
            "missing_artifact", "Run directory is missing a required artifact."
        )

    baseline = _read_json_object(
        directory / "baseline.json", code="baseline_invalid", label="Baseline"
    )
    report_state = _read_json_object(
        directory / ".notebook-coach/report-state.json",
        code="report_state_invalid",
        label="Report state",
    )
    ledger = _read_json_object(
        directory / ".notebook-coach/execution-ledger.json",
        code="ledger_invalid",
        label="Execution ledger",
    )
    run_id = baseline.get("run_id")
    if (
        not isinstance(run_id, str)
        or run_id != directory.name
        or report_state.get("run_id") != run_id
    ):
        raise WorkflowError("run_id_mismatch", "Run artifact IDs do not match.")
    if baseline.get("immutable") is not True:
        raise WorkflowError(
            "baseline_mutable", "Baseline immutability marker is missing."
        )
    if baseline.get("analysis_id") != _analysis_id(baseline):
        raise WorkflowError(
            "analysis_id_mismatch", "Baseline immutable analysis ID is invalid."
        )
    try:
        expected_score = score_issues(baseline["issues"])
    except (KeyError, TypeError, ValueError):
        raise WorkflowError("score_invalid", "Baseline score inputs are invalid.") from None
    if baseline.get("score") != expected_score:
        raise WorkflowError("score_mismatch", "Baseline score arithmetic is invalid.")

    try:
        challenge = nbformat.read(directory / "challenge.ipynb", as_version=4)
        metadata = challenge.metadata["notebook_coach"]
        expected_challenge_ids = [
            item["challenge_id"] for item in baseline["challenges"]
        ]
        if (
            metadata.get("run_id") != run_id
            or list(metadata.get("challenge_ids", [])) != expected_challenge_ids
        ):
            raise KeyError
    except Exception:
        raise WorkflowError(
            "challenge_metadata", "Challenge notebook metadata is invalid."
        ) from None

    try:
        actual_report = (directory / "report.md").read_text("utf-8")
        expected_report = render_report(baseline, report_state)
    except (OSError, UnicodeError, KeyError, TypeError, ValueError):
        raise WorkflowError("report_invalid", "Report cannot be regenerated.") from None
    if actual_report != expected_report:
        raise WorkflowError(
            "report_mismatch", "Report does not match its structured state."
        )
    if (
        ledger.get("schema_version") != SCHEMA_VERSION
        or not isinstance(ledger.get("next_attempt"), int)
        or isinstance(ledger.get("next_attempt"), bool)
        or ledger["next_attempt"] < 1
        or not isinstance(ledger.get("entries"), list)
    ):
        raise WorkflowError("ledger_invalid", "Execution ledger is invalid.")
    return directory
