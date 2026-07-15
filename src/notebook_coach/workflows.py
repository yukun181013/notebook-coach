"""Two-phase, non-executing diagnosis workflows and run validation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nbformat

from notebook_coach import SCHEMA_VERSION
from notebook_coach.contracts import (
    ContractError,
    validate_diagnosis,
    validate_verification,
)
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
    render_verification,
)
from notebook_coach.risk import scan_snapshot
from notebook_coach.runs import RunStore, Stage
from notebook_coach.scoring import score_issues
from notebook_coach.scoring import score_verification
from notebook_coach.sanitize import summarize_text


class WorkflowError(ValueError):
    """Stable workflow error suitable for machine-readable CLI output."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class ConfirmationRequiredError(WorkflowError):
    def __init__(self, code: str, message: str, details: dict[str, Any]) -> None:
        self.details = details
        super().__init__(code, message)


@dataclass(frozen=True)
class DiagnosisPreparation:
    stage: Stage
    assessment_path: Path
    risk: dict[str, Any]


@dataclass(frozen=True)
class VerificationPreparation:
    run_id: str
    stage_dir: Path
    assessment_path: Path
    source_snapshot_path: Path
    challenge_snapshot_path: Path
    challenge_verifiability: dict[str, Any]


@dataclass(frozen=True)
class VerificationResult:
    run_dir: Path
    report_path: Path
    state_path: Path
    before_score: dict[str, Any]
    after_score: dict[str, Any]
    challenge_results: dict[str, str]


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
    snapshot_cells = snapshot.get("cells")
    if not isinstance(snapshot_cells, list) or any(
        not isinstance(cell, dict)
        or isinstance(cell.get("index"), bool)
        or not isinstance(cell.get("index"), int)
        for cell in snapshot_cells
    ):
        raise WorkflowError("snapshot_invalid", "Snapshot cells are invalid.")
    allowed_cell_indices = {cell["index"] for cell in snapshot_cells}
    if len(allowed_cell_indices) != len(snapshot_cells):
        raise WorkflowError("snapshot_invalid", "Snapshot cells are invalid.")
    assessment = validate_diagnosis(
        assessment_value,
        run_id=stage.run_id,
        cell_count=cell_count,
        allowed_cell_indices=allowed_cell_indices,
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


def _verification_assessment_id(
    *,
    run_id: str,
    source_sha256: Any,
    challenge_sha256: Any,
    assessment: dict[str, Any],
) -> str:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "source_sha256": source_sha256,
        "challenge_sha256": challenge_sha256,
        "assessment": assessment,
    }
    return hashlib.sha256(_json_bytes(identity).rstrip(b"\n")).hexdigest()


def _validate_verification_artifacts(
    directory: Path, baseline: dict[str, Any]
) -> None:
    report_path = directory / "verification.md"
    state_path = directory / ".notebook-coach/verification-state.json"
    if report_path.is_file() != state_path.is_file():
        raise WorkflowError(
            "verification_artifact_pair",
            "Verification report and state must either both exist or both be absent.",
        )
    if not report_path.is_file():
        return

    state = _read_json_object(
        state_path,
        code="verification_state_invalid",
        label="Verification state",
    )
    expected_keys = {
        "schema_version",
        "run_id",
        "source_target",
        "challenge_target",
        "challenge_verifiability",
        "assessment",
        "before_score",
        "after_score",
        "assessment_id",
        "revision",
        "execution_reviews",
    }
    source_target = state.get("source_target")
    challenge_target = state.get("challenge_target")
    verifiability = state.get("challenge_verifiability")
    baseline_issues = baseline.get("issues")
    baseline_source = baseline.get("source")
    revision = state.get("revision")
    reviews = state.get("execution_reviews")
    if (
        set(state) != expected_keys
        or state.get("schema_version") != SCHEMA_VERSION
        or state.get("run_id") != baseline.get("run_id")
        or not isinstance(source_target, dict)
        or not isinstance(challenge_target, dict)
        or not isinstance(verifiability, dict)
        or not isinstance(verifiability.get("verifiable"), bool)
        or not isinstance(baseline_issues, list)
        or not isinstance(baseline_source, dict)
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
        or not isinstance(reviews, list)
        or revision != len(reviews) + 1
    ):
        raise WorkflowError(
            "verification_state_invalid", "Verification state is invalid."
        )
    cell_count = source_target.get("cell_count", baseline_source.get("cell_count"))
    if (
        isinstance(cell_count, bool)
        or not isinstance(cell_count, int)
        or cell_count < 0
    ):
        raise WorkflowError(
            "verification_state_invalid", "Verification state is invalid."
        )
    try:
        assessment = validate_verification(
            state.get("assessment"),
            run_id=baseline["run_id"],
            baseline_issue_ids=[item["issue_id"] for item in baseline_issues],
            cell_count=cell_count,
            challenge_verifiable=verifiability["verifiable"],
        )
        after_score = score_verification(
            baseline_issues,
            assessment["issue_results"],
            assessment["new_issues"],
        )
    except (ContractError, KeyError, TypeError, ValueError):
        raise WorkflowError(
            "verification_state_invalid", "Verification state is invalid."
        ) from None
    if state.get("before_score") != baseline.get("score") or state.get(
        "after_score"
    ) != after_score:
        raise WorkflowError(
            "verification_score_mismatch",
            "Verification score arithmetic is invalid.",
        )
    expected_assessment_id = _verification_assessment_id(
        run_id=baseline["run_id"],
        source_sha256=source_target.get("sha256"),
        challenge_sha256=challenge_target.get("sha256"),
        assessment=assessment,
    )
    if state.get("assessment_id") != expected_assessment_id:
        raise WorkflowError(
            "verification_assessment_id_mismatch",
            "Verification assessment ID is invalid.",
        )
    try:
        actual_report = report_path.read_text("utf-8")
        expected_report = render_verification(state)
    except (OSError, UnicodeError, KeyError, TypeError, ValueError):
        raise WorkflowError(
            "verification_report_invalid",
            "Verification report cannot be regenerated.",
        ) from None
    if actual_report != expected_report:
        raise WorkflowError(
            "verification_report_mismatch",
            "Verification report does not match its structured state.",
        )


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
    _validate_verification_artifacts(directory, baseline)
    return directory


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        raise WorkflowError("target_unreadable", "Verification target is unreadable.") from None


def _challenge_snapshot(
    challenge_path: Path, baseline: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_sha256: str | None = None
    reason: str | None = None
    safe_cells: list[dict[str, Any]] = []
    initial_hashes: dict[str, str] = {}
    current_hashes: dict[str, str] = {}
    try:
        target_sha256 = _file_sha256(challenge_path)
        notebook = nbformat.read(challenge_path, as_version=4)
        metadata = notebook.metadata.get("notebook_coach", {})
        expected_ids = [item["challenge_id"] for item in baseline["challenges"]]
        if (
            not isinstance(metadata, dict)
            or metadata.get("run_id") != baseline["run_id"]
            or list(metadata.get("challenge_ids", [])) != expected_ids
            or not isinstance(metadata.get("initial_content_hashes"), dict)
        ):
            reason = "metadata_mismatch"
        else:
            initial_hashes = dict(metadata["initial_content_hashes"])
        static_snapshot = build_snapshot(challenge_path)
        snapshot_cells = {item["index"]: item for item in static_snapshot["cells"]}
        seen: list[str] = []
        for index, cell in enumerate(notebook.cells):
            cell_metadata = cell.metadata.get("notebook_coach", {})
            challenge_id = (
                cell_metadata.get("challenge_id")
                if isinstance(cell_metadata, dict)
                else None
            )
            if challenge_id not in {"C-CODE", "C-CONCEPT"}:
                continue
            seen.append(challenge_id)
            raw_source = str(cell.source)
            current_hashes[challenge_id] = hashlib.sha256(
                raw_source.encode("utf-8")
            ).hexdigest()
            safe = snapshot_cells[index]
            safe_cells.append(
                {
                    "cell_index": index,
                    "challenge_id": challenge_id,
                    "cell_type": safe["cell_type"],
                    "source": summarize_text(
                        safe["source"]["text"], max_chars=4_000
                    ),
                    "outputs": safe.get("outputs", []),
                }
            )
        if seen != ["C-CODE", "C-CONCEPT"]:
            reason = "metadata_mismatch"
        if set(initial_hashes) != {"C-CODE", "C-CONCEPT"}:
            reason = "metadata_mismatch"
    except Exception:
        reason = "challenge_unreadable"
        safe_cells = []
        initial_hashes = {}
        current_hashes = {}
    verifiability = {"verifiable": reason is None, "reason": reason}
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": baseline["run_id"],
            "target_sha256": target_sha256,
            "initial_content_hashes": initial_hashes,
            "current_content_hashes": current_hashes,
            "cells": safe_cells,
            "verifiability": verifiability,
        },
        verifiability,
    )


def prepare_verification(
    source: str | Path,
    run_dir: str | Path,
    *,
    confirm_source_mismatch: bool = False,
    confirm_environment_mismatch: bool = False,
) -> VerificationPreparation:
    directory = Path(run_dir).expanduser().resolve(strict=False)
    baseline = _read_json_object(
        directory / "baseline.json", code="baseline_invalid", label="Baseline"
    )
    run_id = baseline.get("run_id")
    baseline_source = baseline.get("source")
    if not isinstance(run_id, str) or not isinstance(baseline_source, dict):
        raise WorkflowError("baseline_invalid", "Baseline identity is invalid.")
    selected_path = Path(source).expanduser().resolve(strict=False)
    baseline_path = Path(str(baseline_source.get("path"))).resolve(strict=False)
    if selected_path != baseline_path and not confirm_source_mismatch:
        raise ConfirmationRequiredError(
            "source_confirmation_required",
            "Selected source path differs from the baseline.",
            {
                "baseline_path": str(baseline_path),
                "selected_path": str(selected_path),
            },
        )
    source_snapshot = build_snapshot(selected_path)
    current_source = source_snapshot["source"]
    baseline_kernel = baseline_source.get("kernel_name")
    baseline_language = baseline_source.get("language")
    current_kernel = current_source.get("kernel_name")
    current_language = current_source.get("language")
    if (
        current_kernel != baseline_kernel or current_language != baseline_language
    ) and not confirm_environment_mismatch:
        raise ConfirmationRequiredError(
            "environment_confirmation_required",
            "Current kernel or language differs from the baseline.",
            {
                "baseline_kernel": baseline_kernel,
                "current_kernel": current_kernel,
                "baseline_language": baseline_language,
                "current_language": current_language,
            },
        )

    challenge_path = directory / "challenge.ipynb"
    challenge_snapshot, verifiability = _challenge_snapshot(challenge_path, baseline)
    stage_dir = (
        directory
        / ".notebook-coach"
        / "verification-staging"
        / uuid.uuid4().hex
    )
    stage_dir.mkdir(parents=True, exist_ok=False)
    assessment_path = stage_dir / "verification-assessment.json"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_dir": str(directory),
        "source_target": {
            "path": str(selected_path),
            "sha256": current_source["sha256"],
            "kernel_name": current_kernel,
            "language": current_language,
            "cell_count": current_source["cell_count"],
        },
        "challenge_target": {
            "path": str(challenge_path.resolve()),
            "sha256": challenge_snapshot["target_sha256"],
            "kernel_name": baseline_kernel,
        },
        "challenge_verifiability": verifiability,
    }
    try:
        (stage_dir / "stage.json").write_bytes(_json_bytes(metadata))
        (stage_dir / "source-snapshot.json").write_bytes(_json_bytes(source_snapshot))
        (stage_dir / "baseline-issues.json").write_bytes(
            _json_bytes(baseline["issues"])
        )
        challenge_snapshot_path = stage_dir / "challenge-snapshot.json"
        challenge_snapshot_path.write_bytes(_json_bytes(challenge_snapshot))
        assessment_path.touch(exist_ok=False)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    return VerificationPreparation(
        run_id=run_id,
        stage_dir=stage_dir.resolve(),
        assessment_path=assessment_path.resolve(),
        source_snapshot_path=(stage_dir / "source-snapshot.json").resolve(),
        challenge_snapshot_path=challenge_snapshot_path.resolve(),
        challenge_verifiability=verifiability,
    )


def _replace_pair(
    state_path: Path, state_body: bytes, report_path: Path, report_body: bytes
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    old_state = state_path.read_bytes() if state_path.exists() else None
    old_report = report_path.read_bytes() if report_path.exists() else None
    try:
        for destination, body in (
            (state_path, state_body),
            (report_path, report_body),
        ):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary_paths[0], state_path)
        os.replace(temporary_paths[1], report_path)
    except Exception:
        if old_state is None:
            state_path.unlink(missing_ok=True)
        else:
            state_path.write_bytes(old_state)
        if old_report is None:
            report_path.unlink(missing_ok=True)
        else:
            report_path.write_bytes(old_report)
        raise
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


def finalize_verification(
    stage_dir: str | Path, assessment_value: dict[str, Any] | None = None
) -> VerificationResult:
    stage = Path(stage_dir).expanduser().resolve(strict=False)
    metadata = _read_json_object(
        stage / "stage.json", code="verification_stage_invalid", label="Stage"
    )
    run_dir = Path(str(metadata.get("run_dir"))).resolve(strict=False)
    baseline = _read_json_object(
        run_dir / "baseline.json", code="baseline_invalid", label="Baseline"
    )
    source_snapshot = _read_json_object(
        stage / "source-snapshot.json",
        code="snapshot_invalid",
        label="Source snapshot",
    )
    if assessment_value is None:
        try:
            assessment_value = json.loads(
                (stage / "verification-assessment.json").read_text("utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
            raise ContractError(
                "verification_json",
                "Verification assessment is missing or invalid JSON.",
            ) from None
    source_target = metadata.get("source_target")
    challenge_target = metadata.get("challenge_target")
    verifiability = metadata.get("challenge_verifiability")
    if not all(
        isinstance(value, dict)
        for value in (source_target, challenge_target, verifiability)
    ):
        raise WorkflowError("verification_stage_invalid", "Stage targets are invalid.")
    source_changed = (
        _file_sha256(Path(source_target["path"])) != source_target["sha256"]
    )
    challenge_changed = False
    if verifiability.get("verifiable"):
        challenge_changed = (
            _file_sha256(Path(challenge_target["path"]))
            != challenge_target["sha256"]
        )
    if source_changed or challenge_changed:
        raise WorkflowError(
            "analysis_target_changed", "Verification target changed after staging."
        )
    baseline_issues = baseline.get("issues")
    if not isinstance(baseline_issues, list):
        raise WorkflowError("baseline_invalid", "Baseline issues are invalid.")
    assessment = validate_verification(
        assessment_value,
        run_id=metadata["run_id"],
        baseline_issue_ids=[item["issue_id"] for item in baseline_issues],
        cell_count=source_snapshot["source"]["cell_count"],
        challenge_verifiable=bool(verifiability["verifiable"]),
        allowed_cell_indices={
            cell["index"] for cell in source_snapshot["cells"]
        },
    )
    before_score = baseline["score"]
    after_score = score_verification(
        baseline_issues, assessment["issue_results"], assessment["new_issues"]
    )
    assessment_id = _verification_assessment_id(
        run_id=metadata["run_id"],
        source_sha256=source_target["sha256"],
        challenge_sha256=challenge_target["sha256"],
        assessment=assessment,
    )
    state = {
        "schema_version": SCHEMA_VERSION,
        "run_id": metadata["run_id"],
        "source_target": source_target,
        "challenge_target": challenge_target,
        "challenge_verifiability": verifiability,
        "assessment": assessment,
        "before_score": before_score,
        "after_score": after_score,
        "assessment_id": assessment_id,
        "revision": 1,
        "execution_reviews": [],
    }
    report = render_verification(state)
    state_path = run_dir / ".notebook-coach/verification-state.json"
    report_path = run_dir / "verification.md"
    _replace_pair(
        state_path,
        _json_bytes(state),
        report_path,
        report.encode("utf-8"),
    )
    shutil.rmtree(stage)
    return VerificationResult(
        run_dir=run_dir,
        report_path=report_path,
        state_path=state_path,
        before_score=before_score,
        after_score=after_score,
        challenge_results={
            item["challenge_id"]: item["status"]
            for item in assessment["challenge_results"]
        },
    )
