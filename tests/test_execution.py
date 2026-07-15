from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import nbformat
import pytest

from notebook_coach.execution import (
    ExecutionBlockedError,
    cancel_execution,
    execute_request,
    prepare_execution,
)
from notebook_coach.workflows import finalize_diagnosis, prepare_diagnosis


FIXED_NOW = datetime(2026, 7, 15, 9, tzinfo=timezone.utc)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finalized_run(source: Path, output_root: Path, assessment: dict) -> Path:
    prepared = prepare_diagnosis(source, output_root)
    value = json.loads(json.dumps(assessment))
    value["run_id"] = prepared.stage.run_id
    prepared.assessment_path.write_text(json.dumps(value), encoding="utf-8")
    return finalize_diagnosis(prepared.stage.stage_dir)


def _ledger(run_dir: Path) -> dict:
    return json.loads(
        (run_dir / ".notebook-coach/execution-ledger.json").read_text("utf-8")
    )


def _write_verification_state(run_dir: Path, source: Path, *, verifiable=True):
    baseline = json.loads((run_dir / "baseline.json").read_text("utf-8"))
    challenge = run_dir / "challenge.ipynb"
    state = {
        "schema_version": "1.0",
        "run_id": baseline["run_id"],
        "assessment_id": "v" * 64,
        "source_target": {
            "path": str(source.resolve()),
            "sha256": _sha256(source),
            "kernel_name": baseline["source"]["kernel_name"],
            "language": baseline["source"]["language"],
        },
        "challenge_target": {
            "path": str(challenge.resolve()),
            "sha256": _sha256(challenge),
            "kernel_name": baseline["source"]["kernel_name"],
        },
        "challenge_verifiability": {
            "verifiable": verifiable,
            "reason": None if verifiable else "metadata_mismatch",
        },
    }
    path = run_dir / ".notebook-coach/verification-state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    return state


def test_prepare_execution_reserves_bound_request_and_empty_temp_dir(
    notebook_path: Path, tmp_path: Path, valid_diagnosis: dict
):
    run_dir = _finalized_run(notebook_path, tmp_path / "runs", valid_diagnosis)

    prepared = prepare_execution(
        run_dir,
        phase="diagnosis",
        target="source",
        cell_timeout=11,
        total_timeout=22,
        now=lambda: FIXED_NOW,
    )

    request = prepared.request
    expected = {
        "schema_version",
        "run_id",
        "phase",
        "target",
        "attempt_id",
        "request_id",
        "target_path",
        "target_sha256",
        "analysis_id",
        "kernel_name",
        "cell_timeout",
        "total_timeout",
        "risk",
        "created_at",
        "expires_at",
        "temp_dir",
        "run_dir",
    }
    assert set(request) == expected
    assert request["attempt_id"] == "A001"
    assert request["target_path"] == str(notebook_path.resolve())
    assert request["target_sha256"] == _sha256(notebook_path)
    assert request["risk"]["blocked"] is False
    assert prepared.request_path.is_file()
    assert Path(request["temp_dir"]).is_dir()
    assert list(Path(request["temp_dir"]).iterdir()) == []


def test_risky_or_invalid_target_is_blocked_before_attempt_reservation(
    notebook_factory, tmp_path: Path, valid_diagnosis: dict
):
    source = notebook_factory(
        cells=[
            nbformat.v4.new_code_cell("import requests\nrequests.get('https://x')"),
            nbformat.v4.new_code_cell("value = 1"),
        ]
    )
    run_dir = _finalized_run(source, tmp_path / "runs", valid_diagnosis)

    with pytest.raises(ExecutionBlockedError, match="risk") as risk_error:
        prepare_execution(run_dir, phase="diagnosis", target="source")
    with pytest.raises(ExecutionBlockedError, match="target"):
        prepare_execution(run_dir, phase="diagnosis", target="challenge")

    assert risk_error.value.code == "execution_risk_blocked"
    assert _ledger(run_dir)["entries"] == []


def test_cancelled_request_is_removed_and_attempt_id_is_not_reused(
    notebook_path: Path, tmp_path: Path, valid_diagnosis: dict
):
    run_dir = _finalized_run(notebook_path, tmp_path / "runs", valid_diagnosis)
    first = prepare_execution(run_dir, phase="diagnosis", target="source")
    temp_dir = Path(first.request["temp_dir"])

    cancel_execution(first.request_path)
    second = prepare_execution(run_dir, phase="diagnosis", target="source")

    assert not first.request_path.exists()
    assert not temp_dir.exists()
    assert second.request["attempt_id"] == "A002"
    assert [entry["status"] for entry in _ledger(run_dir)["entries"]] == [
        "cancelled",
        "prepared",
    ]


def test_execute_requires_matching_confirmation_rejects_changes_and_replay(
    notebook_path: Path, tmp_path: Path, valid_diagnosis: dict
):
    run_dir = _finalized_run(notebook_path, tmp_path / "runs", valid_diagnosis)
    prepared = prepare_execution(run_dir, phase="diagnosis", target="source")

    with pytest.raises(ExecutionBlockedError) as mismatch:
        execute_request(prepared.request_path, "0" * 64)
    assert mismatch.value.code == "confirmation_hash_mismatch"
    assert _ledger(run_dir)["entries"][0]["status"] == "prepared"

    original = notebook_path.read_bytes()
    notebook_path.write_bytes(original + b"\n")
    with pytest.raises(ExecutionBlockedError) as changed:
        execute_request(prepared.request_path, prepared.request["target_sha256"])
    assert changed.value.code == "target_changed_before_execution"
    assert _ledger(run_dir)["entries"][0]["status"] == "failed"
    assert not Path(prepared.request["temp_dir"]).exists()

    notebook_path.write_bytes(original)
    second = prepare_execution(run_dir, phase="diagnosis", target="source")
    before = _sha256(notebook_path)
    log_path = execute_request(second.request_path, second.request["target_sha256"])
    assert _sha256(notebook_path) == before
    assert log_path.name == "execution-diagnosis-source-A002.json"
    assert json.loads(log_path.read_text("utf-8"))["evidence_eligible"] is True
    with pytest.raises(ExecutionBlockedError):
        execute_request(second.request_path, second.request["target_sha256"])


def test_verification_source_and_challenge_are_bound_separately(
    notebook_path: Path, tmp_path: Path, valid_diagnosis: dict
):
    run_dir = _finalized_run(notebook_path, tmp_path / "runs", valid_diagnosis)
    state = _write_verification_state(run_dir, notebook_path)

    source_request = prepare_execution(
        run_dir, phase="verification", target="source"
    )
    challenge_request = prepare_execution(
        run_dir, phase="verification", target="challenge"
    )

    assert source_request.request["target_path"] == state["source_target"]["path"]
    assert (
        challenge_request.request["target_path"]
        == state["challenge_target"]["path"]
    )
    assert source_request.request["analysis_id"] == state["assessment_id"]
    assert challenge_request.request["analysis_id"] == state["assessment_id"]


def test_unverifiable_challenge_is_rejected_without_reserving_attempt(
    notebook_path: Path, tmp_path: Path, valid_diagnosis: dict
):
    run_dir = _finalized_run(notebook_path, tmp_path / "runs", valid_diagnosis)
    _write_verification_state(run_dir, notebook_path, verifiable=False)

    with pytest.raises(ExecutionBlockedError) as error:
        prepare_execution(run_dir, phase="verification", target="challenge")

    assert error.value.code == "challenge_unverifiable"
    assert _ledger(run_dir)["entries"] == []


def test_expired_request_is_tombstoned_and_cleaned(
    notebook_path: Path, tmp_path: Path, valid_diagnosis: dict
):
    run_dir = _finalized_run(notebook_path, tmp_path / "runs", valid_diagnosis)
    prepared = prepare_execution(
        run_dir,
        phase="diagnosis",
        target="source",
        now=lambda: FIXED_NOW,
    )

    with pytest.raises(ExecutionBlockedError) as error:
        execute_request(
            prepared.request_path,
            prepared.request["target_sha256"],
            now=lambda: FIXED_NOW + timedelta(hours=2),
        )

    assert error.value.code == "request_expired"
    assert _ledger(run_dir)["entries"][0]["status"] == "expired"
    assert not Path(prepared.request["temp_dir"]).exists()


def test_concurrent_preparations_receive_distinct_attempt_ids(
    notebook_path: Path, tmp_path: Path, valid_diagnosis: dict
):
    run_dir = _finalized_run(notebook_path, tmp_path / "runs", valid_diagnosis)

    def prepare(_index: int) -> str:
        return prepare_execution(
            run_dir, phase="diagnosis", target="source"
        ).request["attempt_id"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        attempt_ids = list(pool.map(prepare, range(6)))

    assert sorted(attempt_ids) == [f"A{index:03d}" for index in range(1, 7)]


def test_orphaned_running_attempt_is_failed_and_cleaned_before_reservation(
    notebook_path: Path, tmp_path: Path, valid_diagnosis: dict
):
    run_dir = _finalized_run(notebook_path, tmp_path / "runs", valid_diagnosis)
    first = prepare_execution(
        run_dir,
        phase="diagnosis",
        target="source",
        total_timeout=10,
        now=lambda: FIXED_NOW,
    )
    ledger_path = run_dir / ".notebook-coach/execution-ledger.json"
    ledger = json.loads(ledger_path.read_text("utf-8"))
    ledger["entries"][0]["status"] = "running"
    ledger["entries"][0]["updated_at"] = FIXED_NOW.isoformat()
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    second = prepare_execution(
        run_dir,
        phase="diagnosis",
        target="source",
        now=lambda: FIXED_NOW + timedelta(seconds=71),
    )

    recovered = _ledger(run_dir)
    assert second.request["attempt_id"] == "A002"
    assert recovered["entries"][0]["status"] == "failed"
    assert recovered["entries"][0]["details"]["reason"] == "orphaned_worker"
    assert not Path(first.request["temp_dir"]).exists()


def test_total_timeout_terminates_worker_and_cleans_temp_under_five_seconds(
    notebook_factory, tmp_path: Path, valid_diagnosis: dict
):
    source = notebook_factory(
        cells=[
            nbformat.v4.new_code_cell("while True:\n    pass"),
            nbformat.v4.new_code_cell("value = 1"),
        ]
    )
    run_dir = _finalized_run(source, tmp_path / "runs", valid_diagnosis)
    prepared = prepare_execution(
        run_dir,
        phase="diagnosis",
        target="source",
        cell_timeout=10,
        total_timeout=1,
    )
    started = time.monotonic()

    log_path = execute_request(
        prepared.request_path, prepared.request["target_sha256"]
    )

    elapsed = time.monotonic() - started
    log = json.loads(log_path.read_text("utf-8"))
    assert elapsed < 5
    assert log["status"] == "timeout"
    assert log["evidence_eligible"] is True
    assert not Path(prepared.request["temp_dir"]).exists()


def test_cell_timeout_is_eligible_and_preserves_completed_cell_evidence(
    notebook_factory, tmp_path: Path, valid_diagnosis: dict
):
    source = notebook_factory(
        cells=[
            nbformat.v4.new_code_cell("print('completed-step')"),
            nbformat.v4.new_code_cell("while True:\n    pass"),
        ]
    )
    run_dir = _finalized_run(source, tmp_path / "runs", valid_diagnosis)
    prepared = prepare_execution(
        run_dir,
        phase="diagnosis",
        target="source",
        cell_timeout=1,
        total_timeout=12,
    )

    log_path = execute_request(
        prepared.request_path, prepared.request["target_sha256"]
    )
    log = json.loads(log_path.read_text("utf-8"))

    assert log["status"] == "timeout"
    assert log["evidence_eligible"] is True
    assert log["worker"]["timed_out"] is True
    assert log["worker"]["timeout_cell_index"] == 1
    assert "completed-step" in log_path.read_text("utf-8")


def test_execution_log_redacts_and_bounds_notebook_output(
    notebook_factory, tmp_path: Path, valid_diagnosis: dict
):
    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
    source = notebook_factory(
        cells=[
            nbformat.v4.new_code_cell(
                f"print('{secret}')\nprint('x' * 20000)"
            ),
            nbformat.v4.new_code_cell("value = 1"),
        ]
    )
    run_dir = _finalized_run(source, tmp_path / "runs", valid_diagnosis)
    prepared = prepare_execution(run_dir, phase="diagnosis", target="source")

    log_path = execute_request(
        prepared.request_path, prepared.request["target_sha256"]
    )
    body = log_path.read_text("utf-8")

    assert secret not in body
    assert "[REDACTED]" in body
    assert len(body) < 20_000


def test_tampered_reserved_paths_are_rejected_without_deleting_unrelated_dirs(
    notebook_path: Path, tmp_path: Path, valid_diagnosis: dict
):
    run_dir = _finalized_run(notebook_path, tmp_path / "runs", valid_diagnosis)
    victim = tmp_path / "unrelated-empty-directory"
    victim.mkdir()
    prepared = prepare_execution(run_dir, phase="diagnosis", target="source")
    request = json.loads(prepared.request_path.read_text("utf-8"))
    request["temp_dir"] = str(victim)
    prepared.request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ExecutionBlockedError) as error:
        execute_request(prepared.request_path, prepared.request["target_sha256"])

    assert error.value.code == "request_binding_mismatch"
    assert victim.is_dir()
    assert _ledger(run_dir)["entries"][0]["status"] == "prepared"


def test_cancel_rejects_tampered_request_before_cleanup(
    notebook_path: Path, tmp_path: Path, valid_diagnosis: dict
):
    run_dir = _finalized_run(notebook_path, tmp_path / "runs", valid_diagnosis)
    victim = tmp_path / "unrelated-empty-directory"
    victim.mkdir()
    prepared = prepare_execution(run_dir, phase="diagnosis", target="source")
    request = json.loads(prepared.request_path.read_text("utf-8"))
    request["temp_dir"] = str(victim)
    prepared.request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ExecutionBlockedError) as error:
        cancel_execution(prepared.request_path)

    assert error.value.code == "request_binding_mismatch"
    assert victim.is_dir()
    assert _ledger(run_dir)["entries"][0]["status"] == "prepared"
