"""Hash-bound, confirmed execution of one notebook target in a temp copy."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from notebook_coach import SCHEMA_VERSION
from notebook_coach.notebooks import build_snapshot
from notebook_coach.risk import scan_snapshot
from notebook_coach.runs import LedgerTransitionError, RunStore, RunStoreError
from notebook_coach.sanitize import summarize_text


_SHA256 = re.compile(r"[0-9a-f]{64}")
_EVIDENCE_STATUSES = {"completed", "completed_with_cell_errors", "timeout"}


class ExecutionBlockedError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ExecutionPreparation:
    request_path: Path
    request: dict[str, Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        raise ExecutionBlockedError(
            "target_unreadable", "Execution target is missing or unreadable."
        ) from None
    return digest.hexdigest()


def _read_object(path: Path, *, code: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise ExecutionBlockedError(code, f"{label} is missing or invalid.") from None
    if not isinstance(value, dict):
        raise ExecutionBlockedError(code, f"{label} is missing or invalid.")
    return value


def _target_binding(run_dir: Path, phase: str, target: str) -> dict[str, Any]:
    if phase == "diagnosis":
        if target != "source":
            raise ExecutionBlockedError(
                "invalid_execution_target",
                "Diagnosis execution target must be source.",
            )
        baseline = _read_object(
            run_dir / "baseline.json", code="baseline_invalid", label="Baseline"
        )
        source = baseline.get("source")
        if not isinstance(source, dict):
            raise ExecutionBlockedError("baseline_invalid", "Baseline is invalid.")
        binding = {
            "run_id": baseline.get("run_id"),
            "path": source.get("path"),
            "sha256": source.get("sha256"),
            "kernel_name": source.get("kernel_name"),
            "analysis_id": baseline.get("analysis_id"),
        }
    elif phase == "verification" and target in {"source", "challenge"}:
        state = _read_object(
            run_dir / ".notebook-coach/verification-state.json",
            code="verification_state_invalid",
            label="Verification state",
        )
        if target == "challenge":
            verifiability = state.get("challenge_verifiability")
            if not isinstance(verifiability, dict) or not verifiability.get(
                "verifiable"
            ):
                raise ExecutionBlockedError(
                    "challenge_unverifiable",
                    "Challenge metadata is not verifiable in this cycle.",
                )
        target_state = state.get(f"{target}_target")
        if not isinstance(target_state, dict):
            raise ExecutionBlockedError(
                "verification_state_invalid", "Verification target is invalid."
            )
        binding = {
            "run_id": state.get("run_id"),
            "path": target_state.get("path"),
            "sha256": target_state.get("sha256"),
            "kernel_name": target_state.get("kernel_name"),
            "analysis_id": state.get("assessment_id"),
        }
    else:
        raise ExecutionBlockedError(
            "invalid_execution_target", "Execution phase or target is invalid."
        )
    if (
        not isinstance(binding["run_id"], str)
        or not isinstance(binding["path"], str)
        or not isinstance(binding["kernel_name"], str)
        or not isinstance(binding["analysis_id"], str)
        or _SHA256.fullmatch(str(binding["sha256"])) is None
    ):
        raise ExecutionBlockedError(
            "analysis_target_invalid", "Static analysis target binding is invalid."
        )
    binding["path"] = str(Path(binding["path"]).expanduser().resolve(strict=False))
    return binding


def _cleanup_paths(metadata: dict[str, Any]) -> None:
    temp_dir = metadata.get("temp_dir")
    if isinstance(temp_dir, str):
        shutil.rmtree(temp_dir, ignore_errors=True)
    request_path = metadata.get("request_path")
    if isinstance(request_path, str):
        request = Path(request_path)
        if request.parent.name:
            shutil.rmtree(request.parent, ignore_errors=True)


def _write_new(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise ExecutionBlockedError(
            "artifact_exists", "Refusing to overwrite an execution artifact."
        ) from None


def prepare_execution(
    run_dir: str | Path,
    *,
    phase: str,
    target: str,
    cell_timeout: int = 30,
    total_timeout: int = 120,
    now: Callable[[], datetime] = _utc_now,
) -> ExecutionPreparation:
    directory = Path(run_dir).expanduser().resolve(strict=False)
    if (
        isinstance(cell_timeout, bool)
        or not isinstance(cell_timeout, int)
        or cell_timeout < 1
        or isinstance(total_timeout, bool)
        or not isinstance(total_timeout, int)
        or total_timeout < 1
    ):
        raise ExecutionBlockedError(
            "invalid_execution_limits", "Execution time limits must be positive seconds."
        )
    store = RunStore(directory.parent)
    current = now().astimezone(timezone.utc)
    for metadata in store.recover_stale_attempts(directory, now=current):
        _cleanup_paths(metadata)

    binding = _target_binding(directory, phase, target)
    target_path = Path(binding["path"])
    target_sha256 = _sha256_file(target_path)
    if target_sha256 != binding["sha256"]:
        raise ExecutionBlockedError(
            "analysis_target_changed",
            "Execution target changed after static analysis.",
        )
    snapshot = build_snapshot(target_path)
    risk = scan_snapshot(snapshot)
    if risk["blocked"]:
        raise ExecutionBlockedError(
            "execution_risk_blocked", "Execution risk scan blocked this target."
        )
    if snapshot["source"]["sha256"] != target_sha256:
        raise ExecutionBlockedError(
            "analysis_target_changed", "Execution target changed during preparation."
        )

    request_id = uuid.uuid4().hex
    state_dir = directory / ".notebook-coach"
    temp_root = state_dir / "execution-temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f"{phase}-{target}-", dir=temp_root)
    ).resolve()
    request_path = (
        state_dir / "requests" / request_id / "execution-request.json"
    ).resolve()
    expires_at = current + timedelta(hours=1)
    metadata = {
        "phase": phase,
        "target": target,
        "request_id": request_id,
        "request_path": str(request_path),
        "temp_dir": str(temp_dir),
        "total_timeout": total_timeout,
        "expires_at": expires_at.isoformat(),
    }
    try:
        reservation = store.reserve_attempt(directory, metadata)
        request = {
            "schema_version": SCHEMA_VERSION,
            "run_id": binding["run_id"],
            "phase": phase,
            "target": target,
            "attempt_id": reservation.attempt_id,
            "request_id": request_id,
            "target_path": str(target_path),
            "target_sha256": target_sha256,
            "analysis_id": binding["analysis_id"],
            "kernel_name": binding["kernel_name"],
            "cell_timeout": cell_timeout,
            "total_timeout": total_timeout,
            "risk": risk,
            "created_at": current.isoformat(),
            "expires_at": expires_at.isoformat(),
            "temp_dir": str(temp_dir),
            "run_dir": str(directory),
        }
        _write_new(request_path, _json_bytes(request))
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if "reservation" in locals():
            try:
                store.transition_attempt(
                    directory,
                    reservation.attempt_id,
                    expected_status="prepared",
                    new_status="cancelled",
                    details={"reason": "preparation_failed"},
                )
            except RunStoreError:
                pass
        shutil.rmtree(request_path.parent, ignore_errors=True)
        raise
    return ExecutionPreparation(request_path=request_path, request=request)


def _read_request(path: str | Path) -> tuple[Path, dict[str, Any]]:
    request_path = Path(path).expanduser().resolve(strict=False)
    request = _read_object(
        request_path, code="request_invalid", label="Execution request"
    )
    required = {
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
    if set(request) != required or request.get("schema_version") != SCHEMA_VERSION:
        raise ExecutionBlockedError("request_invalid", "Execution request is invalid.")
    return request_path, request


def cancel_execution(request_path: str | Path) -> None:
    path, request = _read_request(request_path)
    run_dir = Path(request["run_dir"]).resolve(strict=False)
    store = RunStore(run_dir.parent)
    try:
        store.transition_attempt(
            run_dir,
            request["attempt_id"],
            expected_status="prepared",
            new_status="cancelled",
            details={"reason": "user_cancelled"},
        )
    except LedgerTransitionError:
        raise ExecutionBlockedError(
            "request_not_prepared", "Execution request is no longer pending."
        ) from None
    _cleanup_paths({"temp_dir": request["temp_dir"], "request_path": str(path)})


def _sanitize(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return "[OMITTED]"
    if isinstance(value, str):
        return summarize_text(value, max_chars=4_000)
    if isinstance(value, list):
        return [_sanitize(item, depth=depth + 1) for item in value[:200]]
    if isinstance(value, dict):
        return {
            str(key)[:100]: _sanitize(item, depth=depth + 1)
            for key, item in list(value.items())[:100]
        }
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return "[OMITTED]"


def _kill_process_group(process: subprocess.Popen) -> tuple[str, str]:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.communicate(timeout=0.5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.communicate()


def _log_path(request: dict[str, Any]) -> Path:
    return Path(request["run_dir"]) / (
        f"execution-{request['phase']}-{request['target']}-"
        f"{request['attempt_id']}.json"
    )


def _finish_log(
    request: dict[str, Any],
    *,
    status: str,
    target_before: str | None,
    target_copy: str | None,
    target_after: str | None,
    worker: Any = None,
    process_stdout: str = "",
    process_stderr: str = "",
) -> Path:
    eligible = status in _EVIDENCE_STATUSES
    log = {
        "schema_version": SCHEMA_VERSION,
        "run_id": request["run_id"],
        "phase": request["phase"],
        "target": request["target"],
        "attempt_id": request["attempt_id"],
        "request_id": request["request_id"],
        "analysis_id": request["analysis_id"],
        "target_path": request["target_path"],
        "target_sha256": request["target_sha256"],
        "target_sha256_before": target_before,
        "target_sha256_copy": target_copy,
        "target_sha256_after": target_after,
        "status": status,
        "evidence_eligible": eligible,
        "worker": _sanitize(worker),
        "process_stdout": summarize_text(process_stdout, max_chars=2_000),
        "process_stderr": summarize_text(process_stderr, max_chars=2_000),
    }
    path = _log_path(request)
    _write_new(path, _json_bytes(log))
    return path


def execute_request(
    request_path: str | Path,
    confirmed_target_sha256: str,
    *,
    now: Callable[[], datetime] = _utc_now,
) -> Path:
    path, request = _read_request(request_path)
    if (
        not isinstance(confirmed_target_sha256, str)
        or confirmed_target_sha256 != request["target_sha256"]
    ):
        raise ExecutionBlockedError(
            "confirmation_hash_mismatch",
            "Confirmed target hash does not match the prepared request.",
        )
    run_dir = Path(request["run_dir"]).resolve(strict=False)
    store = RunStore(run_dir.parent)
    current = now().astimezone(timezone.utc)
    try:
        expires_at = datetime.fromisoformat(request["expires_at"]).astimezone(
            timezone.utc
        )
    except (TypeError, ValueError):
        raise ExecutionBlockedError("request_invalid", "Execution request is invalid.")
    if current >= expires_at:
        try:
            store.transition_attempt(
                run_dir,
                request["attempt_id"],
                expected_status="prepared",
                new_status="expired",
                details={"reason": "request_expired"},
            )
        except LedgerTransitionError:
            raise ExecutionBlockedError(
                "request_not_prepared", "Execution request is no longer pending."
            ) from None
        _cleanup_paths({"temp_dir": request["temp_dir"], "request_path": str(path)})
        raise ExecutionBlockedError("request_expired", "Execution request expired.")

    try:
        store.transition_attempt(
            run_dir,
            request["attempt_id"],
            expected_status="prepared",
            new_status="running",
        )
    except LedgerTransitionError:
        raise ExecutionBlockedError(
            "request_not_prepared", "Execution request is no longer pending."
        ) from None

    target_before: str | None = None
    target_copy: str | None = None
    target_after: str | None = None
    process_stdout = ""
    process_stderr = ""
    worker: Any = None
    status = "worker_failure"
    pending_error: ExecutionBlockedError | None = None
    try:
        binding = _target_binding(run_dir, request["phase"], request["target"])
        if any(
            (
                binding["run_id"] != request["run_id"],
                binding["path"] != request["target_path"],
                binding["sha256"] != request["target_sha256"],
                binding["kernel_name"] != request["kernel_name"],
                binding["analysis_id"] != request["analysis_id"],
            )
        ):
            raise ExecutionBlockedError(
                "analysis_target_changed", "Static analysis binding changed."
            )
        target_path = Path(request["target_path"])
        target_before = _sha256_file(target_path)
        if target_before != request["target_sha256"]:
            raise ExecutionBlockedError(
                "target_changed_before_execution",
                "Target changed before execution.",
            )
        risk = scan_snapshot(build_snapshot(target_path))
        if risk != request["risk"] or risk["blocked"]:
            raise ExecutionBlockedError(
                "execution_risk_blocked", "Execution risk result changed or is blocked."
            )
        temp_dir = Path(request["temp_dir"])
        if not temp_dir.is_dir() or list(temp_dir.iterdir()):
            raise ExecutionBlockedError(
                "reserved_temp_invalid", "Reserved temporary directory is invalid."
            )
        copied_target = temp_dir / "target.ipynb"
        shutil.copyfile(target_path, copied_target)
        target_copy = _sha256_file(copied_target)
        if target_copy != request["target_sha256"]:
            raise ExecutionBlockedError(
                "target_copy_mismatch", "Temporary target copy hash does not match."
            )
        worker_output = temp_dir / "worker-result.json"
        command = [
            sys.executable,
            "-m",
            "notebook_coach.execution_worker",
            "--input",
            str(copied_target),
            "--output",
            str(worker_output),
            "--kernel",
            request["kernel_name"],
            "--cell-timeout",
            str(request["cell_timeout"]),
        ]
        process = subprocess.Popen(
            command,
            cwd=temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            process_stdout, process_stderr = process.communicate(
                timeout=request["total_timeout"]
            )
        except subprocess.TimeoutExpired:
            process_stdout, process_stderr = _kill_process_group(process)
            status = "timeout"
        else:
            if process.returncode == 0 and worker_output.is_file():
                worker = _read_object(
                    worker_output,
                    code="worker_failure",
                    label="Execution worker result",
                )
                status = (
                    "completed_with_cell_errors"
                    if worker.get("has_cell_errors")
                    else "completed"
                )
            else:
                status = "worker_failure"
                pending_error = ExecutionBlockedError(
                    "worker_failure", "Execution worker failed."
                )
        target_after = _sha256_file(target_path)
        if target_after != request["target_sha256"]:
            status = "target_changed_during_run"
            worker = None
            pending_error = ExecutionBlockedError(
                "target_changed_during_run", "Target changed during execution."
            )
    except ExecutionBlockedError as error:
        status = error.code
        pending_error = error
        try:
            target_after = _sha256_file(Path(request["target_path"]))
        except ExecutionBlockedError:
            target_after = None
    except Exception:
        status = "worker_failure"
        pending_error = ExecutionBlockedError(
            "worker_failure", "Execution worker failed."
        )

    try:
        log_path = _finish_log(
            request,
            status=status,
            target_before=target_before,
            target_copy=target_copy,
            target_after=target_after,
            worker=worker,
            process_stdout=process_stdout,
            process_stderr=process_stderr,
        )
        store.transition_attempt(
            run_dir,
            request["attempt_id"],
            expected_status="running",
            new_status="completed" if status in _EVIDENCE_STATUSES else "failed",
            details={"status": status, "log": log_path.name},
        )
    except Exception:
        try:
            store.transition_attempt(
                run_dir,
                request["attempt_id"],
                expected_status="running",
                new_status="failed",
                details={"reason": "log_failure"},
            )
        except RunStoreError:
            pass
        raise
    finally:
        _cleanup_paths({"temp_dir": request["temp_dir"], "request_path": str(path)})
    if pending_error is not None:
        raise pending_error
    return log_path
