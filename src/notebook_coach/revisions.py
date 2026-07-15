"""Apply target-scoped execution evidence as deterministic report revisions."""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from notebook_coach import SCHEMA_VERSION
from notebook_coach.contracts import (
    DIMENSIONS,
    ISSUE_ID_PATTERN,
    SEVERITIES,
    validate_verification,
)
from notebook_coach.render import render_report, render_verification
from notebook_coach.sanitize import redact_text
from notebook_coach.scoring import score_verification


_LOG_NAME = re.compile(
    r"execution-(diagnosis|verification)-(source|challenge)-(A[0-9]{3,})\.json"
)
_EVIDENCE_LABELS = {"supported", "conflicted", "uncertain"}
_ELIGIBLE_STATUSES = {"completed", "completed_with_cell_errors", "timeout"}
_REVIEW_KEYS = {
    "schema_version",
    "run_id",
    "phase",
    "target",
    "attempt_id",
    "execution_log",
    "issue_updates",
    "new_issues",
    "challenge_updates",
}
_ISSUE_KEYS = {
    "issue_id",
    "dimension",
    "severity",
    "category",
    "cell_indices",
    "evidence",
    "impact",
    "recommendation",
}


class RevisionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class RevisionResult:
    revision: int
    state_path: Path
    report_path: Path


def _read_object(path: Path, *, code: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise RevisionError(code, f"{label} is missing or invalid.") from None
    if not isinstance(value, dict):
        raise RevisionError(code, f"{label} is missing or invalid.")
    return value


def _safe_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 20_000:
        raise RevisionError("review_contract", "Review text is invalid.")
    cleaned, labels = redact_text(value)
    if labels or cleaned != value:
        raise RevisionError(
            "unredacted_text", "Execution review contains sensitive text."
        )
    return value


def _evidence_fields(value: Any, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise RevisionError("review_contract", "Review update fields are invalid.")
    if value.get("evidence_label") not in _EVIDENCE_LABELS:
        raise RevisionError("review_contract", "Evidence label is invalid.")
    _safe_text(value.get("evidence"))
    return value


def _validate_new_issue(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _ISSUE_KEYS:
        raise RevisionError("review_contract", "New issue fields are invalid.")
    issue_id = value.get("issue_id")
    if not isinstance(issue_id, str) or ISSUE_ID_PATTERN.fullmatch(issue_id) is None:
        raise RevisionError("review_contract", "New issue ID is invalid.")
    if value.get("dimension") not in DIMENSIONS or value.get("severity") not in SEVERITIES:
        raise RevisionError("review_contract", "New issue rubric fields are invalid.")
    indices = value.get("cell_indices")
    if (
        not isinstance(indices, list)
        or not indices
        or indices != sorted(set(indices))
        or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices)
    ):
        raise RevisionError("review_contract", "New issue cell indices are invalid.")
    for key in ("category", "evidence", "impact", "recommendation"):
        _safe_text(value.get(key))
    return value


def _validate_review(
    value: dict[str, Any], *, phase: str, target: str, baseline_issue_ids: set[str]
) -> dict[str, Any]:
    if set(value) != _REVIEW_KEYS or value.get("schema_version") != SCHEMA_VERSION:
        raise RevisionError("review_contract", "Execution review contract is invalid.")
    for key in ("issue_updates", "new_issues", "challenge_updates"):
        if not isinstance(value.get(key), list):
            raise RevisionError("review_contract", "Review update lists are invalid.")

    if phase == "diagnosis":
        if target != "source" or value["new_issues"] or value["challenge_updates"]:
            raise RevisionError("target_boundary", "Diagnosis review crossed its target boundary.")
        seen: set[str] = set()
        for update in value["issue_updates"]:
            update = _evidence_fields(
                update, {"issue_id", "evidence_label", "evidence"}
            )
            if update.get("issue_id") not in baseline_issue_ids or update["issue_id"] in seen:
                raise RevisionError("unknown_issue", "Diagnosis review issue ID is invalid.")
            seen.add(update["issue_id"])
    elif target == "source":
        if value["challenge_updates"]:
            raise RevisionError("target_boundary", "Source review contains challenge updates.")
        seen = set()
        for update in value["issue_updates"]:
            update = _evidence_fields(
                update,
                {"issue_id", "status", "evidence_label", "evidence"},
            )
            if update.get("status") not in {"resolved", "remaining", "regressed"}:
                raise RevisionError("review_contract", "Issue update status is invalid.")
            if update.get("issue_id") not in baseline_issue_ids or update["issue_id"] in seen:
                raise RevisionError("unknown_issue", "Source review issue ID is invalid.")
            seen.add(update["issue_id"])
        for issue in value["new_issues"]:
            _validate_new_issue(issue)
    elif target == "challenge":
        if value["issue_updates"] or value["new_issues"] or len(value["challenge_updates"]) != 1:
            raise RevisionError("target_boundary", "Challenge review crossed its target boundary.")
        update = _evidence_fields(
            value["challenge_updates"][0],
            {"challenge_id", "status", "evidence_label", "evidence"},
        )
        if update.get("challenge_id") != "C-CODE" or update.get("status") not in {
            "passed",
            "needs_work",
        }:
            raise RevisionError("target_boundary", "Challenge review target is invalid.")
    else:
        raise RevisionError("target_boundary", "Execution review target is invalid.")
    return copy.deepcopy(value)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _replace_pair(
    state_path: Path, state_body: bytes, report_path: Path, report_body: bytes
) -> None:
    old_state = state_path.read_bytes()
    old_report = report_path.read_bytes()
    temporary_paths: list[Path] = []
    try:
        for destination, body in (
            (state_path, state_body),
            (report_path, report_body),
        ):
            descriptor, name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            temporary = Path(name)
            temporary_paths.append(temporary)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary_paths[0], state_path)
        os.replace(temporary_paths[1], report_path)
    except Exception:
        state_path.write_bytes(old_state)
        report_path.write_bytes(old_report)
        raise
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


def _validate_log(
    log: dict[str, Any], *, expected: dict[str, Any], log_name: str
) -> None:
    if (
        log.get("schema_version") != SCHEMA_VERSION
        or log.get("execution_eligible", log.get("evidence_eligible")) is not True
        or log.get("status") not in _ELIGIBLE_STATUSES
    ):
        raise RevisionError("execution_evidence_ineligible", "Execution log is not eligible.")
    for key in ("run_id", "phase", "target", "attempt_id", "analysis_id", "target_path"):
        if log.get(key) != expected[key]:
            raise RevisionError("execution_binding_mismatch", "Execution log binding is invalid.")
    target_hash = expected["target_sha256"]
    if any(
        log.get(key) != target_hash
        for key in (
            "target_sha256",
            "target_sha256_before",
            "target_sha256_copy",
            "target_sha256_after",
        )
    ):
        raise RevisionError("execution_hash_mismatch", "Execution target hashes do not match.")
    if log_name != expected["execution_log"]:
        raise RevisionError("execution_binding_mismatch", "Execution log name is invalid.")


def apply_execution_review(
    run_dir: str | Path, log_path: str | Path, review_path: str | Path
) -> RevisionResult:
    directory = Path(run_dir).expanduser().resolve(strict=False)
    log_file = Path(log_path).expanduser().resolve(strict=False)
    match = _LOG_NAME.fullmatch(log_file.name)
    if match is None or log_file.parent != directory:
        raise RevisionError("execution_log_path", "Execution log path is invalid.")
    phase, target, attempt_id = match.groups()

    baseline = _read_object(directory / "baseline.json", code="baseline_invalid", label="Baseline")
    baseline_issue_ids = {item["issue_id"] for item in baseline.get("issues", [])}
    if phase == "verification":
        state_path = directory / ".notebook-coach/verification-state.json"
        state = _read_object(state_path, code="verification_state_invalid", label="Verification state")
        if target == "challenge" and not state.get("challenge_verifiability", {}).get("verifiable"):
            raise RevisionError("challenge_unverifiable", "Challenge is unverifiable in this cycle.")
        report_path = directory / "verification.md"
        target_state = state.get(f"{target}_target", {})
        analysis_id = state.get("assessment_id")
    else:
        state_path = directory / ".notebook-coach/report-state.json"
        state = _read_object(state_path, code="report_state_invalid", label="Report state")
        report_path = directory / "report.md"
        target_state = baseline.get("source", {})
        analysis_id = baseline.get("analysis_id")

    previous_reviews = state.get("execution_reviews")
    if not isinstance(previous_reviews, list):
        raise RevisionError("revision_state_invalid", "Execution review state is invalid.")
    if any(item.get("execution_log") == log_file.name for item in previous_reviews if isinstance(item, dict)):
        raise RevisionError("execution_log_replayed", "Execution log was already applied.")

    log = _read_object(log_file, code="execution_log_invalid", label="Execution log")
    review = _read_object(Path(review_path), code="review_invalid", label="Execution review")
    expected = {
        "run_id": baseline["run_id"],
        "phase": phase,
        "target": target,
        "attempt_id": attempt_id,
        "analysis_id": analysis_id,
        "target_path": target_state.get("path"),
        "target_sha256": target_state.get("sha256"),
        "execution_log": log_file.name,
    }
    _validate_log(log, expected=expected, log_name=log_file.name)
    for key in ("run_id", "phase", "target", "attempt_id", "execution_log"):
        if review.get(key) != expected[key]:
            raise RevisionError("review_binding_mismatch", "Execution review binding is invalid.")
    review = _validate_review(
        review, phase=phase, target=target, baseline_issue_ids=baseline_issue_ids
    )

    updated = copy.deepcopy(state)
    updated["revision"] = int(updated.get("revision", 0)) + 1
    updated["execution_reviews"].append(review)
    if phase == "diagnosis":
        report = render_report(baseline, updated)
    else:
        assessment = updated.get("assessment")
        if not isinstance(assessment, dict):
            raise RevisionError("verification_state_invalid", "Verification assessment is invalid.")
        if target == "source":
            results = {item["issue_id"]: item for item in assessment["issue_results"]}
            for change in review["issue_updates"]:
                results[change["issue_id"]]["status"] = change["status"]
                results[change["issue_id"]]["evidence"] = change["evidence"]
            assessment["new_issues"].extend(copy.deepcopy(review["new_issues"]))
            validate_verification(
                assessment,
                run_id=updated["run_id"],
                baseline_issue_ids=sorted(baseline_issue_ids),
                challenge_verifiable=bool(
                    updated["challenge_verifiability"]["verifiable"]
                ),
            )
            updated["after_score"] = score_verification(
                baseline["issues"],
                assessment["issue_results"],
                assessment["new_issues"],
            )
        else:
            change = review["challenge_updates"][0]
            result = next(
                item
                for item in assessment["challenge_results"]
                if item["challenge_id"] == "C-CODE"
            )
            result["status"] = change["status"]
            result["evidence"] = change["evidence"]
        report = render_verification(updated)

    _replace_pair(
        state_path,
        _json_bytes(updated),
        report_path,
        report.encode("utf-8"),
    )
    return RevisionResult(
        revision=updated["revision"],
        state_path=state_path,
        report_path=report_path,
    )
