"""Strict JSON contracts for model-produced notebook assessments."""

from __future__ import annotations

import copy
import re
from typing import Any

from notebook_coach import SCHEMA_VERSION
from notebook_coach.sanitize import redact_text


ISSUE_ID_PATTERN = re.compile(r"I[0-9]{3}")
CHALLENGE_IDS = ("C-CODE", "C-CONCEPT")
DIMENSIONS = (
    "correctness",
    "concept_completeness",
    "reproducibility",
    "clarity",
)
SEVERITIES = ("blocking", "major", "minor")

_DIAGNOSIS_KEYS = {
    "schema_version",
    "run_id",
    "notebook_summary",
    "concept_map",
    "issues",
    "challenges",
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
_CHALLENGE_KEYS = {
    "challenge_id",
    "kind",
    "source_issue_ids",
    "title",
    "prompt",
    "acceptance_criteria",
}
_VERIFICATION_KEYS = {
    "schema_version",
    "run_id",
    "issue_results",
    "new_issues",
    "challenge_results",
    "next_learning_target",
}


class ContractError(ValueError):
    """A stable contract error that never echoes unsafe model text."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _fail(code: str, message: str) -> None:
    raise ContractError(code, message)


def _mapping(value: Any, *, code: str, message: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(code, message)
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], *, code: str) -> None:
    if set(value) != expected:
        _fail(code, "Assessment fields do not match the required contract.")


def _text(value: Any, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(code, "Assessment text must be a non-empty string.")
    if len(value) > 20_000:
        _fail(code, "Assessment text exceeds the supported size.")
    cleaned, labels = redact_text(value)
    if labels or cleaned != value:
        _fail("unredacted_text", "Assessment contains unredacted sensitive text.")
    return value


def _string_list(value: Any, *, code: str, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        _fail(code, "Assessment requires a list of text values.")
    for item in value:
        _text(item, code=code)
    return value


def _cell_indices(
    value: Any,
    *,
    cell_count: int,
    allow_empty: bool = False,
    allowed_cell_indices: set[int] | None = None,
) -> list[int]:
    if not isinstance(value, list) or (not value and not allow_empty):
        _fail("unknown_cell_index", "Issue cell indices are invalid.")
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= cell_count
        or (allowed_cell_indices is not None and index not in allowed_cell_indices)
        for index in value
    ):
        _fail("unknown_cell_index", "Issue references an unknown cell index.")
    if value != sorted(set(value)):
        _fail("unknown_cell_index", "Issue cell indices must be sorted and unique.")
    return value


def _validate_issue(
    value: Any,
    *,
    cell_count: int,
    allowed_cell_indices: set[int] | None = None,
) -> dict[str, Any]:
    issue = _mapping(value, code="issue_contract", message="Issue must be an object.")
    _exact_keys(issue, _ISSUE_KEYS, code="issue_contract")
    issue_id = issue.get("issue_id")
    if not isinstance(issue_id, str) or ISSUE_ID_PATTERN.fullmatch(issue_id) is None:
        _fail("issue_id", "Issue ID must use the I000 format.")
    if issue.get("dimension") not in DIMENSIONS:
        _fail("unsupported_dimension", "Issue dimension is not supported.")
    if issue.get("severity") not in SEVERITIES:
        _fail("unsupported_severity", "Issue severity is not supported.")
    _text(issue.get("category"), code="issue_contract")
    _cell_indices(
        issue.get("cell_indices"),
        cell_count=cell_count,
        allowed_cell_indices=allowed_cell_indices,
    )
    for key in ("evidence", "impact", "recommendation"):
        _text(issue.get(key), code="issue_contract")
    return issue


def validate_diagnosis(
    value: Any,
    *,
    run_id: str,
    cell_count: int,
    allowed_cell_indices: set[int] | None = None,
) -> dict[str, Any]:
    """Validate diagnosis JSON without repairing or reordering it."""

    diagnosis = _mapping(
        value, code="diagnosis_contract", message="Diagnosis must be an object."
    )
    _exact_keys(diagnosis, _DIAGNOSIS_KEYS, code="diagnosis_contract")
    if diagnosis.get("schema_version") != SCHEMA_VERSION:
        _fail("schema_version", "Diagnosis schema version is unsupported.")
    if diagnosis.get("run_id") == "__RUN_ID__" or diagnosis.get("run_id") != run_id:
        _fail("run_id_mismatch", "Diagnosis run ID does not match the stage.")
    if isinstance(cell_count, bool) or not isinstance(cell_count, int) or cell_count < 0:
        _fail("cell_count", "Snapshot cell count is invalid.")
    _text(diagnosis.get("notebook_summary"), code="diagnosis_contract")
    _string_list(diagnosis.get("concept_map"), code="diagnosis_contract")

    issues = diagnosis.get("issues")
    if not isinstance(issues, list):
        _fail("issue_contract", "Diagnosis issues must be a list.")
    issue_ids: list[str] = []
    for issue in issues:
        validated = _validate_issue(
            issue,
            cell_count=cell_count,
            allowed_cell_indices=allowed_cell_indices,
        )
        issue_ids.append(validated["issue_id"])
    if len(issue_ids) != len(set(issue_ids)):
        _fail("duplicate_issue_id", "Diagnosis issue IDs must be unique.")

    challenges = diagnosis.get("challenges")
    if not isinstance(challenges, list) or len(challenges) != 2:
        _fail("challenge_contract", "Diagnosis requires exactly two challenges.")
    if [item.get("challenge_id") for item in challenges if isinstance(item, dict)] != list(
        CHALLENGE_IDS
    ):
        _fail("challenge_contract", "Diagnosis challenge IDs or order are invalid.")
    expected_kinds = {"C-CODE": "code", "C-CONCEPT": "concept"}
    known_issues = set(issue_ids)
    for challenge in challenges:
        challenge = _mapping(
            challenge,
            code="challenge_contract",
            message="Challenge must be an object.",
        )
        _exact_keys(challenge, _CHALLENGE_KEYS, code="challenge_contract")
        if challenge.get("kind") != expected_kinds[challenge["challenge_id"]]:
            _fail("challenge_contract", "Challenge kind does not match its ID.")
        references = challenge.get("source_issue_ids")
        if not isinstance(references, list) or not references:
            _fail("challenge_contract", "Challenge must reference an issue.")
        if references != list(dict.fromkeys(references)) or any(
            reference not in known_issues for reference in references
        ):
            _fail(
                "unknown_issue_reference",
                "Challenge references an unknown or duplicate issue ID.",
            )
        for key in ("title", "prompt"):
            _text(challenge.get(key), code="challenge_contract")
        _string_list(
            challenge.get("acceptance_criteria"), code="challenge_contract"
        )
    return copy.deepcopy(diagnosis)


def validate_verification(
    value: Any,
    *,
    run_id: str,
    baseline_issue_ids: list[str],
    cell_count: int = 1_000_000,
    challenge_verifiable: bool = True,
    allowed_cell_indices: set[int] | None = None,
) -> dict[str, Any]:
    verification = _mapping(
        value,
        code="verification_contract",
        message="Verification must be an object.",
    )
    _exact_keys(verification, _VERIFICATION_KEYS, code="verification_contract")
    if verification.get("schema_version") != SCHEMA_VERSION:
        _fail("schema_version", "Verification schema version is unsupported.")
    if verification.get("run_id") == "__RUN_ID__" or verification.get("run_id") != run_id:
        _fail("run_id_mismatch", "Verification run ID does not match the run.")

    results = verification.get("issue_results")
    if not isinstance(results, list):
        _fail("issue_result_ids", "Verification issue results must be a list.")
    result_ids: list[str] = []
    for result in results:
        result = _mapping(
            result,
            code="verification_contract",
            message="Issue result must be an object.",
        )
        if set(result) != {"issue_id", "status", "current_cell_indices", "evidence"}:
            _fail("verification_contract", "Issue result fields are invalid.")
        if result.get("status") not in {"resolved", "remaining", "regressed"}:
            _fail("verification_contract", "Issue result status is unsupported.")
        issue_id = result.get("issue_id")
        if not isinstance(issue_id, str):
            _fail("issue_result_ids", "Issue result ID is invalid.")
        result_ids.append(issue_id)
        _cell_indices(
            result.get("current_cell_indices"),
            cell_count=cell_count,
            allow_empty=True,
            allowed_cell_indices=allowed_cell_indices,
        )
        _text(result.get("evidence"), code="verification_contract")
    if sorted(result_ids) != sorted(baseline_issue_ids) or len(result_ids) != len(
        set(result_ids)
    ):
        _fail("issue_result_ids", "Verification must cover each baseline issue once.")

    new_issues = verification.get("new_issues")
    if not isinstance(new_issues, list):
        _fail("verification_contract", "New issues must be a list.")
    new_issue_ids: list[str] = []
    for issue in new_issues:
        validated = _validate_issue(
            issue,
            cell_count=cell_count,
            allowed_cell_indices=allowed_cell_indices,
        )
        new_issue_ids.append(validated["issue_id"])
    if set(new_issue_ids).intersection(baseline_issue_ids):
        _fail("new_issue_ids", "New issue IDs collide with baseline issues.")
    baseline_numbers = [
        int(issue_id[1:])
        for issue_id in baseline_issue_ids
        if ISSUE_ID_PATTERN.fullmatch(issue_id)
    ]
    start = max(baseline_numbers, default=0) + 1
    expected_new_ids = [f"I{number:03d}" for number in range(start, start + len(new_issues))]
    if new_issue_ids != expected_new_ids:
        _fail("new_issue_ids", "New issue IDs must be sequential and unique.")

    challenge_results = verification.get("challenge_results")
    if not isinstance(challenge_results, list) or [
        item.get("challenge_id") for item in challenge_results if isinstance(item, dict)
    ] != list(CHALLENGE_IDS):
        _fail("challenge_result_ids", "Verification challenge results are invalid.")
    for result in challenge_results:
        if set(result) != {"challenge_id", "status", "evidence"}:
            _fail("verification_contract", "Challenge result fields are invalid.")
        if result.get("status") not in {"passed", "needs_work", "unverifiable"}:
            _fail("verification_contract", "Challenge result status is unsupported.")
        _text(result.get("evidence"), code="verification_contract")
    challenge_statuses = [result["status"] for result in challenge_results]
    if challenge_verifiable and "unverifiable" in challenge_statuses:
        _fail(
            "challenge_verifiability",
            "Unverifiable challenge status requires a metadata mismatch.",
        )
    if not challenge_verifiable and challenge_statuses != [
        "unverifiable",
        "unverifiable",
    ]:
        _fail(
            "challenge_verifiability",
            "Both challenge results must remain unverifiable in this cycle.",
        )
    _text(
        verification.get("next_learning_target"), code="verification_contract"
    )
    return copy.deepcopy(verification)
