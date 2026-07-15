from __future__ import annotations

import copy

import pytest

from notebook_coach.contracts import (
    ContractError,
    validate_diagnosis,
    validate_verification,
)


def test_diagnosis_requires_exactly_one_code_and_one_concept_challenge(
    valid_diagnosis,
):
    validated = validate_diagnosis(
        valid_diagnosis,
        run_id=valid_diagnosis["run_id"],
        cell_count=2,
    )

    assert [item["challenge_id"] for item in validated["challenges"]] == [
        "C-CODE",
        "C-CONCEPT",
    ]
    assert validated == valid_diagnosis
    assert validated is not valid_diagnosis


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("duplicate_issue", "duplicate_issue_id"),
        ("unknown_cell", "unknown_cell_index"),
        ("missing_challenge", "challenge_contract"),
        ("unknown_issue_reference", "unknown_issue_reference"),
        ("bad_severity", "unsupported_severity"),
        ("bad_dimension", "unsupported_dimension"),
        ("wrong_run", "run_id_mismatch"),
    ],
)
def test_diagnosis_rejects_invalid_contracts(valid_diagnosis, mutation, code):
    value = copy.deepcopy(valid_diagnosis)
    if mutation == "duplicate_issue":
        value["issues"].append(copy.deepcopy(value["issues"][0]))
    elif mutation == "unknown_cell":
        value["issues"][0]["cell_indices"] = [99]
    elif mutation == "missing_challenge":
        value["challenges"].pop()
    elif mutation == "unknown_issue_reference":
        value["challenges"][0]["source_issue_ids"] = ["I999"]
    elif mutation == "bad_severity":
        value["issues"][0]["severity"] = "critical"
    elif mutation == "bad_dimension":
        value["issues"][0]["dimension"] = "style"
    else:
        value["run_id"] = "another-run"

    with pytest.raises(ContractError) as error:
        validate_diagnosis(
            value,
            run_id=valid_diagnosis["run_id"],
            cell_count=2,
        )

    assert error.value.code == code


def test_diagnosis_rejects_unredacted_secret(valid_diagnosis):
    value = copy.deepcopy(valid_diagnosis)
    value["notebook_summary"] = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"

    with pytest.raises(ContractError) as error:
        validate_diagnosis(value, run_id=value["run_id"], cell_count=2)

    assert error.value.code == "unredacted_text"
    assert "sk-proj" not in str(error.value)


def test_verification_requires_all_baseline_issues_once(valid_verification):
    validated = validate_verification(
        valid_verification,
        run_id=valid_verification["run_id"],
        baseline_issue_ids=["I001", "I002"],
    )

    assert [item["issue_id"] for item in validated["issue_results"]] == [
        "I001",
        "I002",
    ]

    missing = copy.deepcopy(valid_verification)
    missing["issue_results"].pop()
    with pytest.raises(ContractError) as error:
        validate_verification(
            missing,
            run_id=missing["run_id"],
            baseline_issue_ids=["I001", "I002"],
        )
    assert error.value.code == "issue_result_ids"
