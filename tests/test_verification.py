from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pytest

from notebook_coach.contracts import ContractError, validate_verification
from notebook_coach.execution import prepare_execution
from notebook_coach.workflows import (
    ConfirmationRequiredError,
    finalize_diagnosis,
    finalize_verification,
    prepare_diagnosis,
    prepare_verification,
)


def _run(source: Path, root: Path, diagnosis: dict) -> Path:
    prepared = prepare_diagnosis(source, root)
    value = json.loads(json.dumps(diagnosis))
    value["run_id"] = prepared.stage.run_id
    prepared.assessment_path.write_text(json.dumps(value), encoding="utf-8")
    return finalize_diagnosis(prepared.stage.stage_dir)


def _verification(value: dict, run_id: str, *, issue_statuses, challenge_statuses):
    result = json.loads(json.dumps(value))
    result["run_id"] = run_id
    for item, status in zip(result["issue_results"], issue_statuses, strict=True):
        item["status"] = status
    for item, status in zip(
        result["challenge_results"], challenge_statuses, strict=True
    ):
        item["status"] = status
    return result


def test_challenge_completion_does_not_inflate_unchanged_source_score(
    notebook_path: Path,
    tmp_path: Path,
    valid_diagnosis: dict,
    valid_verification: dict,
):
    run_dir = _run(notebook_path, tmp_path / "runs", valid_diagnosis)
    stage = prepare_verification(notebook_path, run_dir)
    assessment = _verification(
        valid_verification,
        stage.run_id,
        issue_statuses=("remaining", "remaining"),
        challenge_statuses=("passed", "passed"),
    )

    verification = finalize_verification(stage.stage_dir, assessment)

    assert verification.before_score == verification.after_score
    assert verification.challenge_results["C-CODE"] == "passed"
    assert "Source notebook changes" in verification.report_path.read_text("utf-8")
    assert "Challenge results" in verification.report_path.read_text("utf-8")


def test_source_score_can_improve_while_challenges_still_need_work(
    notebook_path: Path,
    tmp_path: Path,
    valid_diagnosis: dict,
    valid_verification: dict,
):
    run_dir = _run(notebook_path, tmp_path / "runs", valid_diagnosis)
    stage = prepare_verification(notebook_path, run_dir)
    assessment = _verification(
        valid_verification,
        stage.run_id,
        issue_statuses=("resolved", "resolved"),
        challenge_statuses=("needs_work", "needs_work"),
    )

    verification = finalize_verification(stage.stage_dir, assessment)

    assert verification.after_score["total"] > verification.before_score["total"]
    assert set(verification.challenge_results.values()) == {"needs_work"}


def test_environment_mismatch_requires_separate_confirmation(
    notebook_path: Path, tmp_path: Path, valid_diagnosis: dict
):
    run_dir = _run(notebook_path, tmp_path / "runs", valid_diagnosis)
    notebook = nbformat.read(notebook_path, as_version=4)
    notebook.metadata.kernelspec.name = "different-kernel"
    notebook.metadata.language_info.name = "julia"
    nbformat.write(notebook, notebook_path)

    with pytest.raises(ConfirmationRequiredError) as error:
        prepare_verification(notebook_path, run_dir)

    assert error.value.code == "environment_confirmation_required"
    assert error.value.details["baseline_kernel"] == "python3"
    assert error.value.details["current_kernel"] == "different-kernel"
    stage = prepare_verification(
        notebook_path, run_dir, confirm_environment_mismatch=True
    )
    assert stage.stage_dir.is_dir()


def test_confirmed_different_source_path_is_persisted_and_used_for_execution(
    notebook_factory,
    tmp_path: Path,
    valid_diagnosis: dict,
    valid_verification: dict,
):
    original = notebook_factory(
        cells=[
            nbformat.v4.new_code_cell("scores = [[1.0, 2.0]]"),
            nbformat.v4.new_code_cell("weights = [0.2, 0.8]"),
        ]
    )
    run_dir = _run(original, tmp_path / "runs", valid_diagnosis)
    selected = notebook_factory(
        cells=[
            nbformat.v4.new_code_cell("scores = [[1.0, 2.0]]"),
            nbformat.v4.new_code_cell("weights = [0.3, 0.7]"),
        ]
    )

    with pytest.raises(ConfirmationRequiredError) as error:
        prepare_verification(selected, run_dir)
    assert error.value.code == "source_confirmation_required"

    stage = prepare_verification(
        selected, run_dir, confirm_source_mismatch=True
    )
    assessment = _verification(
        valid_verification,
        stage.run_id,
        issue_statuses=("resolved", "remaining"),
        challenge_statuses=("needs_work", "needs_work"),
    )
    finalize_verification(stage.stage_dir, assessment)
    state = json.loads(
        (run_dir / ".notebook-coach/verification-state.json").read_text("utf-8")
    )
    request = prepare_execution(run_dir, phase="verification", target="source")

    assert state["source_target"]["path"] == str(selected.resolve())
    assert request.request["target_path"] == str(selected.resolve())
    assert request.request["target_path"] != str(original.resolve())


def test_challenge_metadata_mismatch_forces_unverifiable_statuses(
    notebook_path: Path,
    tmp_path: Path,
    valid_diagnosis: dict,
    valid_verification: dict,
):
    run_dir = _run(notebook_path, tmp_path / "runs", valid_diagnosis)
    challenge_path = run_dir / "challenge.ipynb"
    challenge = nbformat.read(challenge_path, as_version=4)
    challenge.metadata.notebook_coach.run_id = "wrong-run"
    nbformat.write(challenge, challenge_path)
    stage = prepare_verification(notebook_path, run_dir)
    invalid = _verification(
        valid_verification,
        stage.run_id,
        issue_statuses=("remaining", "remaining"),
        challenge_statuses=("passed", "needs_work"),
    )

    assert stage.challenge_verifiability["verifiable"] is False
    with pytest.raises(ContractError):
        finalize_verification(stage.stage_dir, invalid)

    valid = _verification(
        valid_verification,
        stage.run_id,
        issue_statuses=("remaining", "remaining"),
        challenge_statuses=("unverifiable", "unverifiable"),
    )
    result = finalize_verification(stage.stage_dir, valid)
    assert set(result.challenge_results.values()) == {"unverifiable"}


def test_challenge_snapshot_and_outputs_are_redacted_and_bounded(
    notebook_path: Path,
    tmp_path: Path,
    valid_diagnosis: dict,
    valid_verification: dict,
):
    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
    run_dir = _run(notebook_path, tmp_path / "runs", valid_diagnosis)
    challenge_path = run_dir / "challenge.ipynb"
    challenge = nbformat.read(challenge_path, as_version=4)
    challenge.cells[2].source += f"\n\npassword = '{secret}'" + "x" * 20_000
    nbformat.write(challenge, challenge_path)

    stage = prepare_verification(notebook_path, run_dir)
    snapshot_body = stage.challenge_snapshot_path.read_text("utf-8")
    assessment = _verification(
        valid_verification,
        stage.run_id,
        issue_statuses=("remaining", "remaining"),
        challenge_statuses=("needs_work", "needs_work"),
    )
    assessment["challenge_results"][1]["evidence"] = "The answer remains incomplete."
    result = finalize_verification(stage.stage_dir, assessment)
    persisted = (
        (run_dir / ".notebook-coach/verification-state.json").read_text("utf-8")
        + result.report_path.read_text("utf-8")
    )

    assert secret not in snapshot_body
    assert "[REDACTED]" in snapshot_body
    assert len(snapshot_body) < 20_000
    assert secret not in persisted


def test_verification_contract_rejects_colliding_new_ids_and_unjustified_unverifiable(
    valid_verification: dict,
):
    value = json.loads(json.dumps(valid_verification))
    value["new_issues"] = [
        {
            "issue_id": "I002",
            "dimension": "clarity",
            "severity": "minor",
            "category": "explanation",
            "cell_indices": [1],
            "evidence": "Cell 1 remains unclear.",
            "impact": "The reasoning is hard to follow.",
            "recommendation": "Explain the axis.",
        }
    ]
    with pytest.raises(ContractError):
        validate_verification(
            value,
            run_id=value["run_id"],
            baseline_issue_ids=["I001", "I002"],
            cell_count=2,
            challenge_verifiable=True,
        )

    value["new_issues"] = []
    value["challenge_results"][0]["status"] = "unverifiable"
    with pytest.raises(ContractError):
        validate_verification(
            value,
            run_id=value["run_id"],
            baseline_issue_ids=["I001", "I002"],
            cell_count=2,
            challenge_verifiable=True,
        )


def test_verification_cli_stages_and_finalizes_machine_readable_json(
    cli_runner,
    notebook_path: Path,
    tmp_path: Path,
    valid_diagnosis: dict,
    valid_verification: dict,
):
    run_dir = _run(notebook_path, tmp_path / "runs", valid_diagnosis)
    prepared_result = cli_runner(
        "prepare-verification", str(notebook_path), "--run", str(run_dir)
    )
    prepared = json.loads(prepared_result.stdout)
    assessment = _verification(
        valid_verification,
        prepared["run_id"],
        issue_statuses=("resolved", "remaining"),
        challenge_statuses=("needs_work", "needs_work"),
    )
    Path(prepared["assessment_path"]).write_text(
        json.dumps(assessment), encoding="utf-8"
    )

    result = cli_runner("finalize-verification", "--stage", prepared["stage"])
    payload = json.loads(result.stdout)

    assert prepared_result.exit_code == 0
    assert prepared["status"] == "awaiting_model_assessment"
    assert result.exit_code == 0
    assert payload["status"] == "finalized"
    assert Path(payload["verification_report"]).is_file()
