from __future__ import annotations

import json
from pathlib import Path

import pytest

from notebook_coach.revisions import RevisionError, apply_execution_review
from notebook_coach.workflows import (
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


def _verification(run_dir: Path, source: Path, value: dict, *, statuses):
    stage = prepare_verification(source, run_dir)
    assessment = json.loads(json.dumps(value))
    assessment["run_id"] = stage.run_id
    for result, status in zip(assessment["issue_results"], statuses, strict=True):
        result["status"] = status
    for result in assessment["challenge_results"]:
        result["status"] = "needs_work"
    return finalize_verification(stage.stage_dir, assessment)


def _write_log(
    run_dir: Path,
    *,
    phase: str,
    target: str,
    attempt_id: str,
) -> Path:
    if phase == "diagnosis":
        state = json.loads((run_dir / "baseline.json").read_text("utf-8"))
        analysis_id = state["analysis_id"]
        target_state = state["source"]
    else:
        state = json.loads(
            (run_dir / ".notebook-coach/verification-state.json").read_text("utf-8")
        )
        analysis_id = state["assessment_id"]
        target_state = state[f"{target}_target"]
    body = {
        "schema_version": "1.0",
        "run_id": state["run_id"],
        "phase": phase,
        "target": target,
        "attempt_id": attempt_id,
        "analysis_id": analysis_id,
        "target_path": target_state["path"],
        "target_sha256": target_state["sha256"],
        "target_sha256_before": target_state["sha256"],
        "target_sha256_copy": target_state["sha256"],
        "target_sha256_after": target_state["sha256"],
        "status": "completed",
        "evidence_eligible": True,
    }
    path = run_dir / f"execution-{phase}-{target}-{attempt_id}.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _write_review(path: Path, **overrides) -> Path:
    body = {
        "schema_version": "1.0",
        "run_id": overrides.pop("run_id"),
        "phase": overrides.pop("phase"),
        "target": overrides.pop("target"),
        "attempt_id": overrides.pop("attempt_id"),
        "execution_log": overrides.pop("execution_log"),
        "issue_updates": [],
        "new_issues": [],
        "challenge_updates": [],
        **overrides,
    }
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_diagnosis_review_revises_report_without_mutating_baseline_or_challenge(
    notebook_path: Path, tmp_path: Path, valid_diagnosis: dict
):
    run_dir = _run(notebook_path, tmp_path / "runs", valid_diagnosis)
    baseline_before = (run_dir / "baseline.json").read_bytes()
    challenge_before = (run_dir / "challenge.ipynb").read_bytes()
    baseline = json.loads(baseline_before)
    log_path = _write_log(
        run_dir, phase="diagnosis", target="source", attempt_id="A001"
    )
    review_path = _write_review(
        tmp_path / "review.json",
        run_id=baseline["run_id"],
        phase="diagnosis",
        target="source",
        attempt_id="A001",
        execution_log=log_path.name,
        issue_updates=[
            {
                "issue_id": "I001",
                "evidence_label": "supported",
                "evidence": "Cell 1 reproduced the normalization error.",
            }
        ],
    )

    result = apply_execution_review(run_dir, log_path, review_path)

    state = json.loads(
        (run_dir / ".notebook-coach/report-state.json").read_text("utf-8")
    )
    assert result.revision == 2
    assert state["revision"] == 2
    assert state["execution_reviews"][0]["execution_log"] == log_path.name
    assert (run_dir / "baseline.json").read_bytes() == baseline_before
    assert (run_dir / "challenge.ipynb").read_bytes() == challenge_before
    assert json.loads(baseline_before)["score"] == baseline["score"]


def test_verification_source_and_challenge_reviews_keep_target_boundaries(
    notebook_path: Path,
    tmp_path: Path,
    valid_diagnosis: dict,
    valid_verification: dict,
):
    run_dir = _run(notebook_path, tmp_path / "runs", valid_diagnosis)
    _verification(
        run_dir, notebook_path, valid_verification, statuses=("remaining", "remaining")
    )
    state = json.loads(
        (run_dir / ".notebook-coach/verification-state.json").read_text("utf-8")
    )
    source_log = _write_log(
        run_dir, phase="verification", target="source", attempt_id="A001"
    )
    source_review = _write_review(
        tmp_path / "source-review.json",
        run_id=state["run_id"],
        phase="verification",
        target="source",
        attempt_id="A001",
        execution_log=source_log.name,
        issue_updates=[
            {
                "issue_id": "I001",
                "status": "resolved",
                "evidence_label": "supported",
                "evidence": "The temporary source copy normalized every row.",
            }
        ],
    )
    first = apply_execution_review(run_dir, source_log, source_review)
    after_source = json.loads(
        (run_dir / ".notebook-coach/verification-state.json").read_text("utf-8")
    )
    score_after_source = after_source["after_score"]

    challenge_log = _write_log(
        run_dir, phase="verification", target="challenge", attempt_id="A002"
    )
    challenge_review = _write_review(
        tmp_path / "challenge-review.json",
        run_id=state["run_id"],
        phase="verification",
        target="challenge",
        attempt_id="A002",
        execution_log=challenge_log.name,
        challenge_updates=[
            {
                "challenge_id": "C-CODE",
                "status": "passed",
                "evidence_label": "supported",
                "evidence": "The code challenge ran successfully.",
            }
        ],
    )
    second = apply_execution_review(run_dir, challenge_log, challenge_review)
    final = json.loads(
        (run_dir / ".notebook-coach/verification-state.json").read_text("utf-8")
    )

    assert first.revision == 2
    assert second.revision == 3
    assert final["after_score"] == score_after_source
    statuses = {
        item["challenge_id"]: item["status"]
        for item in final["assessment"]["challenge_results"]
    }
    assert statuses == {"C-CODE": "passed", "C-CONCEPT": "needs_work"}


def test_cross_target_or_replayed_review_is_rejected(
    notebook_path: Path,
    tmp_path: Path,
    valid_diagnosis: dict,
    valid_verification: dict,
):
    run_dir = _run(notebook_path, tmp_path / "runs", valid_diagnosis)
    _verification(
        run_dir, notebook_path, valid_verification, statuses=("remaining", "remaining")
    )
    state = json.loads(
        (run_dir / ".notebook-coach/verification-state.json").read_text("utf-8")
    )
    log_path = _write_log(
        run_dir, phase="verification", target="source", attempt_id="A001"
    )
    invalid = _write_review(
        tmp_path / "invalid.json",
        run_id=state["run_id"],
        phase="verification",
        target="source",
        attempt_id="A001",
        execution_log=log_path.name,
        challenge_updates=[
            {
                "challenge_id": "C-CODE",
                "status": "passed",
                "evidence_label": "supported",
                "evidence": "Wrong target evidence.",
            }
        ],
    )
    with pytest.raises(RevisionError):
        apply_execution_review(run_dir, log_path, invalid)

    valid = _write_review(
        tmp_path / "valid.json",
        run_id=state["run_id"],
        phase="verification",
        target="source",
        attempt_id="A001",
        execution_log=log_path.name,
        issue_updates=[
            {
                "issue_id": "I001",
                "status": "resolved",
                "evidence_label": "supported",
                "evidence": "Valid source evidence.",
            }
        ],
    )
    apply_execution_review(run_dir, log_path, valid)
    with pytest.raises(RevisionError) as replay:
        apply_execution_review(run_dir, log_path, valid)
    assert replay.value.code == "execution_log_replayed"


def test_verification_review_rejects_new_issue_outside_source_cells(
    notebook_path: Path,
    tmp_path: Path,
    valid_diagnosis: dict,
    valid_verification: dict,
):
    run_dir = _run(notebook_path, tmp_path / "runs", valid_diagnosis)
    _verification(
        run_dir, notebook_path, valid_verification, statuses=("remaining", "remaining")
    )
    state = json.loads(
        (run_dir / ".notebook-coach/verification-state.json").read_text("utf-8")
    )
    log_path = _write_log(
        run_dir, phase="verification", target="source", attempt_id="A001"
    )
    review_path = _write_review(
        tmp_path / "review.json",
        run_id=state["run_id"],
        phase="verification",
        target="source",
        attempt_id="A001",
        execution_log=log_path.name,
        new_issues=[
            {
                "issue_id": "I003",
                "dimension": "correctness",
                "severity": "major",
                "category": "code",
                "cell_indices": [999999],
                "evidence": "The new issue is outside the analyzed notebook.",
                "impact": "Evidence cannot support this issue.",
                "recommendation": "Reference an existing source cell.",
            }
        ],
    )

    with pytest.raises(RevisionError) as error:
        apply_execution_review(run_dir, log_path, review_path)

    assert error.value.code == "unknown_cell_index"


def test_verification_review_accepts_legacy_state_without_source_cell_count(
    notebook_path: Path,
    tmp_path: Path,
    valid_diagnosis: dict,
    valid_verification: dict,
):
    run_dir = _run(notebook_path, tmp_path / "runs", valid_diagnosis)
    _verification(
        run_dir, notebook_path, valid_verification, statuses=("remaining", "remaining")
    )
    state_path = run_dir / ".notebook-coach/verification-state.json"
    state = json.loads(state_path.read_text("utf-8"))
    state["source_target"].pop("cell_count")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    log_path = _write_log(
        run_dir, phase="verification", target="source", attempt_id="A001"
    )
    review_path = _write_review(
        tmp_path / "legacy-review.json",
        run_id=state["run_id"],
        phase="verification",
        target="source",
        attempt_id="A001",
        execution_log=log_path.name,
        issue_updates=[
            {
                "issue_id": "I001",
                "status": "resolved",
                "evidence_label": "supported",
                "evidence": "Legacy verification state remains reviewable.",
            }
        ],
    )

    result = apply_execution_review(run_dir, log_path, review_path)

    assert result.revision == 2
    migrated = json.loads(state_path.read_text("utf-8"))
    assert migrated["source_target"]["cell_count"] == 2


def test_unverifiable_challenge_rejects_review_before_reading_log(
    notebook_path: Path,
    tmp_path: Path,
    valid_diagnosis: dict,
    valid_verification: dict,
):
    run_dir = _run(notebook_path, tmp_path / "runs", valid_diagnosis)
    _verification(
        run_dir, notebook_path, valid_verification, statuses=("remaining", "remaining")
    )
    state_path = run_dir / ".notebook-coach/verification-state.json"
    state = json.loads(state_path.read_text("utf-8"))
    state["challenge_verifiability"] = {
        "verifiable": False,
        "reason": "metadata_mismatch",
    }
    for result in state["assessment"]["challenge_results"]:
        result["status"] = "unverifiable"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(RevisionError) as error:
        apply_execution_review(
            run_dir,
            run_dir / "execution-verification-challenge-A001.json",
            tmp_path / "missing-review.json",
        )

    assert error.value.code == "challenge_unverifiable"


def test_two_file_failure_rolls_back_revision(
    notebook_path: Path, tmp_path: Path, valid_diagnosis: dict, monkeypatch
):
    import notebook_coach.revisions as revisions

    run_dir = _run(notebook_path, tmp_path / "runs", valid_diagnosis)
    baseline = json.loads((run_dir / "baseline.json").read_text("utf-8"))
    state_path = run_dir / ".notebook-coach/report-state.json"
    report_path = run_dir / "report.md"
    state_before = state_path.read_bytes()
    report_before = report_path.read_bytes()
    log_path = _write_log(
        run_dir, phase="diagnosis", target="source", attempt_id="A001"
    )
    review_path = _write_review(
        tmp_path / "review.json",
        run_id=baseline["run_id"],
        phase="diagnosis",
        target="source",
        attempt_id="A001",
        execution_log=log_path.name,
        issue_updates=[
            {
                "issue_id": "I001",
                "evidence_label": "supported",
                "evidence": "Runtime evidence.",
            }
        ],
    )
    real_replace = revisions.os.replace
    calls = 0

    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected")
        return real_replace(source, destination)

    monkeypatch.setattr(revisions.os, "replace", fail_second)
    with pytest.raises(OSError):
        apply_execution_review(run_dir, log_path, review_path)

    assert state_path.read_bytes() == state_before
    assert report_path.read_bytes() == report_before
