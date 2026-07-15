from __future__ import annotations

import json
from pathlib import Path


def test_prepare_diagnosis_cli_prints_machine_readable_next_action(
    cli_runner, notebook_path: Path, tmp_path: Path
):
    result = cli_runner(
        "prepare-diagnosis",
        str(notebook_path),
        "--output-root",
        str(tmp_path / "runs"),
    )

    payload = json.loads(result.stdout)
    assert result.exit_code == 0
    assert result.stderr == ""
    assert payload["status"] == "awaiting_model_assessment"
    assert Path(payload["assessment_path"]).is_absolute()
    assert payload["assessment_path"].endswith("diagnosis-assessment.json")
    assert payload["risk"]["blocked"] is False


def test_prepare_diagnosis_cli_supports_one_based_cell_ranges(
    cli_runner, notebook_path: Path, tmp_path: Path
):
    result = cli_runner(
        "prepare-diagnosis",
        str(notebook_path),
        "--output-root",
        str(tmp_path / "runs"),
        "--cells",
        "1-2",
    )

    payload = json.loads(result.stdout)
    stage = Path(payload["stage"])
    snapshot = json.loads((stage / "snapshot.json").read_text("utf-8"))
    assert result.exit_code == 0
    assert [cell["index"] for cell in snapshot["cells"]] == [0, 1]


def test_finalize_and_validate_cli_complete_the_run(
    cli_runner, notebook_path: Path, tmp_path: Path
):
    prepared_result = cli_runner(
        "prepare-diagnosis",
        str(notebook_path),
        "--output-root",
        str(tmp_path / "runs"),
    )
    prepared = json.loads(prepared_result.stdout)
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/diagnosis-assessment.json").read_text(
            "utf-8"
        )
    )
    fixture["run_id"] = prepared["run_id"]
    Path(prepared["assessment_path"]).write_text(
        json.dumps(fixture), encoding="utf-8"
    )

    finalized_result = cli_runner("finalize-diagnosis", "--stage", prepared["stage"])
    finalized = json.loads(finalized_result.stdout)
    validated_result = cli_runner("validate-run", finalized["run_dir"])
    validated = json.loads(validated_result.stdout)

    assert finalized_result.exit_code == 0
    assert finalized["status"] == "finalized"
    assert Path(finalized["run_dir"]).is_absolute()
    assert validated_result.exit_code == 0
    assert validated == {
        "run_dir": finalized["run_dir"],
        "status": "valid",
    }


def test_contract_error_is_json_on_stderr_and_keeps_stage(
    cli_runner, notebook_path: Path, tmp_path: Path
):
    prepared_result = cli_runner(
        "prepare-diagnosis",
        str(notebook_path),
        "--output-root",
        str(tmp_path / "runs"),
    )
    prepared = json.loads(prepared_result.stdout)
    Path(prepared["assessment_path"]).write_text("{}", encoding="utf-8")

    result = cli_runner("finalize-diagnosis", "--stage", prepared["stage"])
    error = json.loads(result.stderr)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert set(error) == {"code", "message"}
    assert Path(prepared["stage"]).is_dir()


def test_resolve_run_cli_does_not_guess_between_multiple_runs(
    cli_runner, notebook_path: Path, tmp_path: Path, baseline_factory
):
    from notebook_coach.runs import RunStore

    output_root = tmp_path / "runs"
    store = RunStore(output_root)
    baseline_factory(store, source_path=notebook_path, run_id="r1")
    baseline_factory(store, source_path=notebook_path, run_id="r2")

    result = cli_runner(
        "resolve-run",
        str(notebook_path),
        "--output-root",
        str(output_root),
    )
    error = json.loads(result.stderr)

    assert result.exit_code == 3
    assert error["code"] == "ambiguous_run"
    assert error["candidates"] == ["r1", "r2"]


def test_validate_run_detects_report_tampering(
    cli_runner, notebook_path: Path, tmp_path: Path
):
    prepared_result = cli_runner(
        "prepare-diagnosis",
        str(notebook_path),
        "--output-root",
        str(tmp_path / "runs"),
    )
    prepared = json.loads(prepared_result.stdout)
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/diagnosis-assessment.json").read_text(
            "utf-8"
        )
    )
    fixture["run_id"] = prepared["run_id"]
    Path(prepared["assessment_path"]).write_text(
        json.dumps(fixture), encoding="utf-8"
    )
    finalized = json.loads(
        cli_runner("finalize-diagnosis", "--stage", prepared["stage"]).stdout
    )
    (Path(finalized["run_dir"]) / "report.md").write_text(
        "tampered", encoding="utf-8"
    )

    result = cli_runner("validate-run", finalized["run_dir"])

    assert result.exit_code == 2
    assert json.loads(result.stderr)["code"] == "report_mismatch"


def test_prepare_execution_cli_provides_run_scoped_review_path(
    cli_runner, notebook_path: Path, tmp_path: Path
):
    prepared = json.loads(
        cli_runner(
            "prepare-diagnosis",
            str(notebook_path),
            "--output-root",
            str(tmp_path / "runs"),
        ).stdout
    )
    assessment = json.loads(
        (Path(__file__).parent / "fixtures/diagnosis-assessment.json").read_text(
            "utf-8"
        )
    )
    assessment["run_id"] = prepared["run_id"]
    Path(prepared["assessment_path"]).write_text(
        json.dumps(assessment), encoding="utf-8"
    )
    finalized = json.loads(
        cli_runner("finalize-diagnosis", "--stage", prepared["stage"]).stdout
    )

    result = cli_runner(
        "prepare-execution",
        "--run",
        finalized["run_dir"],
        "--phase",
        "diagnosis",
        "--target",
        "source",
    )
    payload = json.loads(result.stdout)

    assert result.exit_code == 0
    assert payload["review_path"] == str(
        Path(finalized["run_dir"])
        / ".notebook-coach/reviews/A001/execution-review.json"
    )
    assert Path(payload["review_path"]).parent.is_dir()
