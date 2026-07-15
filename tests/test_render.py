from __future__ import annotations

import json

import nbformat

from notebook_coach.render import (
    build_baseline,
    build_challenge_notebook,
    build_report_state,
    render_report,
)


def test_rendered_diagnosis_artifacts_are_linked(snapshot, valid_diagnosis):
    baseline = build_baseline(snapshot, valid_diagnosis)
    report_state = build_report_state(baseline)
    report = render_report(baseline, report_state)
    challenge = build_challenge_notebook(valid_diagnosis)

    assert baseline["run_id"] == valid_diagnosis["run_id"]
    assert baseline["score"]["total"] == 93
    assert "## Learning Evidence Score" in report
    assert "Revision: 1" in report
    assert report_state == {
        "schema_version": "1.0",
        "run_id": baseline["run_id"],
        "revision": 1,
        "execution_reviews": [],
    }
    assert len(challenge.cells) == 4
    assert challenge.metadata["notebook_coach"]["run_id"] == valid_diagnosis["run_id"]
    assert challenge.metadata["notebook_coach"]["challenge_ids"] == [
        "C-CODE",
        "C-CONCEPT",
    ]
    assert "TODO" in "\n".join(cell.source for cell in challenge.cells)


def test_report_names_issue_ids_cells_and_all_six_sections(
    snapshot, valid_diagnosis
):
    baseline = build_baseline(snapshot, valid_diagnosis)
    report = render_report(baseline, build_report_state(baseline))

    for heading in [
        "## Notebook Overview",
        "## Concept Map",
        "## Key Issues and Cell Evidence",
        "## Learning Evidence Score",
        "## Recommended Challenges",
        "## Optional Execution Results and Limits",
    ]:
        assert heading in report
    assert "I001" in report
    assert "Cell 1" in report


def test_challenge_notebook_has_no_hidden_solution_metadata(valid_diagnosis):
    challenge = build_challenge_notebook(valid_diagnosis)
    serialized = nbformat.writes(challenge)
    metadata_text = json.dumps(challenge.metadata, sort_keys=True).lower()

    assert "solution" not in metadata_text
    assert "answer" not in metadata_text
    assert "acceptance criteria" in serialized.lower()
    assert "I001" in serialized
