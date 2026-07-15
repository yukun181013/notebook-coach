"""Deterministic baseline, report, and challenge renderers."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import nbformat

from notebook_coach import RUBRIC_VERSION, SCHEMA_VERSION
from notebook_coach.scoring import DEDUCTIONS, WEIGHTS, score_issues


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def build_baseline(
    snapshot: dict[str, Any],
    assessment: dict[str, Any],
    *,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    source = copy.deepcopy(snapshot["source"])
    if source_path is not None:
        source["path"] = str(Path(source_path).expanduser().resolve(strict=False))
    baseline = {
        "schema_version": SCHEMA_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "run_id": assessment["run_id"],
        "immutable": True,
        "analysis_mode": "static",
        "evidence_origins": ["notebook_source", "saved_notebook_output"],
        "source": source,
        "snapshot": copy.deepcopy(snapshot),
        "notebook_summary": assessment["notebook_summary"],
        "concept_map": copy.deepcopy(assessment["concept_map"]),
        "issues": copy.deepcopy(assessment["issues"]),
        "challenges": copy.deepcopy(assessment["challenges"]),
        "score": score_issues(assessment["issues"]),
    }
    baseline["analysis_id"] = hashlib.sha256(_canonical_json(baseline)).hexdigest()
    return baseline


def build_report_state(baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": baseline["run_id"],
        "revision": 1,
        "execution_reviews": [],
    }


def _cell_label(indices: list[int]) -> str:
    return ", ".join(f"Cell {index}" for index in indices)


def render_report(
    baseline: dict[str, Any], report_state: dict[str, Any]
) -> str:
    if report_state.get("run_id") != baseline.get("run_id"):
        raise ValueError("Report state run ID does not match baseline.")

    lines = [
        "# Notebook Coach Report",
        "",
        f"Run ID: `{baseline['run_id']}`  ",
        f"Revision: {report_state['revision']}  ",
        "Analysis mode: static",
        "",
        "## Notebook Overview",
        "",
        baseline["notebook_summary"],
        "",
        "## Concept Map",
        "",
    ]
    lines.extend(f"- {concept}" for concept in baseline["concept_map"])
    lines.extend(["", "## Key Issues and Cell Evidence", ""])
    if not baseline["issues"]:
        lines.append("No evidence-backed issues were identified.")
    for issue in baseline["issues"]:
        deduction = DEDUCTIONS[issue["severity"]]
        lines.extend(
            [
                f"### {issue['issue_id']} — {issue['category']}",
                "",
                f"- Severity: `{issue['severity']}`",
                f"- Dimension: `{issue['dimension']}`",
                f"- Evidence: {_cell_label(issue['cell_indices'])} — {issue['evidence']}",
                f"- Impact: {issue['impact']}",
                f"- Recommendation: {issue['recommendation']}",
                f"- Rubric deduction: {deduction} points",
                "",
            ]
        )

    score = baseline["score"]
    lines.extend(["## Learning Evidence Score", ""])
    for dimension in WEIGHTS:
        lines.append(f"- {dimension}: {score['dimensions'][dimension]}")
    lines.extend([f"- Total: **{score['total']}/100**", ""])

    lines.extend(["## Recommended Challenges", ""])
    for challenge in baseline["challenges"]:
        criteria = "; ".join(challenge["acceptance_criteria"])
        lines.extend(
            [
                f"### {challenge['challenge_id']} — {challenge['title']}",
                "",
                challenge["prompt"],
                "",
                f"Acceptance criteria: {criteria}",
                "",
            ]
        )

    lines.extend(["## Optional Execution Results and Limits", ""])
    reviews = report_state.get("execution_reviews", [])
    if not reviews:
        lines.append(
            "No notebook code was executed. Findings use source and saved-output evidence only."
        )
    else:
        lines.extend(f"- {review['summary']}" for review in reviews)
    return "\n".join(lines).rstrip() + "\n"


def build_challenge_notebook(assessment: dict[str, Any]):
    challenges = assessment["challenges"]
    instruction = nbformat.v4.new_markdown_cell(
        "# Notebook Coach Challenges\n\n"
        "Complete both TODO sections, then run verification from the original run directory."
    )
    editable_cells = []
    for challenge in challenges:
        criteria = "\n".join(
            f"- {item}" for item in challenge["acceptance_criteria"]
        )
        if challenge["kind"] == "code":
            source = (
                f"# {challenge['challenge_id']}: {challenge['title']}\n"
                f"# {challenge['prompt']}\n"
                f"# Acceptance criteria:\n"
                + "\n".join(f"# - {item}" for item in challenge["acceptance_criteria"])
                + "\n\n# TODO: write your code here\n"
            )
            cell = nbformat.v4.new_code_cell(source)
        else:
            source = (
                f"## {challenge['challenge_id']}: {challenge['title']}\n\n"
                f"{challenge['prompt']}\n\n"
                f"Acceptance criteria:\n{criteria}\n\n"
                "TODO: write your explanation here."
            )
            cell = nbformat.v4.new_markdown_cell(source)
        cell.metadata["notebook_coach"] = {
            "challenge_id": challenge["challenge_id"],
            "source_issue_ids": list(challenge["source_issue_ids"]),
        }
        editable_cells.append(cell)

    rubric = nbformat.v4.new_markdown_cell(
        "# Self-check Rubric\n\n"
        "- The code task meets every acceptance criterion.\n"
        "- The concept task explains the reasoning in your own words."
    )
    notebook = nbformat.v4.new_notebook(
        cells=[instruction, *editable_cells, rubric]
    )
    notebook.metadata["notebook_coach"] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": assessment["run_id"],
        "challenge_ids": [item["challenge_id"] for item in challenges],
        "source_issue_ids": {
            item["challenge_id"]: list(item["source_issue_ids"])
            for item in challenges
        },
        "initial_content_hashes": {
            cell.metadata["notebook_coach"]["challenge_id"]: hashlib.sha256(
                cell.source.encode("utf-8")
            ).hexdigest()
            for cell in editable_cells
        },
    }
    return notebook
