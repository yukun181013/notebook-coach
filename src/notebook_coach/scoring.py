"""Deterministic Learning Evidence Score arithmetic."""

from __future__ import annotations

from typing import Any, Iterable

from notebook_coach import RUBRIC_VERSION
from notebook_coach.contracts import DIMENSIONS, SEVERITIES


WEIGHTS = {
    "correctness": 30,
    "concept_completeness": 30,
    "reproducibility": 20,
    "clarity": 20,
}
DEDUCTIONS = {"blocking": 10, "major": 5, "minor": 2}


def score_issues(issues: Iterable[dict[str, Any]]) -> dict[str, Any]:
    deductions = {dimension: 0 for dimension in DIMENSIONS}
    for issue in issues:
        dimension = issue.get("dimension")
        severity = issue.get("severity")
        if dimension not in WEIGHTS:
            raise ValueError("Unsupported scoring dimension.")
        if severity not in SEVERITIES:
            raise ValueError("Unsupported scoring severity.")
        deductions[dimension] += DEDUCTIONS[severity]

    dimensions = {
        dimension: max(0, weight - deductions[dimension])
        for dimension, weight in WEIGHTS.items()
    }
    return {
        "rubric_version": RUBRIC_VERSION,
        "dimensions": dimensions,
        "total": sum(dimensions.values()),
    }


def score_verification(
    baseline_issues: list[dict[str, Any]],
    issue_results: list[dict[str, Any]],
    new_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    statuses = {result["issue_id"]: result["status"] for result in issue_results}
    remaining = [
        issue
        for issue in baseline_issues
        if statuses.get(issue.get("issue_id")) != "resolved"
    ]
    return score_issues([*remaining, *new_issues])
