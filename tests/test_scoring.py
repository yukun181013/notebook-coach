from notebook_coach.scoring import score_issues, score_verification


def test_rubric_v1_applies_dimension_caps_and_severity_deductions():
    issues = [
        {"dimension": "correctness", "severity": "blocking"},
        {"dimension": "correctness", "severity": "major"},
        {"dimension": "clarity", "severity": "minor"},
    ]

    score = score_issues(issues)

    assert score["dimensions"] == {
        "correctness": 15,
        "concept_completeness": 30,
        "reproducibility": 20,
        "clarity": 18,
    }
    assert score["total"] == 83
    assert score["rubric_version"] == "1.0"


def test_verification_removes_only_resolved_deductions(valid_diagnosis):
    results = [
        {"issue_id": "I001", "status": "resolved"},
        {"issue_id": "I002", "status": "remaining"},
    ]

    score = score_verification(valid_diagnosis["issues"], results, [])

    assert score["dimensions"]["correctness"] == 30
    assert score["dimensions"]["clarity"] == 18
    assert score["total"] == 98
