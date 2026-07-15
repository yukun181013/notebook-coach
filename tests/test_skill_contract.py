from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _skill_text() -> str:
    return (ROOT / "SKILL.md").read_text("utf-8")


def test_skill_frontmatter_and_trigger_contract():
    text = _skill_text()
    frontmatter = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
    assert frontmatter is not None
    header = frontmatter.group(1)
    assert re.search(r"^name: notebook-coach$", header, re.MULTILINE)
    description = re.search(r"^description: (.+)$", header, re.MULTILINE).group(1)
    assert description.startswith("Use when")
    for trigger in ("checking", "diagnosing", "coaching", "challenging", "rechecking"):
        assert trigger in description.lower()
    assert "jupyter notebook" in description.lower()


def test_skill_requires_static_gpt56_flow_without_api_key():
    text = _skill_text()
    assert "GPT-5.6 Codex" in text
    assert "default" in text.lower() and "static" in text.lower()
    assert "API key" in text
    assert "never ask" in text.lower()
    assert "does not execute" in text.lower()


def test_execution_confirmation_is_target_and_hash_bound():
    text = _skill_text()
    for field in (
        "phase",
        "target",
        "path",
        "SHA-256",
        "kernel",
        "temporary directory",
        "cell timeout",
        "total timeout",
    ):
        assert field in text
    assert "prepare-execution" in text
    assert "--confirmed-target-sha256" in text
    assert "source and challenge separately" in text.lower()
    assert "cancel-execution" in text
    assert "expired" in text and "new request" in text.lower()


def test_skill_handles_changed_or_blocked_targets_and_identity_mismatches():
    text = _skill_text()
    assert "risk" in text.lower() and "static-only" in text.lower()
    assert "target hash changes" in text.lower()
    assert "new static diagnosis or verification" in text.lower()
    assert "new immutable analysis ID" in text
    for mismatch in ("path mismatch", "kernel mismatch", "language mismatch"):
        assert mismatch in text.lower()
    assert "separate confirmation" in text.lower()


def test_skill_uses_only_staging_paths_and_never_fabricates_evidence():
    text = _skill_text()
    assert "CLI-provided assessment path" in text
    assert "diagnosis-assessment.json" in text
    assert "verification-assessment.json" in text
    assert "canonical JSON contract" in text
    assert "never fabricate execution evidence" in text.lower()
    assert "${SKILL_DIR}/.venv/bin/notebook-coach-tool" in text
    assert "/feedback" in text and "Session ID" in text
    assert "judge" in text.lower()
    assert "not an OS sandbox" in text


def test_openai_display_metadata_matches_skill():
    text = (ROOT / "agents/openai.yaml").read_text("utf-8")
    assert 'display_name: "Notebook Coach"' in text
    assert (
        'short_description: "Diagnose and recheck learning evidence in Jupyter notebooks"'
        in text
    )
    assert "$notebook-coach" in text

