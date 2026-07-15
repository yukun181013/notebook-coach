from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_readme_covers_judge_install_safety_and_architecture():
    text = (ROOT / "README.md").read_text("utf-8")
    for phrase in (
        "Python 3.11",
        "macOS",
        "Linux",
        "Windows",
        "WSL",
        "API key",
        "Codex",
        "GPT-5.6",
        "credits",
        "privacy",
        "redaction",
        "not an OS sandbox",
        "pytest -q",
        "$notebook-coach",
        "diagnose",
        "recheck",
    ):
        assert phrase.lower() in text.lower()
    assert "scripts/install.sh" in text
    assert "samples/transformer_attention_buggy.ipynb" in text
    assert "examples/transformer_attention_buggy/report.md" in text
    assert "Python" in text and "responsib" in text.lower()


def test_all_relative_markdown_links_exist():
    markdown_files = [
        ROOT / "README.md",
        ROOT / "docs/demo-script.md",
        ROOT / "docs/devpost-checklist.md",
        ROOT / "examples/transformer_attention_buggy/evaluation.md",
    ]
    for markdown in markdown_files:
        text = markdown.read_text("utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            clean = target.split("#", 1)[0]
            assert (markdown.parent / clean).resolve().exists(), (
                markdown,
                target,
            )


def test_demo_script_checklist_and_ci_are_submission_ready():
    demo = (ROOT / "docs/demo-script.md").read_text("utf-8")
    assert "0:00" in demo and "3:00" in demo
    assert "prepare-diagnosis" in demo
    assert "finalize-verification" in demo
    assert "Voiceover" in demo

    checklist = (ROOT / "docs/devpost-checklist.md").read_text("utf-8")
    for item in (
        "public repository",
        "YouTube",
        "GPT-5.6",
        "Codex",
        "/feedback",
        "Session ID",
        "fresh checkout",
        "three minutes",
    ):
        assert item.lower() in checklist.lower()

    workflow = (ROOT / ".github/workflows/tests.yml").read_text("utf-8")
    assert "ubuntu-latest" in workflow
    assert "macos-latest" in workflow
    assert "python-version: \"3.11\"" in workflow
    assert "pip install -e '.[test]'" in workflow
    assert "pytest -q" in workflow
    assert "pull_request:" in workflow and "push:" in workflow
