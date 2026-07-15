from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/install.sh"


def _run(home: Path, *arguments: str):
    environment = os.environ.copy()
    environment["CODEX_HOME"] = str(home)
    return subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_link_only_installs_repository_without_touching_real_codex_home(tmp_path: Path):
    codex_home = tmp_path / "codex"

    result = _run(codex_home, "--link-only")

    destination = codex_home / "skills/notebook-coach"
    assert result.returncode == 0, result.stderr
    assert destination.is_symlink()
    assert destination.resolve() == ROOT.resolve()


def test_reinstall_is_noop_but_unrelated_destination_is_refused(tmp_path: Path):
    codex_home = tmp_path / "codex"
    first = _run(codex_home, "--link-only")
    second = _run(codex_home, "--link-only")
    assert first.returncode == second.returncode == 0

    unrelated_home = tmp_path / "other-codex"
    destination = unrelated_home / "skills/notebook-coach"
    destination.mkdir(parents=True)
    (destination / "unrelated.txt").write_text("keep", encoding="utf-8")
    refused = _run(unrelated_home, "--link-only")
    assert refused.returncode != 0
    assert (destination / "unrelated.txt").read_text("utf-8") == "keep"


def test_installer_has_strict_shell_and_python311_checks():
    text = SCRIPT.read_text("utf-8")
    assert "set -eu" in text
    assert "python3.11" in text
    assert "-m venv" in text
    assert "pip install" in text
    assert subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, check=False
    ).returncode == 0
