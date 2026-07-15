# Notebook Coach Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an installable Codex Skill that statically diagnoses Python/LLM course notebooks, generates evidence-based learning challenges, and verifies improvements without requiring an OpenAI API key.

**Architecture:** A local Python 3.11 package performs deterministic parsing, redaction, risk scanning, scoring, run management, optional temporary-copy execution, and artifact rendering. The Codex Skill orchestrates GPT-5.6 judgments through validated JSON handoff files; no Python module calls a model or network API. Immutable baselines and run-scoped IDs keep diagnosis, challenges, execution evidence, and verification linked without modifying the source notebook.

**Tech Stack:** Python 3.11, `nbformat`, `nbclient`, `pytest`, standard-library `argparse`/`json`/`hashlib`/`subprocess`/`tempfile`, Codex Skills Markdown and YAML.

---

## Source specification and execution rules

- Implement against `docs/superpowers/specs/2026-07-15-notebook-coach-design.md`.
- Before Task 1, create or enter a dedicated feature worktree with `@superpowers:using-git-worktrees`.
- Use `@superpowers:test-driven-development` for every production-code task: red test, observed failure, minimum implementation, green test, commit.
- Do not add OpenAI SDKs, API-key handling, a web service, a database, or automatic dependency installation at runtime.
- All paths below are relative to the repository root.
- Default execution limits are 30 seconds per cell and 120 seconds per selected target; tests inject shorter limits.
- All model-facing JSON is already redacted and bounded before Codex reads it.

## File and responsibility map

| Path | Responsibility |
| --- | --- |
| `SKILL.md` | User triggers, GPT-5.6 orchestration, confirmations, model-output rules, error flow |
| `agents/openai.yaml` | Codex display metadata |
| `pyproject.toml` | Python package, dependencies, console entry point, pytest settings |
| `src/notebook_coach/contracts.py` | Schema constants and strict validators for GPT handoff JSON |
| `src/notebook_coach/sanitize.py` | Secret redaction, output truncation, safe text summaries |
| `src/notebook_coach/notebooks.py` | Notebook loading, cell selection, normalized static snapshots |
| `src/notebook_coach/risk.py` | Static risky-code findings and execution blocking decision |
| `src/notebook_coach/runs.py` | Source/run IDs, staging, atomic writes, exact run resolution |
| `src/notebook_coach/scoring.py` | Rubric v1 arithmetic and before/after score calculation |
| `src/notebook_coach/render.py` | Baseline, report, challenge notebook, verification rendering |
| `src/notebook_coach/workflows.py` | Diagnosis and verification staging/finalization use cases |
| `src/notebook_coach/execution.py` | Confirmation gate, target isolation, subprocess timeout, log creation |
| `src/notebook_coach/execution_worker.py` | Executes one temporary Notebook copy with `nbclient` |
| `src/notebook_coach/revisions.py` | Validates execution reviews and atomically revises reports |
| `src/notebook_coach/cli.py` | Machine-readable CLI used by `SKILL.md` |
| `scripts/install.sh` | One-minute local Skill installation and development linking |
| `tests/` | Unit, integration, CLI, Skill-contract, and acceptance tests |
| `samples/transformer_attention_buggy.ipynb` | Safe judge/demo input |
| `examples/transformer_attention_buggy/` | Pre-generated baseline, report, challenge, and verification |
| `README.md` | Judge-first install, usage, architecture, safety, cost, tests |
| `docs/demo-script.md` | Under-three-minute narrated demo script |
| `docs/devpost-checklist.md` | Submission requirements and evidence checklist |
| `.github/workflows/tests.yml` | Python 3.11 tests on Ubuntu and macOS |

Each finalized run also contains mutable machine state under `.notebook-coach/`: `report-state.json`, `execution-ledger.json`, and, after recheck, `verification-state.json`. Report/verification state is the source of truth for Markdown revisions. The execution ledger persists monotonic attempt IDs and lifecycle tombstones across CLI processes. All contain only already-redacted structured data and are validated by `validate-run`; `baseline.json` remains immutable.

## Stable JSON contracts

Use these values everywhere; do not invent parallel names in individual tasks:

```python
SCHEMA_VERSION = "1.0"
RUBRIC_VERSION = "1.0"
ISSUE_ID_PATTERN = r"I[0-9]{3}"
CHALLENGE_IDS = ("C-CODE", "C-CONCEPT")
DIMENSIONS = ("correctness", "concept_completeness", "reproducibility", "clarity")
SEVERITIES = ("blocking", "major", "minor")
```

Diagnosis assessment shape:

```json
{
  "schema_version": "1.0",
  "run_id": "20260715T080000Z-a1b2c3d4",
  "notebook_summary": "...",
  "concept_map": ["scaled dot-product attention"],
  "issues": [
    {
      "issue_id": "I001",
      "dimension": "correctness",
      "severity": "major",
      "category": "code",
      "cell_indices": [3],
      "evidence": "Cell 3 applies softmax on the wrong axis.",
      "impact": "Attention weights do not normalize over keys.",
      "recommendation": "Normalize the score row over the key dimension."
    }
  ],
  "challenges": [
    {
      "challenge_id": "C-CODE",
      "kind": "code",
      "source_issue_ids": ["I001"],
      "title": "Repair attention normalization",
      "prompt": "Complete the TODO without copying an answer.",
      "acceptance_criteria": ["Each query row sums to one"]
    },
    {
      "challenge_id": "C-CONCEPT",
      "kind": "concept",
      "source_issue_ids": ["I001"],
      "title": "Explain the normalization axis",
      "prompt": "Explain why the key axis is normalized.",
      "acceptance_criteria": ["Mentions one distribution per query"]
    }
  ]
}
```

Verification assessment shape:

```json
{
  "schema_version": "1.0",
  "run_id": "20260715T080000Z-a1b2c3d4",
  "issue_results": [
    {
      "issue_id": "I001",
      "status": "resolved",
      "current_cell_indices": [3],
      "evidence": "Cell 3 now normalizes over the key dimension."
    }
  ],
  "new_issues": [],
  "challenge_results": [
    {"challenge_id": "C-CODE", "status": "passed", "evidence": "The TODO is completed."},
    {"challenge_id": "C-CONCEPT", "status": "passed", "evidence": "The explanation identifies one distribution per query."}
  ],
  "next_learning_target": "Relate attention weights to value aggregation."
}
```

## Task 1: Bootstrap the Python package

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/notebook_coach/__init__.py`
- Create: `src/notebook_coach/cli.py`
- Create: `tests/test_package.py`

- [ ] **Step 1: Create packaging metadata and install the test environment**

Create `pyproject.toml` with this contract:

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "notebook-coach"
version = "0.1.0"
description = "Evidence-based Codex coaching for Jupyter notebooks"
requires-python = ">=3.11"
dependencies = [
  "nbformat>=5.10,<6",
  "nbclient>=0.10,<1",
  "ipykernel>=6.29,<7",
]

[project.optional-dependencies]
test = ["pytest>=8,<9"]

[project.scripts]
notebook-coach-tool = "notebook_coach.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests"]
```

Create `src/notebook_coach/__init__.py` exporting only `__version__ = "0.1.0"`, `SCHEMA_VERSION = "1.0"`, and `RUBRIC_VERSION = "1.0"`. Create `.gitignore` with `.venv/`, `.acceptance-venv/`, `artifacts/`, `__pycache__/`, `.pytest_cache/`, `*.egg-info/`, `notebook-coach-output/`, and `dist/`.

Run: `python3.11 -m venv .venv`

Run: `.venv/bin/python -m pip install -e '.[test]'`

Expected: installation succeeds; no product behavior has been implemented yet.

- [ ] **Step 2: Write the failing package smoke test**

```python
# tests/test_package.py
from notebook_coach import RUBRIC_VERSION, SCHEMA_VERSION, __version__
from notebook_coach.cli import build_parser


def test_package_exports_versions_and_cli_commands():
    assert __version__ == "0.1.0"
    assert SCHEMA_VERSION == "1.0"
    assert RUBRIC_VERSION == "1.0"
    assert set(build_parser()._subparsers._group_actions[0].choices) == {
        "prepare-diagnosis",
        "finalize-diagnosis",
        "resolve-run",
        "prepare-verification",
        "finalize-verification",
        "prepare-execution",
        "execute",
        "cancel-execution",
        "apply-execution-review",
        "validate-run",
    }
```

- [ ] **Step 3: Run the test and observe the expected product failure**

Run: `.venv/bin/python -m pytest tests/test_package.py -q`

Expected: FAIL because `notebook_coach.cli` does not exist, not because pytest is missing.

- [ ] **Step 4: Add the minimal CLI parser**

Create `build_parser()` in `cli.py` with the ten empty subcommands above and `main(argv=None)` returning the selected handler's integer exit code.

- [ ] **Step 5: Run the smoke test**

Run: `.venv/bin/python -m pytest tests/test_package.py -q`

Expected: `1 passed`.

- [ ] **Step 6: Commit the bootstrap**

```bash
git add .gitignore pyproject.toml src/notebook_coach tests/test_package.py
git commit -m "chore: bootstrap Notebook Coach package"
```

## Task 2: Redact secrets and bound model-facing text

**Files:**
- Create: `src/notebook_coach/sanitize.py`
- Create: `tests/test_sanitize.py`

- [ ] **Step 1: Write failing redaction and truncation tests**

```python
# tests/test_sanitize.py
from notebook_coach.sanitize import redact_text, summarize_text


def test_redacts_secret_without_returning_original_value():
    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
    cleaned, labels = redact_text(f"OPENAI_API_KEY={secret}")
    assert cleaned == "OPENAI_API_KEY=[REDACTED]"
    assert secret not in repr((cleaned, labels))
    assert labels == ["openai_api_key"]


def test_summarize_text_keeps_length_hash_and_marker():
    result = summarize_text("x" * 5000, max_chars=100)
    assert result["truncated"] is True
    assert result["original_chars"] == 5000
    assert len(result["text"]) <= 100
    assert len(result["sha256"]) == 64
```

- [ ] **Step 2: Run the tests and observe import failure**

Run: `.venv/bin/python -m pytest tests/test_sanitize.py -q`

Expected: FAIL because `sanitize.py` does not exist.

- [ ] **Step 3: Implement bounded, label-only redaction**

Implement `redact_text(text) -> tuple[str, list[str]]` with named patterns for OpenAI-style keys, GitHub tokens, PEM private keys, and assignments whose names contain `api_key`, `token`, `password`, or `secret`. Never return or log the matched value. Implement `summarize_text(text, max_chars=4000)` returning only `text`, `original_chars`, `sha256`, and `truncated`.

Use `[REDACTED]` for every replacement and apply redaction before truncation. Add a regression test proving a short ordinary value such as `token_count = 32` is not redacted.

- [ ] **Step 4: Run sanitizer tests**

Run: `.venv/bin/python -m pytest tests/test_sanitize.py -q`

Expected: all sanitizer tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/notebook_coach/sanitize.py tests/test_sanitize.py
git commit -m "feat: redact and bound notebook text"
```

## Task 3: Build deterministic static notebook snapshots

**Files:**
- Create: `src/notebook_coach/notebooks.py`
- Create: `tests/conftest.py`
- Create: `tests/test_notebooks.py`

- [ ] **Step 1: Add fixtures and failing snapshot tests**

Create a `notebook_factory(tmp_path)` fixture that writes valid notebooks with `nbformat`. Test these behaviors:

```python
from notebook_coach.notebooks import SnapshotLimitError, build_snapshot, parse_cell_selection


def test_snapshot_is_redacted_bounded_and_hashes_original(notebook_factory):
    path = notebook_factory(code="OPENAI_API_KEY='sk-proj-abcdefghijklmnopqrstuvwxyz123456'")
    before = path.read_bytes()
    snapshot = build_snapshot(path)
    assert snapshot["source"]["sha256"]
    assert snapshot["source"]["cell_count"] == 1
    assert "sk-proj" not in str(snapshot)
    assert snapshot["cells"][0]["index"] == 0
    assert path.read_bytes() == before


def test_cell_selection_is_one_based_for_users():
    assert parse_cell_selection("1-3,5", total_cells=6) == [0, 1, 2, 4]
```

Also test:

- invalid JSON raises `NotebookInputError`;
- more than 200 selected cells or more than 150,000 post-sanitization source characters raises `SnapshotLimitError` with a cell-range instruction;
- Base64 images and attachments are replaced by type/length/hash metadata;
- a saved `ModuleNotFoundError` is retained as static evidence without importing or installing the missing package;
- saved text/error outputs are redacted and bounded.

- [ ] **Step 2: Run the tests and observe failure**

Run: `.venv/bin/python -m pytest tests/test_notebooks.py -q`

Expected: FAIL because snapshot functions do not exist.

- [ ] **Step 3: Implement snapshot normalization**

`build_snapshot(path, selected_cells=None)` must:

1. read raw bytes and compute SHA-256 before parsing;
2. reject non-`.ipynb`, invalid JSON, and unsupported notebook structure;
3. whitelist only kernel name, language, notebook format, cell type/index/source/execution count;
4. keep redacted and bounded text outputs, error names, and error summaries;
5. replace image/base64/attachment bodies with type, length, SHA-256, and `omitted: true`;
6. record sanitization labels and truncation counts without original secret values;
7. enforce limits after the optional cell selection.

Return a JSON-serializable dictionary with `schema_version`, `source`, `cells`, and `sanitization` keys. Do not import execution code.

- [ ] **Step 4: Run notebook and sanitizer tests**

Run: `.venv/bin/python -m pytest tests/test_notebooks.py tests/test_sanitize.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/notebook_coach/notebooks.py tests/conftest.py tests/test_notebooks.py
git commit -m "feat: create safe notebook snapshots"
```

## Task 4: Detect risky execution patterns

**Files:**
- Create: `src/notebook_coach/risk.py`
- Modify: `tests/conftest.py`
- Create: `tests/test_risk.py`

- [ ] **Step 1: Write failing risk tests**

```python
from notebook_coach.risk import scan_snapshot


def test_blocks_shell_network_subprocess_and_delete_calls(snapshot_factory):
    snapshot = snapshot_factory([
        "!curl https://example.com",
        "import subprocess; subprocess.run(['echo', 'x'])",
        "import os; os.remove('/tmp/x')",
        "import requests; requests.get('https://example.com')",
    ])
    result = scan_snapshot(snapshot)
    assert result["blocked"] is True
    assert {item["category"] for item in result["findings"]} >= {
        "shell", "subprocess", "filesystem_delete", "network"
    }


def test_safe_math_notebook_is_not_blocked(snapshot_factory):
    assert scan_snapshot(snapshot_factory(["weights = scores.softmax(dim=-1)"]))["blocked"] is False
```

- [ ] **Step 2: Run and observe expected failure**

Run: `.venv/bin/python -m pytest tests/test_risk.py -q`

Expected: FAIL because `risk.py` does not exist.

- [ ] **Step 3: Implement conservative static findings**

Add `snapshot_factory` to `tests/conftest.py`. Use Python AST when parsing succeeds and regex fallback when it does not. Detect shell magics, `os.system`, `subprocess`, destructive `os`/`pathlib`/`shutil` calls, socket/HTTP libraries, package-install magics, and obvious credential-file reads. Each finding contains only `cell_index`, `category`, `severity`, and a redacted one-line explanation.

`blocked` is true for any high-severity execution finding. Syntax errors are findings for diagnosis but do not by themselves mark the notebook dangerous.

- [ ] **Step 4: Run risk tests**

Run: `.venv/bin/python -m pytest tests/test_risk.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/notebook_coach/risk.py tests/conftest.py tests/test_risk.py
git commit -m "feat: block risky notebook execution"
```

## Task 5: Add immutable run storage and exact run resolution

**Files:**
- Create: `src/notebook_coach/runs.py`
- Modify: `tests/conftest.py`
- Create: `tests/test_runs.py`

- [ ] **Step 1: Write failing run lifecycle tests**

```python
from datetime import datetime, timezone
import pytest

from notebook_coach.runs import AmbiguousRunError, RunStore


def test_run_ids_and_paths_are_deterministic(tmp_path):
    store = RunStore(tmp_path, now=lambda: datetime(2026, 7, 15, 8, tzinfo=timezone.utc))
    stage = store.create_stage("/course/lesson.ipynb", "a" * 64)
    assert stage.run_id == "20260715T080000Z-aaaaaaaa"
    assert stage.final_dir.parent.name.startswith("lesson-")


def test_multiple_matching_runs_never_choose_latest(tmp_path, baseline_factory):
    store = RunStore(tmp_path)
    baseline_factory(store, source_path="/course/lesson.ipynb", run_id="r1")
    baseline_factory(store, source_path="/course/lesson.ipynb", run_id="r2")
    with pytest.raises(AmbiguousRunError) as exc:
        store.resolve("/course/lesson.ipynb")
    assert exc.value.candidates == ["r1", "r2"]
```

Also test atomic writes leave an existing `baseline.json` unchanged after an injected failure, explicit run selection rejects a baseline whose normalized source path does not match until the caller confirms the mismatch, and two diagnoses created during the same UTC second receive distinct IDs (`...-02`, then `...-03`) without overwriting.

Add an execution-ledger test that reserves `A001`, transitions it to `cancelled`, deletes any request file, then reserves `A002` in a new `RunStore` instance. Add a concurrent reservation test proving two processes cannot receive the same attempt ID.

- [ ] **Step 2: Run and observe failure**

Run: `.venv/bin/python -m pytest tests/test_runs.py -q`

Expected: FAIL because run storage does not exist.

- [ ] **Step 3: Implement the run store**

Implement:

```python
class RunStore:
    def create_stage(self, source_path: str | Path, source_sha256: str) -> Stage: ...
    def finalize_stage(self, stage: Stage, files: dict[str, bytes]) -> Path: ...
    def resolve(
        self,
        source_path: str | Path,
        explicit_run: Path | None = None,
        allow_source_mismatch: bool = False,
    ) -> Path: ...
    def initialize_execution_ledger(self, run_dir: Path) -> None: ...
    def reserve_attempt(self, run_dir: Path, metadata: dict) -> AttemptReservation: ...
    def transition_attempt(
        self,
        run_dir: Path,
        attempt_id: str,
        expected_status: str,
        new_status: str,
        details: dict | None = None,
    ) -> None: ...
```

Add `baseline_factory` to `tests/conftest.py`. Use canonical absolute path text only to derive an eight-character `source-id`; do not expose it as user data beyond the directory name. Use same-filesystem temporary files plus `os.replace` for atomic writes. Never overwrite `baseline.json` or an execution log. Keep `.staging/` under the output root and delete a stage only after successful finalization. A source mismatch raises `SourceMismatchError` unless the Skill has separately confirmed the mismatch and passes `allow_source_mismatch=True`. If timestamp plus source hash collides, append the first available two-digit suffix while preserving monotonic creation order.

The ledger shape is `{"schema_version":"1.0","next_attempt":1,"entries":[]}`. Reserve and transition operations use an exclusive `fcntl.flock` on a sibling lock file, reload current JSON while locked, atomically replace the ledger, and then release. Entries are append-only except for legal status transitions (`prepared → running → completed|failed|expired`, or `prepared → cancelled|expired`). Cancelled/expired entries remain as tombstones, so deleted request files never permit ID reuse.

- [ ] **Step 4: Run run-store tests**

Run: `.venv/bin/python -m pytest tests/test_runs.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/notebook_coach/runs.py tests/conftest.py tests/test_runs.py
git commit -m "feat: manage immutable notebook runs"
```

## Task 6: Validate GPT assessments and compute rubric scores

**Files:**
- Create: `src/notebook_coach/contracts.py`
- Create: `src/notebook_coach/scoring.py`
- Modify: `tests/conftest.py`
- Create: `tests/fixtures/diagnosis-assessment.json`
- Create: `tests/fixtures/verification-assessment.json`
- Create: `tests/test_contracts.py`
- Create: `tests/test_scoring.py`

- [ ] **Step 1: Write failing assessment-contract tests**

Test the full diagnosis JSON shown above. Add negative cases for duplicate issue IDs, an unknown cell index, a missing challenge type, a challenge referencing an unknown issue, unsupported severity/dimension values, a mismatched `run_id`, and text containing a known unredacted fixture secret.

```python
def test_diagnosis_requires_exactly_one_code_and_one_concept_challenge(valid_diagnosis):
    validated = validate_diagnosis(valid_diagnosis, run_id=valid_diagnosis["run_id"], cell_count=6)
    assert [item["challenge_id"] for item in validated["challenges"]] == ["C-CODE", "C-CONCEPT"]
```

- [ ] **Step 2: Write failing scoring tests**

```python
from notebook_coach.scoring import score_issues


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
```

- [ ] **Step 3: Run and observe failure**

Run: `.venv/bin/python -m pytest tests/test_contracts.py tests/test_scoring.py -q`

Expected: FAIL because contract and scoring modules do not exist.

- [ ] **Step 4: Implement strict validators and local-only arithmetic**

Create canonical checked-in assessment fixtures with `"run_id": "__RUN_ID__"`. Add `valid_diagnosis` and `valid_verification` helpers to `tests/conftest.py` that load these JSON files and replace only that sentinel with the test stage's run ID; production validators must reject the sentinel. Validators return normalized copies and raise `ContractError` with a stable code plus a human-readable message. They must not repair model output silently. Scoring weights are 30/30/20/20 and deductions are 10/5/2 for blocking/major/minor, clamped to zero per dimension. Add `score_verification(baseline_issues, issue_results, new_issues)` that removes deductions only for `resolved` original issues and adds validated new issues.

- [ ] **Step 5: Run contract and score tests**

Run: `.venv/bin/python -m pytest tests/test_contracts.py tests/test_scoring.py -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/notebook_coach/contracts.py src/notebook_coach/scoring.py tests/conftest.py tests/fixtures tests/test_contracts.py tests/test_scoring.py
git commit -m "feat: validate assessments and score evidence"
```

## Task 7: Render baseline, report, and challenge artifacts

**Files:**
- Create: `src/notebook_coach/render.py`
- Modify: `tests/conftest.py`
- Create: `tests/test_render.py`

- [ ] **Step 1: Write failing artifact tests**

```python
def test_rendered_diagnosis_artifacts_are_linked(snapshot, valid_diagnosis):
    baseline = build_baseline(snapshot, valid_diagnosis)
    report_state = build_report_state(baseline)
    report = render_report(baseline, report_state)
    challenge = build_challenge_notebook(valid_diagnosis)
    assert baseline["run_id"] == valid_diagnosis["run_id"]
    assert "## Learning Evidence Score" in report
    assert "Revision: 1" in report
    assert report_state == {"schema_version": "1.0", "run_id": baseline["run_id"], "revision": 1, "execution_reviews": []}
    assert len(challenge.cells) == 4
    assert challenge.metadata["notebook_coach"]["run_id"] == valid_diagnosis["run_id"]
    assert challenge.metadata["notebook_coach"]["challenge_ids"] == ["C-CODE", "C-CONCEPT"]
    assert "TODO" in "\n".join(cell.source for cell in challenge.cells)
```

Assert the challenge contains no `answer`, `solution`, or hidden answer metadata; each task has acceptance criteria; every report deduction names an issue ID and cell; and binary data is absent from serialized artifacts.

- [ ] **Step 2: Run and observe failure**

Run: `.venv/bin/python -m pytest tests/test_render.py -q`

Expected: FAIL because render functions do not exist.

- [ ] **Step 3: Implement deterministic renderers**

Add `snapshot` and rendered-artifact fixtures to `tests/conftest.py`. `build_baseline` combines the redacted snapshot, validated assessment, rubric score, `analysis_mode: "static"`, and evidence-origin tags. `build_report_state` creates the mutable redacted revision state shown in the test; `render_report` always renders from immutable baseline plus that state and emits the six fixed sections from the spec. `build_challenge_notebook` creates one Markdown instruction cell plus one editable cell per challenge and a final rubric cell; store `run_id`, challenge IDs, source issue IDs, and the initial editable-content hashes under `metadata.notebook_coach`.

Use `nbformat` to create the Notebook and stable ordering for JSON/Markdown so fixtures are reviewable.

- [ ] **Step 4: Run render and contract tests**

Run: `.venv/bin/python -m pytest tests/test_render.py tests/test_contracts.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/notebook_coach/render.py tests/conftest.py tests/test_render.py
git commit -m "feat: render diagnosis learning artifacts"
```

## Task 8: Implement diagnosis staging and finalization CLI

**Files:**
- Create: `src/notebook_coach/workflows.py`
- Modify: `src/notebook_coach/cli.py`
- Modify: `tests/conftest.py`
- Create: `tests/test_diagnosis_workflow.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing diagnosis workflow tests**

Test this exact two-phase flow:

1. `prepare_diagnosis(source, output_root)` creates `.staging/<run-id>/snapshot.json`, `risk.json`, and an empty expected path `diagnosis-assessment.json` without final artifacts.
2. The test writes a valid assessment to the expected path, simulating Codex/GPT-5.6.
3. `finalize_diagnosis(stage)` validates it and atomically creates `baseline.json`, `report.md`, `challenge.ipynb`, `.notebook-coach/report-state.json`, and an empty `.notebook-coach/execution-ledger.json` in the final run.

Assert the source hash is unchanged and an invalid assessment leaves the stage intact with no final run.

Name the checked-fixture immutability case `test_static_flow_preserves_source_hash_with_checked_in_assessment`; it must load `tests/fixtures/diagnosis-assessment.json` and replace only the `__RUN_ID__` sentinel.

- [ ] **Step 2: Write failing CLI JSON-output tests**

```python
def test_prepare_diagnosis_cli_prints_machine_readable_next_action(cli_runner, notebook_path, tmp_path):
    result = cli_runner("prepare-diagnosis", str(notebook_path), "--output-root", str(tmp_path))
    payload = json.loads(result.stdout)
    assert result.exit_code == 0
    assert payload["status"] == "awaiting_model_assessment"
    assert payload["assessment_path"].endswith("diagnosis-assessment.json")
    assert payload["risk"]["blocked"] is False
```

Error output must be JSON on stderr with `code`, `message`, and optional `candidates`; exit codes: 0 success, 2 invalid input/contract, 3 ambiguous run, 4 execution blocked, 5 runtime failure.

- [ ] **Step 3: Run and observe failure**

Run: `.venv/bin/python -m pytest tests/test_diagnosis_workflow.py tests/test_cli.py -q`

Expected: FAIL because workflows and CLI handlers are incomplete.

- [ ] **Step 4: Implement workflow functions and CLI handlers**

Add `cli_runner`, `notebook_path`, and finalized-run fixtures to `tests/conftest.py`. The CLI must support `--cells 1-20,25`, use `notebook-coach-output/` as its default output root, print absolute paths for Codex, and never print full cell source or secrets. `finalize-diagnosis` takes `--stage <path>`. `resolve-run` accepts `--run <path>` and, only after the Skill has shown the mismatch and received confirmation, `--confirm-source-mismatch`. `validate-run` checks required artifacts, hidden state, baseline immutability marker, run IDs, challenge metadata, score arithmetic, and whether Markdown can be deterministically regenerated from structured state.

- [ ] **Step 5: Run diagnosis, CLI, and run tests**

Run: `.venv/bin/python -m pytest tests/test_diagnosis_workflow.py tests/test_cli.py tests/test_runs.py -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/notebook_coach/workflows.py src/notebook_coach/cli.py tests/conftest.py tests/test_diagnosis_workflow.py tests/test_cli.py
git commit -m "feat: add diagnosis workflow commands"
```

## Task 9: Prepare, confirm, and execute one hash-bound target

**Files:**
- Create: `src/notebook_coach/execution.py`
- Create: `src/notebook_coach/execution_worker.py`
- Modify: `src/notebook_coach/runs.py`
- Modify: `src/notebook_coach/cli.py`
- Create: `tests/test_execution.py`

- [ ] **Step 1: Write failing execution-preparation tests**

Cover all of these cases:

- `prepare-execution` resolves one target, scans it, reserves the actual empty temporary directory, and writes `execution-request.json` containing run ID, phase, target, attempt ID, one-time request ID, canonical target path, target SHA-256, immutable analysis ID, kernel, cell/total limits, risk result, expiry time, and temp directory;
- a high-risk finding returns exit 4 before any request can be confirmed;
- diagnosis only permits `diagnosis/source`;
- verification permits `verification/source` and `verification/challenge` separately;
- `cancel-execution --request <path>` removes the reserved directory/request without launching a kernel;
- first request reserves `A001`, a cancelled request leaves the gap, and the next request uses `A002` without reuse.

- [ ] **Step 2: Write failing confirmation-binding and target-isolation tests**

Cover all of these cases:

- `execute` without `--request` and `--confirmed-target-sha256` returns exit 4 without starting a process;
- the confirmed SHA-256 must exactly equal the request's displayed SHA-256;
- if the target changes after preparation, execution returns `target_changed_before_execution`, launches no worker, and requires a new preparation/confirmation cycle;
- the reserved temporary directory used by the worker is exactly the one shown before confirmation;
- selecting source never reads challenge execution output and vice versa;
- a verification state with `challenge_verifiability.verifiable: false` rejects challenge preparation without reserving an attempt;
- in a fresh run with no cancellation the first log is `execution-<phase>-<target>-A001.json` and the second is `A002`; if `A001` was cancelled, the first executed log is `A002`; no ID or log is reused or overwritten;
- source and challenge input hashes are unchanged after execution;
- one request cannot be executed twice, and a request older than one hour is atomically marked `expired` and cleaned without launching a worker;
- concurrent `prepare-execution` processes receive distinct attempt IDs through the persistent ledger.
- a `running` entry older than its total timeout plus 60 seconds is recovered as `failed` with reason `orphaned_worker`, and its temp directory is cleaned before a new reservation.

- [ ] **Step 3: Write failing timeout and log-sanitization tests**

Use a fixture Notebook containing `while True: pass` and inject `total_timeout=1`. Assert the worker is terminated, log status is `timeout`, and the test completes in under five seconds. Use a second Notebook that prints a fixture secret and 20,000 characters; assert stdout, stderr, and exception summaries in the log are redacted and bounded.

- [ ] **Step 4: Run and observe failure**

Run: `.venv/bin/python -m pytest tests/test_execution.py -q`

Expected: FAIL because the execution modules do not exist.

- [ ] **Step 5: Implement two-phase parent/worker execution**

`prepare-execution` takes `--run`, `--phase`, `--target`, `--cell-timeout`, and `--total-timeout`. It performs target resolution/risk scan, reserves the attempt in the persistent ledger, reserves the actual empty temp directory, hashes the target, and atomically writes the request. This read-only preparation is what the Skill displays before asking for confirmation.

Target resolution and analysis binding are exact:

- `diagnosis/source` reads its canonical path, target SHA-256, kernel, and immutable analysis ID from `baseline.json`; if the current file hash differs from the baseline snapshot, reject preparation and require a new diagnosis run;
- `verification/source` reads the selected current-source canonical path, target SHA-256, kernel, language, and immutable `assessment_id` from `verification-state.json`;
- `verification/challenge` always uses `<run>/challenge.ipynb`, requires its hash to equal `verification-state.json`'s challenge-target hash, uses the baseline kernel explicitly, and binds to the same immutable verification `assessment_id`;
- `verification/challenge` is rejected with `challenge_unverifiable` before reserving an attempt when `verification-state.json.challenge_verifiability.verifiable` is false;
- any mismatch returns `analysis_target_changed` and requires a new static diagnosis/verification before execution evidence can be attached.

`cancel-execution` transitions the ledger entry from `prepared` to `cancelled`, then removes the request and reserved temp directory. On each preparation, mark prepared requests older than one hour `expired`; recover `running` entries older than their total timeout plus 60 seconds as failed `orphaned_worker`; remove their temp directories but never delete ledger tombstones.

`execute` takes only `--request <path>` and `--confirmed-target-sha256 <hash>`. The parent process must:

1. validate the request and confirmed hash;
2. atomically transition the one-time ledger entry from `prepared` to `running`; reject replay or expired/cancelled requests;
3. revalidate the request's immutable analysis ID and expected target path/hash against the current baseline or verification state;
4. rehash and rescan the same target immediately before copying;
5. refuse execution if target hash, risk result, phase, target, run, or analysis binding no longer matches;
6. copy the target into the already-displayed reserved temporary directory, hash the copied bytes, and require that hash to equal the confirmed/request hash before launching a worker;
7. call `[sys.executable, "-m", "notebook_coach.execution_worker", ...]` with JSON input and that directory as its working directory;
8. launch with `subprocess.Popen(..., start_new_session=True)` and enforce the total timeout with `communicate(timeout=...)`;
9. terminate the worker process group on timeout and wait for cleanup;
10. rehash the original target before constructing the final log; if it changed, set status `target_changed_during_run`, discard worker evidence, and mark the log `evidence_eligible: false`;
11. otherwise set an accurate status and `evidence_eligible` flag, then write the sanitized execution log exactly once and transition the ledger to `completed` or `failed`;
12. remove the temporary directory while preserving the run-scoped log and ledger entry.

Evidence-eligible statuses are `completed`, `completed_with_cell_errors`, and `timeout`. `blocked`, `cancelled`, `expired`, `worker_failure`, `target_changed_before_execution`, and `target_changed_during_run` are never eligible for an execution review. Every log stores target hashes before/copy/after plus the immutable analysis ID.

Any failure after the ledger reaches `running` must transition it exactly once to `failed`, write at most one non-eligible log after the final target-hash check, and clean the temp directory. No error path may leave a `running` entry or overwrite a previous log.

The worker uses `NotebookClient(timeout=cell_timeout, allow_errors=True)` and returns structured cell status only. It must never install packages or make safety claims.

- [ ] **Step 6: Run execution tests**

Run: `.venv/bin/python -m pytest tests/test_execution.py -q`

Expected: all tests PASS, including timeout cleanup.

- [ ] **Step 7: Commit**

```bash
git add src/notebook_coach/execution.py src/notebook_coach/execution_worker.py src/notebook_coach/runs.py src/notebook_coach/cli.py tests/test_execution.py
git commit -m "feat: execute confirmed notebook targets safely"
```

## Task 10: Prepare and finalize verification

**Files:**
- Modify: `src/notebook_coach/contracts.py`
- Modify: `src/notebook_coach/render.py`
- Modify: `src/notebook_coach/workflows.py`
- Modify: `src/notebook_coach/cli.py`
- Modify: `tests/conftest.py`
- Create: `tests/test_verification.py`

- [ ] **Step 1: Write failing verification-assessment tests**

Validate the canonical verification JSON above with these exact rules:

- allowed original-issue statuses are `resolved`, `remaining`, and `regressed`;
- every baseline issue ID appears exactly once in `issue_results`—missing, duplicate, or unknown IDs are rejected;
- allowed challenge statuses are `passed`, `needs_work`, and `unverifiable`;
- `C-CODE` and `C-CONCEPT` each appear exactly once—missing, duplicate, or unknown challenge IDs are rejected;
- `unverifiable` is allowed only when the stage records missing/mismatched challenge metadata;
- new issue IDs cannot collide with baseline IDs and require dimension, severity, category, current cell indices, evidence, impact, and recommendation;
- all evidence text passes the same secret and size checks as diagnosis input.

- [ ] **Step 2: Write failing source/challenge separation tests**

```python
def test_challenge_completion_does_not_inflate_unchanged_source_score(run_fixture):
    stage = prepare_verification(run_fixture.source, run_fixture.run_dir)
    assessment = run_fixture.verification_assessment(
        source_issue_status="remaining",
        challenge_statuses=("passed", "passed"),
    )
    verification = finalize_verification(stage, assessment)
    assert verification.before_score == verification.after_score
    assert verification.challenge_results["C-CODE"] == "passed"
```

Also assert:

- an edited source can improve score while an untouched challenge remains `needs_work`;
- multiple candidate runs return candidates instead of selecting latest;
- metadata mismatch produces `unverifiable` without blocking source verification;
- current kernel name or language differing from baseline returns `environment_confirmation_required` before staging, and only proceeds with `--confirm-environment-mismatch` after the Skill displays both values;
- when a different source path is explicitly confirmed, the finalized verification state stores that selected canonical path and later `verification/source` execution resolves only that path, never the baseline path;
- the stage contains a redacted, bounded `challenge-snapshot.json` with the editable code and concept answer cells, not only hashes;
- a fixture secret in a challenge answer never appears in the snapshot, assessment, state, or Markdown.

- [ ] **Step 3: Run and observe failure**

Run: `.venv/bin/python -m pytest tests/test_verification.py -q`

Expected: FAIL because verification handlers do not exist.

- [ ] **Step 4: Implement verification staging and rendering**

Add a `run_fixture` helper to `tests/conftest.py`. `prepare-verification` accepts `--run`, optional `--confirm-source-mismatch`, and optional `--confirm-environment-mismatch`. Before creating a stage it compares normalized source identity, kernel name, and language with the baseline; unconfirmed differences return structured confirmation-required JSON.

After identity checks, `prepare_verification` writes:

- a redacted current-source snapshot;
- the complete baseline issue subset required for one-to-one status coverage;
- `challenge-snapshot.json` containing validated metadata, initial/current content hashes, and redacted/bounded editable answer cells plus safe saved outputs;
- a challenge-verifiability flag and reason;
- the expected `verification-assessment.json` path.

`finalize_verification` validates the assessment, reuses original issue IDs, validates sequential new IDs, computes score locally, and atomically writes `verification.md` revision 1 plus `.notebook-coach/verification-state.json`.

The state stores:

- `source_target`: selected canonical current-source path, SHA-256, kernel, and language;
- `challenge_target`: canonical `<run>/challenge.ipynb` path, SHA-256, and the baseline kernel that must be passed explicitly to `NotebookClient`;
- `challenge_verifiability`: immutable `{ "verifiable": true|false, "reason": null|string }` copied from staging; metadata missing/mismatch sets false and both challenge statuses remain `unverifiable` for this cycle;
- validated static verification assessment and before/after scores;
- immutable `assessment_id`, computed from schema version, run ID, both target hashes, and canonical validated assessment while excluding revision/execution-review fields;
- revision 1 and an empty execution-review list.

Future revisions render from this state rather than parsing Markdown. The selected source path and `assessment_id` never change within this static verification cycle; target edits require a new `prepare-verification`/`finalize-verification` cycle.

The Markdown must visibly separate “Source notebook changes” from “Challenge results” and contain before/after dimension scores, resolved/remaining/regressed/new issues, both challenge statuses, evidence, and exactly one next learning target.

- [ ] **Step 5: Run verification and scoring tests**

Run: `.venv/bin/python -m pytest tests/test_verification.py tests/test_scoring.py tests/test_contracts.py -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/notebook_coach/contracts.py src/notebook_coach/render.py src/notebook_coach/workflows.py src/notebook_coach/cli.py tests/conftest.py tests/test_verification.py
git commit -m "feat: verify notebook learning evidence"
```

## Task 11: Apply target-scoped execution evidence as report revisions

**Files:**
- Create: `src/notebook_coach/revisions.py`
- Modify: `src/notebook_coach/contracts.py`
- Modify: `src/notebook_coach/render.py`
- Modify: `src/notebook_coach/scoring.py`
- Modify: `src/notebook_coach/cli.py`
- Create: `tests/test_revisions.py`

- [ ] **Step 1: Write failing diagnosis revision tests**

An execution review for `diagnosis/source` has this complete shape:

```json
{
  "schema_version": "1.0",
  "run_id": "...",
  "phase": "diagnosis",
  "target": "source",
  "attempt_id": "A001",
  "execution_log": "execution-diagnosis-source-A001.json",
  "issue_updates": [
    {"issue_id": "I001", "evidence_label": "supported", "evidence": "Cell 3 reproduced the saved error."}
  ],
  "new_issues": [],
  "challenge_updates": []
}
```

Allowed evidence labels are exactly `supported`, `conflicted`, and `uncertain`. Diagnosis updates may reference only baseline issue IDs and cannot contain statuses, new issues, or challenge updates. Applying the review appends it to `report-state.json`, creates report revision 2, preserves baseline bytes and initial score, links the exact log, and never rewrites `challenge.ipynb`.

- [ ] **Step 2: Write failing verification target-boundary tests**

For `verification/source`, use the common fields above with `phase: "verification"`, `target: "source"`, empty `challenge_updates`, and:

```json
{
  "issue_updates": [
    {
      "issue_id": "I001",
      "status": "resolved",
      "evidence_label": "supported",
      "evidence": "The temporary copy ran with normalized rows."
    }
  ],
  "new_issues": []
}
```

`new_issues` may contain full validated issue objects when runtime evidence reveals a genuinely new source problem; their IDs must be the next unused `Ixxx` values. Merging source updates changes only issue status/evidence, then locally recomputes the after score from `verification-state.json`.

For `verification/challenge`, `issue_updates` and `new_issues` must be empty and `challenge_updates` must contain exactly one update for `C-CODE` with status `passed` or `needs_work`, evidence label, and evidence. Reject `C-CONCEPT` because it is judged from text, and reject any source update from a challenge log. Challenge execution never changes the source score. Assert revisions increase 1, 2, 3 without gaps when source and challenge are executed separately.

Add a negative test proving a persisted `challenge_verifiability.verifiable: false` state rejects both challenge execution preparation and any manually supplied challenge execution review, leaving both challenge statuses `unverifiable`.

- [ ] **Step 3: Run and observe failure**

Run: `.venv/bin/python -m pytest tests/test_revisions.py -q`

Expected: FAIL because revision support does not exist.

- [ ] **Step 4: Implement execution-review contracts and atomic revision updates**

`apply_execution_review(run_dir, log_path, review_path)` must derive phase/target/attempt ID from the validated log, require every common review field to match it, reject cross-run or cross-target evidence, and validate all referenced IDs and allowed status transitions. It must also require:

- `evidence_eligible: true` and a status in `completed`, `completed_with_cell_errors`, or `timeout`;
- identical request/copy/after hashes matching the target hash recorded in the corresponding analysis state;
- diagnosis log `analysis_id` equal to the current immutable `baseline.json` SHA-256;
- verification log `analysis_id` equal to the immutable `verification-state.json.assessment_id`;
- the verification source log target path equal to `verification-state.json.source_target.path`, or the challenge log path equal to `challenge_target.path`.

For `verification/challenge`, additionally require `verification-state.json.challenge_verifiability.verifiable` to be true. If false, reject the review before reading execution evidence and preserve both `unverifiable` challenge results; runtime success cannot override missing or mismatched metadata.

Explicitly reject `target_changed_before_execution`, `target_changed_during_run`, cancelled, expired, blocked, worker-failure, stale-analysis, or replayed logs before reading any worker evidence.

Diagnosis revisions render from immutable `baseline.json` plus `.notebook-coach/report-state.json`. Verification revisions merge into `.notebook-coach/verification-state.json`, recompute local scores only for source updates, increment its revision, then rerender `verification.md`. Never parse Markdown to recover state. Write new state and Markdown to temporary files; if either replacement fails, restore the previous state/Markdown and leave the revision unchanged.

The review JSON itself must be redacted and bounded before persistence. Add tests that attempt IDs are monotonic, logs never overwrite, report revisions are contiguous, injected two-file write failure rolls back, and stdout/stderr/exception summaries remain redacted and truncated before disk or model use.

- [ ] **Step 5: Run revision, execution, and render tests**

Run: `.venv/bin/python -m pytest tests/test_revisions.py tests/test_execution.py tests/test_render.py -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/notebook_coach/revisions.py src/notebook_coach/contracts.py src/notebook_coach/render.py src/notebook_coach/scoring.py src/notebook_coach/cli.py tests/test_revisions.py
git commit -m "feat: revise reports with scoped runtime evidence"
```

## Task 12: Author and test the Codex Skill

**Required skills:** Read and follow `@superpowers:writing-skills` and `@skill-creator` before changing Skill files.

**Files:**
- Create: `SKILL.md`
- Create: `agents/openai.yaml`
- Create: `scripts/install.sh`
- Create: `tests/test_skill_contract.py`
- Create: `tests/test_install_script.py`

- [ ] **Step 1: Write failing Skill contract tests**

Tests must parse frontmatter and assert:

- name is `notebook-coach`;
- description triggers on checking, diagnosing, coaching, challenging, or rechecking a Jupyter Notebook;
- instructions explicitly require a GPT-5.6 Codex session;
- default flow is static and never asks for an API key;
- execution requires showing phase, target, path, hash, kernel, temp directory, and limits before confirmation;
- source and challenge confirmations are separate;
- every confirmation is based on `prepare-execution` output and bound to its target SHA-256;
- an expired/cancelled request with an unchanged target may be prepared again, but any target-hash change forces a new static diagnosis or verification cycle, a new immutable analysis ID, and then a new confirmation;
- risk-blocked notebooks remain static-only;
- path, kernel, and language mismatches are displayed and separately confirmed before recheck staging;
- the Skill writes model assessments only to the CLI-provided staging path;
- model output follows the canonical contracts and never fabricates execution evidence;
- `/feedback` Session ID and judge artifacts are mentioned.

- [ ] **Step 2: Run and observe failure**

Run: `.venv/bin/python -m pytest tests/test_skill_contract.py -q`

Expected: FAIL because `SKILL.md` does not exist.

- [ ] **Step 3: Write `SKILL.md` using the validated workflow**

The Skill must orchestrate these states exactly:

```text
prepare-diagnosis
→ read redacted snapshot/risk JSON
→ write diagnosis-assessment.json
→ finalize-diagnosis
→ optionally prepare-execution for diagnosis/source
→ display the exact target/hash/kernel/temp directory/limits
→ on denial run cancel-execution; on approval execute the hash-bound request
→ optionally write/apply execution review

resolve-run
→ prepare-verification
→ if required, display and confirm path/kernel/language mismatch, then prepare again
→ read redacted source and redacted challenge-answer snapshots
→ write verification-assessment.json
→ finalize-verification
→ optionally prepare, display, and confirm source and/or challenge separately
→ cancel denied requests
→ if a request merely expires and the target is unchanged, prepare again
→ if any target hash changes before/during execution, return to static diagnosis/verification and create a new analysis ID before preparing again
→ optionally write/apply target-scoped execution reviews
```

Every local command must use `${SKILL_DIR}/.venv/bin/notebook-coach-tool`, where `SKILL_DIR` is the directory containing `SKILL.md`; do not rely on shell `PATH` or a globally installed package. Keep `SKILL.md` focused on decisions and orchestration; link scripts instead of embedding Python logic. Include honest language that temporary copies and timeouts are not an OS sandbox.

- [ ] **Step 4: Add display metadata and a testable installer**

`agents/openai.yaml`:

```yaml
interface:
  display_name: "Notebook Coach"
  short_description: "Diagnose and recheck learning evidence in Jupyter notebooks"
```

`scripts/install.sh` must use `set -eu`, detect Python 3.11, create `.venv`, install the local project, and link the repository to `${CODEX_HOME:-$HOME/.codex}/skills/notebook-coach`. Add `--link-only` for an offline test, treat “repository already equals destination” as a successful no-op, and refuse to replace an unrelated existing destination.

- [ ] **Step 5: Run Skill and installer tests**

Run: `bash -n scripts/install.sh`

Run: `.venv/bin/python -m pytest tests/test_skill_contract.py tests/test_install_script.py -q`

Expected: all tests PASS; install test uses a temporary `CODEX_HOME` and leaves the real Codex directory untouched.

- [ ] **Step 6: Run the available Skill validator**

Run: `python3.11 "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" .`

Expected: validation succeeds. If the validator path is unavailable, record that fact and rely on the repository contract tests; do not download an alternate validator.

- [ ] **Step 7: Commit**

```bash
git add SKILL.md agents/openai.yaml scripts/install.sh tests/test_skill_contract.py tests/test_install_script.py
git commit -m "feat: add Notebook Coach Codex Skill"
```

## Task 13: Add the safe Transformer demo and pre-generated evidence

**Files:**
- Create: `samples/transformer_attention_buggy.ipynb`
- Create: `tests/test_sample.py`
- Create: `examples/transformer_attention_buggy/baseline.json`
- Create: `examples/transformer_attention_buggy/report.md`
- Create: `examples/transformer_attention_buggy/challenge.ipynb`
- Create: `examples/transformer_attention_buggy/verification.md`
- Create: `examples/transformer_attention_buggy/.notebook-coach/report-state.json`
- Create: `examples/transformer_attention_buggy/.notebook-coach/verification-state.json`
- Create: `examples/transformer_attention_buggy/.notebook-coach/execution-ledger.json`
- Create: `examples/transformer_attention_buggy/evaluation.md`

- [ ] **Step 1: Write the failing sample acceptance test**

The test must assert the sample is valid, risk scan is not blocked, contains no secret/network/shell/file-write operations, and intentionally contains exactly these teachable signals:

1. code uses the wrong softmax axis;
2. Markdown claims each key distributes attention across queries;
3. random values are generated without a seed.

Keep the sample under 12 cells and implement the tiny score/softmax example with only `math`, `random`, and lists from the Python standard library.

- [ ] **Step 2: Run and observe failure**

Run: `.venv/bin/python -m pytest tests/test_sample.py -q`

Expected: FAIL because the sample does not exist.

- [ ] **Step 3: Create the safe sample and pass static tests**

Run: `.venv/bin/python -m pytest tests/test_sample.py tests/test_risk.py tests/test_notebooks.py -q`

Expected: all tests PASS.

- [ ] **Step 4: Create a portable, answer-free judge example**

Use the installed `$notebook-coach` Skill in a GPT-5.6 Codex session on the sample. Immediately preserve the generated, unfilled `challenge.ipynb`. Create a temporary corrected source copy, leave the run's challenge unanswered, and finalize verification with both challenge statuses `needs_work`; this produces a consistent complete run without shipping answers.

Copy the baseline, report, original unfilled challenge, verification, and all three internal state files (`report-state.json`, `verification-state.json`, and the execution ledger) into `examples/transformer_attention_buggy/`. Replace machine-specific display paths with repository-relative paths only where the schema permits, then rerun validation. Assert the checked-in challenge still contains both TODOs, no answers/solutions, and hashes consistent with the checked-in state; assert the ledger is valid even when no optional execution occurred.

- [ ] **Step 5: Evaluate completed challenges only in a disposable copy**

Duplicate the run into a temporary directory ignored by Git, complete both challenge tasks there, and rerun verification to confirm both can reach `passed`. Record the result in `evaluation.md`, then delete the completed challenge and temporary run. Never copy the filled challenge or its answer-bearing state into `examples/`.

- [ ] **Step 6: Record the human/model evaluation**

In `evaluation.md`, record:

- session date and declared model `GPT-5.6`;
- whether the code bug, misconception, and reproducibility gap were found with correct cell evidence;
- whether both challenges map to issue IDs and omit answers;
- whether verification separates source score from challenge status;
- elapsed time for the full loop;
- failures or limitations observed.

Do not put private credentials in the file. Keep the `/feedback` Session ID for the submission checklist.

- [ ] **Step 7: Validate generated artifacts and commit**

Run: `.venv/bin/notebook-coach-tool validate-run examples/transformer_attention_buggy`

Run: `.venv/bin/python -m pytest tests/test_sample.py -q`

Expected: validation and tests PASS.

```bash
git add samples tests/test_sample.py examples/transformer_attention_buggy
git commit -m "test: add Transformer coaching demo"
```

## Task 14: Write judge-facing docs and continuous tests

**Files:**
- Create: `README.md`
- Create: `docs/demo-script.md`
- Create: `docs/devpost-checklist.md`
- Create: `.github/workflows/tests.yml`
- Create: `tests/test_docs.py`

- [ ] **Step 1: Write failing documentation-contract tests**

Test that README contains install commands, `$notebook-coach` diagnosis/recheck examples, Python 3.11, macOS/Linux support, no-API-key wording plus Codex-credit caveat, GPT-5.6/Codex/Python responsibility split, privacy, redaction, non-sandbox warning, test command, sample links, and Windows/WSL unverified status. Verify all relative Markdown links exist.

- [ ] **Step 2: Run and observe failure**

Run: `.venv/bin/python -m pytest tests/test_docs.py -q`

Expected: FAIL because judge-facing docs do not exist.

- [ ] **Step 3: Write the README and submission documents**

Use English as the judge-facing primary language with a concise Chinese quick-start section. The first screen must answer: what it does, why it is educational, why no API key is needed, and how to inspect the pre-generated example.

`docs/demo-script.md` must follow the approved 0:00–3:00 timeline and include exact on-screen commands plus voiceover. `docs/devpost-checklist.md` must track public repository, README/install/platform/test instructions, under-three-minute public YouTube link, GPT-5.6 use, Codex development explanation, `/feedback` Session ID, and final fresh-checkout verification.

- [ ] **Step 4: Add CI**

Create a workflow triggered on push and pull request with a matrix for `ubuntu-latest` and `macos-latest`, Python 3.11, `pip install -e '.[test]'`, and `pytest -q`. Do not run GPT/model evaluation or unknown Notebook execution in CI.

- [ ] **Step 5: Run documentation and full local tests**

Run: `.venv/bin/python -m pytest tests/test_docs.py -q`

Run: `.venv/bin/python -m pytest -q`

Expected: documentation tests and the full suite PASS.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/demo-script.md docs/devpost-checklist.md .github/workflows/tests.yml tests/test_docs.py
git commit -m "docs: add judge guide and submission checklist"
```

## Task 15: Perform final acceptance and code review

**Required skills:** Use `@superpowers:verification-before-completion` and `@superpowers:requesting-code-review`.

**Files:**
- Modify only files required by verified failures or review findings.

- [ ] **Step 1: Verify a clean install without touching the real Skill directory**

Run in a temporary clone or clean worktree:

```bash
python3.11 -m venv .acceptance-venv
.acceptance-venv/bin/python -m pip install -e '.[test]'
.acceptance-venv/bin/python -m pytest -q
```

Expected: install succeeds and all tests PASS.

- [ ] **Step 2: Verify default static behavior and source immutability**

Run: `.venv/bin/python -m pytest tests/test_diagnosis_workflow.py::test_static_flow_preserves_source_hash_with_checked_in_assessment -v`

The named test must load `tests/fixtures/diagnosis-assessment.json`, replace only `__RUN_ID__` with the stage ID, call real prepare/finalize workflow functions, and compare the sample SHA-256 plus kernel-process list before and after. Expected: PASS, identical hashes, and no kernel process starts during static flow.

- [ ] **Step 3: Verify execution boundaries**

Run the no-confirmation, risky-code, source-only, challenge-only, timeout, log-redaction, and target-hash-change tests individually with `-v`. Expected: each reports PASS and no original Notebook changes.

- [ ] **Step 4: Verify the judge path in under five minutes**

From a fresh checkout, follow only README instructions to install or inspect pre-generated artifacts. Time diagnosis → challenge → corrected-source verification. Expected: first report and complete learning loop meet the five-minute spec on the safe sample; record observed times in `examples/transformer_attention_buggy/evaluation.md`.

- [ ] **Step 5: Request code and spec-conformance review**

Ask the reviewer to compare implementation against the approved spec, prioritizing source immutability, secret handling, no API dependency, execution confirmation, run ambiguity, target evidence isolation, score arithmetic, and judge installability. Fix blocking findings using TDD and rerun affected plus full tests.

- [ ] **Step 6: Commit verified fixes, if any**

```bash
git add <only-files-changed-for-verified-findings>
git commit -m "fix: address Notebook Coach acceptance findings"
```

If no files changed, do not create an empty commit.

- [ ] **Step 7: Rerun final verification after the final commit**

Run: `.venv/bin/python -m pytest -q`

Run: `.venv/bin/notebook-coach-tool validate-run examples/transformer_attention_buggy`

Run: `git diff --check`

Run: `git status --short`

Expected: tests PASS, example validation succeeds, no whitespace errors, and the worktree is clean after any final commit.

## Task 16: Produce and publish the real Devpost submission assets

**Authority gate:** Creating a public repository, uploading a public video, and submitting the Devpost form are external actions. Ask the user for explicit approval immediately before each action. If approval is not granted, prepare all local assets, record the exact blocker, and do not claim the submission is complete.

**Required skills when applicable:** Use `@github:yeet` for authorized GitHub publication. If ChatCut is used for the demo, use `@chatcut:chatcut-plugin-basics`, `@chatcut:export`, and `@chatcut:verification`; use `@chatcut:voice` only if the user explicitly chooses generated narration.

**Files:**
- Create locally but do not commit: `artifacts/notebook-coach-demo.mp4`
- Create locally but do not commit: `artifacts/devpost-submission.private.md`
- Modify: `docs/devpost-checklist.md`
- Modify if needed: `README.md`

- [ ] **Step 1: Capture the real Codex feedback evidence**

Run `/feedback` in the GPT-5.6 Codex development session. Write the real Session ID to ignored `artifacts/devpost-submission.private.md`, together with the model name and date. In the public checklist record only “Session ID captured privately: yes”; do not expose credentials or unrelated conversation data.

- [ ] **Step 2: Record and verify the narrated demo**

Ask the user whether narration will be their recorded voice or explicitly authorized generated narration. Produce `artifacts/notebook-coach-demo.mp4` from the approved script, showing real install/run commands, report evidence, both challenges, source correction/recheck, GPT-5.6/Codex/Python roles, no-API-key caveat, and source immutability.

Run: `ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 artifacts/notebook-coach-demo.mp4`

Expected: duration is greater than 0 and strictly below 180 seconds. If `ffprobe` is unavailable, use the selected video verification skill to obtain and record equivalent duration evidence. Review the exported video for audible narration, readable text, no private paths/secrets, and no misleading safety claim.

- [ ] **Step 3: Publish the repository after explicit approval**

Ask the user to approve creation or use of a public GitHub repository and pushing the verified branch. After approval, publish it, confirm anonymous/public read access, and record the real repository URL in `docs/devpost-checklist.md`. Update README links if the final URL requires it, commit that documentation change, push it, and verify the public default branch contains the example artifacts and test instructions.

- [ ] **Step 4: Upload the video after explicit approval**

Ask the user to approve a public YouTube upload. After approval, upload the verified MP4 with an accurate title/description, confirm the final visibility is Public, open the URL logged out or in an anonymous context, and record the real URL plus measured duration in `docs/devpost-checklist.md` and the private submission note.

- [ ] **Step 5: Eliminate placeholders and verify submission facts**

Run: `! rg -n 'TBD|TODO-URL|YOUR_|<[^>]*(URL|ID)[^>]*>' README.md docs/devpost-checklist.md artifacts/devpost-submission.private.md`

Expected: the inverted command exits 0 because `rg` finds no placeholders. Confirm the public repository URL, public YouTube URL, real Session ID, GPT-5.6 declaration, platform/install/test instructions, and under-three-minute duration are all present in the correct public or private location.

- [ ] **Step 6: Complete and submit Devpost only after final approval**

Populate the existing Devpost draft with the real project name, Education track, summary, public repository, public video, technologies, install/platform/test instructions, Codex/GPT-5.6 explanation, and `/feedback` Session ID. Show the user the final field summary and ask for explicit approval before pressing the final Submit button. After approval, submit and verify the project page reports a submitted state rather than a draft.

- [ ] **Step 7: Commit the public checklist update and verify clean state**

Do not add `artifacts/`. Commit only public documentation updates, push after approval, and rerun:

```bash
git add README.md docs/devpost-checklist.md
git commit -m "docs: record Notebook Coach submission links"
git push
.venv/bin/python -m pytest tests/test_docs.py -q
git diff --check
git status --short
```

Expected: no whitespace errors; worktree clean; private submission note and video remain ignored; public URLs resolve; Devpost state is submitted.

## Definition of done

- All local deterministic tests pass on Python 3.11, including timeout and secret-leak regressions.
- `SKILL.md` passes repository contract tests and the available Codex Skill validator.
- Static diagnosis and verification never launch a kernel by default.
- Every execution target is separately scanned, described, confirmed, copied, timed out, logged, and hash-checked.
- No model/API dependency exists in Python package metadata or runtime code.
- The checked-in sample and pre-generated artifacts demonstrate the full Education-track story.
- README, demo script, Devpost checklist, platform notes, and `/feedback` Session ID evidence are ready.
- A fresh checkout reaches the first report in under five minutes and leaves the source Notebook unchanged.
- After explicit user approvals, the public repository and narrated YouTube video resolve anonymously, the real `/feedback` Session ID is recorded privately, and Devpost shows the project as submitted.
