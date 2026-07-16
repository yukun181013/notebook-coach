# Contributor Guide for Coding Agents

## Project overview

Notebook Coach is a Codex Skill backed by a deterministic Python 3.11 toolchain.
Codex provides the teaching judgment; the local package validates notebooks,
redacts sensitive content, scans risk, manages immutable runs, validates JSON
contracts, renders reports, scores evidence, and optionally executes a
user-confirmed temporary copy.

Keep that responsibility split intact. Do not move open-ended teaching judgment
into the Python package, and do not make the Skill responsible for deterministic
validation that belongs in Python.

## Repository map

- `SKILL.md`: user-facing workflow and the authoritative agent behavior.
- `references/contracts.md`: canonical model-authored JSON shapes and rules.
- `agents/openai.yaml`: Codex display metadata.
- `src/notebook_coach/cli.py`: machine-readable CLI and stable error mapping.
- `src/notebook_coach/notebooks.py`: notebook validation and bounded snapshots.
- `src/notebook_coach/sanitize.py`: redaction and text bounding.
- `src/notebook_coach/risk.py`: static execution-risk analysis.
- `src/notebook_coach/contracts.py`: exact diagnosis and verification validation.
- `src/notebook_coach/runs.py`: staging, immutable run state, and run resolution.
- `src/notebook_coach/workflows.py`: diagnosis and verification orchestration.
- `src/notebook_coach/execution.py` and `execution_worker.py`: hash-bound,
  temporary-copy execution.
- `src/notebook_coach/revisions.py`: target-scoped execution-evidence updates.
- `src/notebook_coach/render.py` and `scoring.py`: deterministic artifacts and
  Learning Evidence Score arithmetic.
- `tests/`: pytest coverage organized by the corresponding production module.
- `samples/`: input notebooks; `examples/`: checked-in, pre-generated outputs.
- `docs/superpowers/specs/2026-07-15-notebook-coach-design.md`: original design
  decisions and safety rationale.

## Setup and validation

Create an isolated development environment from the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

Run the smallest relevant test file while iterating, then the complete suite:

```bash
.venv/bin/python -m pytest tests/test_<area>.py -q
.venv/bin/python -m pytest -q
```

Use `bash -n scripts/install.sh` for installer-only changes. CI runs the full
pytest suite on Python 3.11 on both Ubuntu and macOS. There is no separate
formatter or linter configuration; follow the existing Python style, type hints,
and import conventions.

## Non-negotiable safety boundaries

- Diagnosis and recheck are static by default and must not launch a kernel.
- Never modify the learner's source notebook or install its dependencies.
- Never describe temporary-copy execution and timeouts as an OS sandbox.
- Execution requires a prepared request plus explicit confirmation bound to the
  displayed target SHA-256. Source and challenge targets require separate
  confirmations. A changed hash invalidates the request.
- Keep risk-blocked notebooks static-only.
- Redact and bound all model-facing source, saved outputs, challenge answers,
  and runtime summaries. Never persist or echo a detected secret.
- Never fabricate runtime evidence. Use execution language only when an eligible
  log exists, and keep static findings clearly labeled as static evidence.
- Preserve target isolation: source execution may update source findings only;
  challenge execution may update the code challenge only.
- Keep `baseline.json` immutable. Preserve atomic publication, append-only
  attempt numbering, and the run ledger's state-transition checks.

Changes that weaken any boundary above require an explicit product decision,
corresponding documentation updates, and focused regression tests.

## Contracts and compatibility

- Treat `references/contracts.md` and the validators in `contracts.py` as one
  interface. Model-authored JSON uses exact keys; do not silently accept extras.
- Keep CLI output machine-readable, error codes stable, and failure messages
  non-sensitive.
- Preserve deterministic serialization, hashing, scoring, and artifact naming.
- If a schema, command, artifact, or Skill instruction changes, update every
  affected surface together: implementation, `SKILL.md`, contract reference,
  package/version constants when applicable, docs/examples, and tests.
- Do not guess the newest run when resolution is ambiguous. Preserve explicit
  source, path, kernel, and language mismatch confirmations.
- Keep the Learning Evidence Score framed as evidence from one notebook, never
  as a grade, ability measure, or certification.

## Change and test expectations

- Add a regression test before or alongside every bug fix.
- Place tests with the closest existing module-level test file and reuse fixtures
  from `tests/conftest.py`.
- Test success paths and boundary failures, especially malformed notebooks,
  redaction, exact contract keys, path traversal, hash mismatches, timeouts,
  ledger transitions, and atomic-write failures.
- Avoid real network access, real credentials, and writes to the user's actual
  Codex home in tests. Installer tests must use a temporary `CODEX_HOME`.
- Keep sample notebooks and checked-in example artifacts synchronized when a
  user-visible format or workflow change intentionally affects them.
- When editing Markdown links, Skill metadata, CLI commands, or packaging,
  include the relevant docs, Skill-contract, CLI, and package tests.

## Scope and repository hygiene

- Prefer focused changes over unrelated refactors, especially in the large risk,
  execution, sanitization, and workflow modules.
- Do not commit `.venv`, caches, build outputs, local logs, credentials, or
  `notebook-coach-output/`.
- Before handing off a change, inspect the diff, run the relevant targeted tests,
  run the full suite, and report any validation that could not be completed.
