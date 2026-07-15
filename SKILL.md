---
name: notebook-coach
description: Use when checking, diagnosing, coaching, challenging, or rechecking a Jupyter Notebook for evidence-based learning feedback.
---

# Notebook Coach

Use this Skill only in a GPT-5.6 Codex session. Turn a Python Jupyter Notebook into a static, evidence-linked coaching loop without modifying the learner's source file.

## Non-negotiable boundaries

- Keep the default flow static. The default flow does not execute Notebook code.
- Never ask for an API key. The local Python tool has no model or API dependency; the current Codex session may still consume the user's existing ChatGPT/Codex allowance or event credits.
- Never install packages for a Notebook or modify the source Notebook.
- Treat temporary copies and timeouts as risk reduction, not an OS sandbox.
- Never fabricate execution evidence. Mention runtime behavior only when an eligible execution log exists.
- Write model JSON only to the CLI-provided assessment path. Do not invent a different path.

Set `SKILL_DIR` to the directory containing this file. Run every local operation through `${SKILL_DIR}/.venv/bin/notebook-coach-tool`; do not rely on `PATH` or a global package.

## Diagnose

1. Run:

   ```bash
   ${SKILL_DIR}/.venv/bin/notebook-coach-tool prepare-diagnosis NOTEBOOK --output-root notebook-coach-output
   ```

   Add `--cells 1-20,25` only when the full Notebook exceeds local limits.

2. Read only the returned redacted `snapshot.json` and `risk.json`. Base each finding on a concrete cell. Do not infer unseen outputs or learner intent.

3. Produce the canonical JSON contract with a summary, concept map, evidence-linked issues, `C-CODE`, and `C-CONCEPT`. Write it exactly to the CLI-provided assessment path named `diagnosis-assessment.json`.

4. Run `finalize-diagnosis --stage STAGE` and show the resulting report and unfilled challenge Notebook.

5. Keep a risk-blocked Notebook static-only. Do not offer execution when `risk.blocked` is true.

## Optional execution

Execution always has two phases: prepare, then explicit user confirmation.

1. Run `prepare-execution` for exactly one phase/target pair.
2. Before asking, display every returned field below verbatim:

   - phase
   - target
   - canonical path
   - target SHA-256
   - kernel
   - temporary directory
   - cell timeout
   - total timeout

3. Ask for one explicit confirmation bound to that displayed target SHA-256.
4. Confirm source and challenge separately. Never combine source and challenge confirmation.
5. On approval, run:

   ```bash
   ${SKILL_DIR}/.venv/bin/notebook-coach-tool execute \
     --request REQUEST \
     --confirmed-target-sha256 DISPLAYED_SHA256
   ```

6. On denial, run `cancel-execution --request REQUEST`.

An expired or cancelled request may be prepared as a new request when the target is unchanged. If the target hash changes before or during execution, do not reuse the request: return to a new static diagnosis or verification, obtain a new immutable analysis ID, then prepare and confirm again.

If eligible runtime evidence materially changes confidence, create the canonical execution-review JSON and use `apply-execution-review`. Keep diagnosis updates, verification source updates, and verification challenge updates within their target boundaries.

## Recheck

1. Resolve the intended run. Never guess the newest run when multiple candidates exist.
2. Run `prepare-verification SOURCE --run RUN`.
3. If the CLI reports a path mismatch, show both paths and request a separate confirmation before adding `--confirm-source-mismatch`.
4. If it reports a kernel mismatch or language mismatch, show baseline/current kernel and language and request a separate confirmation before adding `--confirm-environment-mismatch`.
5. Read the redacted current-source snapshot, baseline issue subset, and redacted challenge-answer snapshot.
6. Produce the canonical JSON contract covering every baseline issue exactly once, both challenge IDs exactly once, any sequential new issue IDs, and exactly one next learning target.
7. When challenge metadata is unverifiable, keep both challenge statuses `unverifiable`; source verification may still proceed.
8. Write JSON exactly to the CLI-provided assessment path named `verification-assessment.json`.
9. Run `finalize-verification --stage STAGE` and present source-score changes separately from challenge results.
10. Prepare and confirm optional verification/source and verification/challenge execution separately, using the same hash-bound process above.

## Evidence language

- Say “static evidence indicates” for source and saved-output findings.
- Say “temporary-copy execution showed” only when linked to an eligible log.
- Call the score a Learning Evidence Score, not a grade, ability measure, or certification.
- Keep challenges answer-free: prompts and acceptance criteria are allowed; solutions are not.

## Judge handoff

Before submission, retain the `/feedback` Session ID privately, record that GPT-5.6 Codex was used, and verify the judge artifacts: public repository, README, pre-generated example, tests, narrated video under three minutes, and Devpost field summary. Never publish private conversation data or credentials.
