# Notebook Coach model contracts

Write UTF-8 JSON only. Keep every key shown, add no extra keys, use the actual
`run_id`, and reference only cell indices present in the supplied snapshot.
Never copy a credential or unredacted private value into an assessment.

## Diagnosis assessment

```json
{
  "schema_version": "1.0",
  "run_id": "ACTUAL_RUN_ID",
  "notebook_summary": "Non-empty static summary.",
  "concept_map": ["concept"],
  "issues": [
    {
      "issue_id": "I001",
      "dimension": "correctness",
      "severity": "major",
      "category": "code",
      "cell_indices": [0],
      "evidence": "Static evidence from the visible cell.",
      "impact": "Why this affects the learning artifact.",
      "recommendation": "A bounded next step, without doing the challenge."
    }
  ],
  "challenges": [
    {
      "challenge_id": "C-CODE",
      "kind": "code",
      "source_issue_ids": ["I001"],
      "title": "Code challenge title",
      "prompt": "Answer-free code task.",
      "acceptance_criteria": ["Observable criterion"]
    },
    {
      "challenge_id": "C-CONCEPT",
      "kind": "concept",
      "source_issue_ids": ["I001"],
      "title": "Concept challenge title",
      "prompt": "Answer-free explanation task.",
      "acceptance_criteria": ["Observable criterion"]
    }
  ]
}
```

Rules:

- `issue_id` values are unique `I000`-format IDs.
- `dimension` is one of `correctness`, `concept_completeness`,
  `reproducibility`, or `clarity`.
- `severity` is one of `blocking`, `major`, or `minor`.
- `cell_indices` are sorted, unique, non-empty, and limited to cells in the
  redacted snapshot.
- The two challenges occur exactly once and in `C-CODE`, `C-CONCEPT` order.
  Each references at least one declared issue.

## Verification assessment

```json
{
  "schema_version": "1.0",
  "run_id": "ACTUAL_RUN_ID",
  "issue_results": [
    {
      "issue_id": "I001",
      "status": "remaining",
      "current_cell_indices": [0],
      "evidence": "Static evidence from the current source snapshot."
    }
  ],
  "new_issues": [],
  "challenge_results": [
    {
      "challenge_id": "C-CODE",
      "status": "needs_work",
      "evidence": "Evidence from the redacted challenge snapshot."
    },
    {
      "challenge_id": "C-CONCEPT",
      "status": "needs_work",
      "evidence": "Evidence from the redacted challenge snapshot."
    }
  ],
  "next_learning_target": "Exactly one focused next learning target."
}
```

Rules:

- Cover every baseline issue exactly once. Issue status is `resolved`,
  `remaining`, or `regressed`; current cell indices may be empty.
- A new issue uses the complete diagnosis issue shape. Its ID starts after the
  highest baseline issue ID and remains sequential.
- Challenge results occur in `C-CODE`, `C-CONCEPT` order. Status is `passed`,
  `needs_work`, or `unverifiable`.
- When challenge metadata is unverifiable, both challenge statuses must be
  `unverifiable`; otherwise neither may be `unverifiable`.

## Execution review

Use the path returned as `review_path`. The binding fields must match the
eligible execution log exactly.

```json
{
  "schema_version": "1.0",
  "run_id": "ACTUAL_RUN_ID",
  "phase": "diagnosis",
  "target": "source",
  "attempt_id": "A001",
  "execution_log": "execution-diagnosis-source-A001.json",
  "issue_updates": [],
  "new_issues": [],
  "challenge_updates": []
}
```

Use only the update shape allowed for the exact phase and target:

- Diagnosis/source `issue_updates`:
  `{"issue_id":"I001","evidence_label":"supported","evidence":"..."}`.
  Keep `new_issues` and `challenge_updates` empty.
- Verification/source `issue_updates`:
  `{"issue_id":"I001","status":"resolved","evidence_label":"supported","evidence":"..."}`.
  `new_issues` may contain complete diagnosis issue objects;
  `challenge_updates` stays empty.
- Verification/challenge requires exactly one `challenge_updates` item for
  `C-CODE`:
  `{"challenge_id":"C-CODE","status":"passed","evidence_label":"supported","evidence":"..."}`.
  Keep `issue_updates` and `new_issues` empty.

`evidence_label` is `supported`, `conflicted`, or `uncertain`. Runtime evidence
must describe only what the eligible temporary-copy execution log shows.
