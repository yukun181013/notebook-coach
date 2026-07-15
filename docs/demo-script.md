# Notebook Coach — 3-minute demo script

## 0:00–0:20 — Problem and promise

On screen: open `samples/transformer_attention_buggy.ipynb` and show the wrong normalization plus the confident Markdown claim.

Voiceover: “A Notebook can run and still teach the wrong mental model. Notebook Coach turns cell-level evidence into diagnosis, two targeted challenges, and a measurable recheck—without an API key.”

## 0:20–0:45 — Install and static preparation

On screen:

```bash
bash scripts/install.sh
.venv/bin/notebook-coach-tool prepare-diagnosis samples/transformer_attention_buggy.ipynb
```

Voiceover: “Python validates, redacts, hashes, and risk-scans the Notebook. GPT-5.6 Codex reads only this bounded static snapshot. No kernel starts.”

## 0:45–1:20 — Evidence-linked report

On screen: write the canonical assessment to the returned path, run `finalize-diagnosis --stage <stage>`, then scroll through `report.md`.

Voiceover: “The report catches the wrong softmax axis, the query/key misconception, and the missing random seed. Each issue cites a cell and contributes to deterministic local score arithmetic.”

## 1:20–1:45 — Two answer-free challenges

On screen: open `challenge.ipynb`, highlighting `C-CODE`, `C-CONCEPT`, their issue IDs, and TODOs.

Voiceover: “The learner gets one code task and one concept task. The repository ships prompts and acceptance criteria, never solutions.”

## 1:45–2:30 — Correct and recheck

On screen:

```bash
.venv/bin/notebook-coach-tool prepare-verification corrected.ipynb --run <run> --confirm-source-mismatch
.venv/bin/notebook-coach-tool finalize-verification --stage <verification-stage>
```

Show “Source notebook changes” separately from “Challenge results.”

Voiceover: “The corrected source improves the Learning Evidence Score. The untouched challenges remain needs-work, so challenge completion cannot inflate source evidence.”

## 2:30–2:50 — Optional execution boundary

On screen: `prepare-execution` JSON with phase, target, path, SHA-256, kernel, temporary directory, and timeouts. Do not execute during the short demo.

Voiceover: “Execution is optional and target-specific. Source and challenge require separate hash-bound confirmations. Temporary copies and timeouts reduce risk but are not an operating-system sandbox.”

## 2:50–3:00 — Close

On screen: pre-generated example, passing test command, and Devpost Education track label.

Voiceover: “Notebook Coach makes learning evidence visible, reproducible, and inspectable in the Codex workflow students already use.”
