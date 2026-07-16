# Notebook Coach

Notebook Coach turns a Jupyter Notebook into a short learning loop: evidence-linked diagnosis, two answer-free challenges, and a recheck that separates source-code improvement from challenge completion. It is educational because every substantive finding points to a cell and every challenge maps back to an issue ID.

No OpenAI API key is needed. GPT-5.6 Codex performs the teaching judgment in the current Codex session, while the local Python 3.11 tool deterministically parses, redacts, hashes, scores, renders, and optionally executes confirmed temporary copies. The session may still use existing ChatGPT/Codex credits or event allowance.

Inspect the complete pre-generated example immediately:

- [Buggy Transformer attention Notebook](samples/transformer_attention_buggy.ipynb)
- [Evidence-linked report](examples/transformer_attention_buggy/report.md)
- [Answer-free challenge Notebook](examples/transformer_attention_buggy/challenge.ipynb)
- [Verification report](examples/transformer_attention_buggy/verification.md)
- [Evaluation notes](examples/transformer_attention_buggy/evaluation.md)

## Demo

Watch the public 1:58 walkthrough: [Notebook Coach — Evidence-Linked Jupyter Learning | OpenAI Build Week](https://youtu.be/AkKbNcasbzM).

The demo covers installation, evidence-linked diagnosis, answer-free challenge generation, and measurable recheck without requiring an OpenAI API key.

## Install

Requirements: Python 3.11, macOS or Linux, and Codex CLI/Desktop. Native Windows and WSL are currently unverified.

```bash
git clone https://github.com/yukun181013/notebook-coach.git notebook-coach
cd notebook-coach
bash scripts/install.sh
```

The installer creates `.venv`, installs the local package, and links `$notebook-coach` into `${CODEX_HOME:-$HOME/.codex}/skills/notebook-coach` without replacing an unrelated Skill.

## Use with Codex

In a GPT-5.6 Codex session:

```text
$notebook-coach diagnose samples/transformer_attention_buggy.ipynb
$notebook-coach recheck samples/transformer_attention_buggy.ipynb --run notebook-coach-output/<run>
```

The Skill calls its local CLI through its own `.venv`. Static diagnosis and recheck never launch a kernel. Optional execution is a separate two-phase action: prepare one exact target, display its path/SHA-256/kernel/temp directory/time limits, then execute only after explicit hash-bound confirmation.

## Responsibility split

- GPT-5.6 Codex: concept diagnosis, evidence-based explanation, challenge design, and open-ended recheck judgment.
- Local Python: Notebook validation, privacy redaction, bounded snapshots, risk scan, hashes, immutable run state, score arithmetic, reports, and confirmed execution mechanics.
- Learner: source edits, challenge answers, and every optional execution decision.

## Privacy and safety

Model-facing source, saved outputs, challenge answers, and runtime summaries pass through bounded redaction. The source Notebook is never rewritten. Static mode is the default. Optional execution uses a separately scanned, hash-checked temporary copy with per-cell and total timeouts; this is not an OS sandbox. Do not execute untrusted code merely because the scanner found no known pattern.

## Develop and test

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

See the [three-minute demo script](docs/demo-script.md), [Devpost checklist](docs/devpost-checklist.md), and [design specification](docs/superpowers/specs/2026-07-15-notebook-coach-design.md).

## 中文快速开始

需要 Python 3.11、macOS 或 Linux。运行 `bash scripts/install.sh` 后，在 GPT-5.6 Codex 会话中输入 `$notebook-coach diagnose <文件.ipynb>`。默认只做静态分析，不需要 API Key；可选执行必须先展示目标与哈希，并由你单独确认。
