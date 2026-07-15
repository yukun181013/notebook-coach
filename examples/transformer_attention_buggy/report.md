# Notebook Coach Report

Run ID: `20260715T160241Z-c4c6bced`
Revision: 1
Analysis mode: static

## Notebook Overview

A tiny standard-library attention walkthrough contains an axis bug, a query/key misconception, and a reproducibility gap.

## Concept Map

- scaled dot-product attention
- softmax normalization over keys
- reproducible random examples

## Key Issues and Cell Evidence

### I001 — code

- Severity: `major`
- Dimension: `correctness`
- Evidence: Cell 4 — Cell 4 transposes scores and normalizes each key column across queries.
- Impact: Each query row does not form its own distribution over keys.
- Recommendation: Apply softmax independently to each row in scores.
- Rubric deduction: 5 points

### I002 — explanation

- Severity: `major`
- Dimension: `concept_completeness`
- Evidence: Cell 5 — Cell 5 says each key distributes attention across queries.
- Impact: The explanation reverses which axis represents one attention distribution.
- Recommendation: Explain that each query distributes weight over the available keys.
- Rubric deduction: 5 points

### I003 — practice

- Severity: `minor`
- Dimension: `reproducibility`
- Evidence: Cell 2 — Cell 2 generates random queries and keys without setting a seed.
- Impact: The teaching example changes on every fresh run.
- Recommendation: Set a fixed random seed before generating values.
- Rubric deduction: 2 points

## Learning Evidence Score

- correctness: 25
- concept_completeness: 25
- reproducibility: 18
- clarity: 20
- Total: **88/100**

## Recommended Challenges

### C-CODE — Repair and reproduce attention weights

Complete the TODO so each query row is normalized over keys and repeated runs are reproducible.

Acceptance criteria: Every query row sums to one; A fixed random seed is set before generating inputs

### C-CONCEPT — Explain the attention axis

Explain which items form one softmax distribution and why.

Acceptance criteria: States that each query distributes attention over keys; Connects the softmax axis to rows of the score matrix

## Optional Execution Results and Limits

No notebook code was executed. Findings use source and saved-output evidence only.
