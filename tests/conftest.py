from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import nbformat
import pytest
from nbformat import NotebookNode

from notebook_coach import SCHEMA_VERSION
from notebook_coach.risk import build_source_risk_metadata
from notebook_coach.sanitize import redact_text, summarize_text


@pytest.fixture
def notebook_factory(
    tmp_path: Path,
) -> Callable[..., Path]:
    """Write a valid v4 notebook with caller-controlled cells and saved data."""

    created = 0

    def make_notebook(
        *,
        code: str | None = None,
        markdown: str | None = None,
        source: str | None = None,
        outputs: Sequence[NotebookNode] | None = None,
        attachments: dict | None = None,
        metadata: dict | None = None,
        cell_metadata: dict | None = None,
        cells: Sequence[NotebookNode] | None = None,
    ) -> Path:
        nonlocal created
        created += 1

        if cells is None:
            cell_source = source
            if cell_source is None:
                cell_source = markdown if markdown is not None else (code or "")

            if markdown is not None:
                cell = nbformat.v4.new_markdown_cell(
                    source=cell_source,
                    metadata=cell_metadata or {},
                )
                if attachments is not None:
                    cell["attachments"] = attachments
            else:
                cell = nbformat.v4.new_code_cell(
                    source=cell_source,
                    outputs=list(outputs or ()),
                    metadata=cell_metadata or {},
                )
            notebook_cells = [cell]
        else:
            notebook_cells = list(cells)

        notebook_metadata = metadata
        if notebook_metadata is None:
            notebook_metadata = {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python"},
            }

        notebook = nbformat.v4.new_notebook(
            cells=notebook_cells,
            metadata=notebook_metadata,
        )
        path = tmp_path / f"notebook-{created}.ipynb"
        nbformat.write(notebook, path)
        return path

    return make_notebook


@pytest.fixture
def snapshot_factory() -> Callable[[list[str]], dict]:
    """Build deterministic Task 3-shaped snapshots without writing notebooks."""

    def make_snapshot(sources: list[str]) -> dict:
        labels: set[str] = set()
        redacted_fields = 0
        cells = []
        risk_metadata = build_source_risk_metadata(sources)

        for index, (source, source_risk) in enumerate(
            zip(sources, risk_metadata, strict=True)
        ):
            _, source_labels = redact_text(source)
            if source_labels:
                labels.update(source_labels)
                redacted_fields += 1
            cells.append(
                {
                    "index": index,
                    "cell_type": "code",
                    "source": summarize_text(source, max_chars=150_000),
                    "risk": source_risk,
                    "execution_count": None,
                    "outputs": [],
                }
            )

        return {
            "schema_version": SCHEMA_VERSION,
            "source": {
                "sha256": "0" * 64,
                "cell_count": len(cells),
                "kernel_name": "python3",
                "language": "python",
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            "cells": cells,
            "sanitization": {
                "labels": sorted(labels),
                "redacted_fields": redacted_fields,
                "truncated_fields": 0,
                "omitted_binary_fields": 0,
            },
        }

    return make_snapshot


@pytest.fixture
def baseline_factory() -> Callable[..., Path]:
    """Create a minimal finalized run for resolution tests."""

    def make_baseline(
        store,
        *,
        source_path: str | Path,
        run_id: str,
    ) -> Path:
        canonical = str(Path(source_path).expanduser().resolve(strict=False))
        source_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
        source_name = Path(canonical).stem
        run_dir = store.output_root / f"{source_name}-{source_id}" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "baseline.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "source": {"path": canonical, "sha256": "a" * 64},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return run_dir

    return make_baseline


@pytest.fixture
def valid_diagnosis() -> dict:
    fixture_path = Path(__file__).parent / "fixtures/diagnosis-assessment.json"
    value = json.loads(fixture_path.read_text("utf-8"))
    value["run_id"] = "20260715T080000Z-aaaaaaaa"
    return value


@pytest.fixture
def valid_verification() -> dict:
    fixture_path = Path(__file__).parent / "fixtures/verification-assessment.json"
    value = json.loads(fixture_path.read_text("utf-8"))
    value["run_id"] = "20260715T080000Z-aaaaaaaa"
    return value


@pytest.fixture
def snapshot(snapshot_factory) -> dict:
    return snapshot_factory(
        [
            "import numpy as np\nscores = np.array([[1.0, 2.0]])",
            "weights = scores.softmax(axis=0)",
        ]
    )
