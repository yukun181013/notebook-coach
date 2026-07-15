from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import nbformat
import pytest
from nbformat import NotebookNode


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
