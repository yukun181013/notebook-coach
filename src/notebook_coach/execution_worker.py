"""Subprocess worker for executing a prepared notebook copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellTimeoutError


def _cell_result(index: int, cell: Any) -> dict[str, Any]:
    stdout: list[str] = []
    stderr: list[str] = []
    errors: list[dict[str, Any]] = []
    for output in cell.get("outputs", []):
        output_type = output.get("output_type")
        if output_type == "stream":
            destination = stderr if output.get("name") == "stderr" else stdout
            destination.append(str(output.get("text", "")))
        elif output_type == "error":
            errors.append(
                {
                    "name": str(output.get("ename", "Error")),
                    "value": str(output.get("evalue", "")),
                    "traceback": "\n".join(output.get("traceback", [])),
                }
            )
    return {
        "cell_index": index,
        "status": "error" if errors else "completed",
        "stdout": "".join(stdout),
        "stderr": "".join(stderr),
        "errors": errors,
    }


def _timeout_cell_index(notebook: Any) -> int | None:
    executed = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code" and cell.get("execution_count") is not None
    ]
    if executed:
        return executed[-1]
    return next(
        (
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "code"
        ),
        None,
    )


def run(input_path: Path, output_path: Path, kernel: str, cell_timeout: int) -> None:
    notebook = nbformat.read(input_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=cell_timeout,
        allow_errors=True,
        kernel_name=kernel,
    )
    timed_out = False
    timeout_summary: str | None = None
    timeout_cell_index: int | None = None
    try:
        client.execute()
    except CellTimeoutError as error:
        timed_out = True
        timeout_summary = str(error)
        timeout_cell_index = _timeout_cell_index(notebook)
    cells = [
        _cell_result(index, cell)
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
    ]
    if timed_out:
        for cell in cells:
            if cell["cell_index"] == timeout_cell_index:
                cell["status"] = "timeout"
                break
    result = {
        "cells": cells,
        "has_cell_errors": any(cell["status"] == "error" for cell in cells),
        "timed_out": timed_out,
        "timeout_cell_index": timeout_cell_index,
        "timeout_summary": timeout_summary,
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--cell-timeout", required=True, type=int)
    args = parser.parse_args(argv)
    run(args.input, args.output, args.kernel, args.cell_timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
