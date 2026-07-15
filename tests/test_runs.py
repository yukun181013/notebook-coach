from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

from notebook_coach.runs import (
    AmbiguousRunError,
    RunStore,
    SourceMismatchError,
)


FIXED_TIME = datetime(2026, 7, 15, 8, tzinfo=timezone.utc)


def test_run_ids_and_paths_are_deterministic(tmp_path: Path):
    store = RunStore(tmp_path, now=lambda: FIXED_TIME)

    stage = store.create_stage("/course/lesson.ipynb", "a" * 64)

    assert stage.run_id == "20260715T080000Z-aaaaaaaa"
    assert stage.final_dir.parent.name.startswith("lesson-")
    assert stage.stage_dir.parent == tmp_path / ".staging"
    assert stage.stage_dir.is_dir()


def test_same_second_collisions_use_monotonic_suffixes(tmp_path: Path):
    store = RunStore(tmp_path, now=lambda: FIXED_TIME)

    first = store.create_stage("/course/lesson.ipynb", "a" * 64)
    second = store.create_stage("/course/lesson.ipynb", "a" * 64)
    third = store.create_stage("/course/lesson.ipynb", "a" * 64)

    assert [first.run_id, second.run_id, third.run_id] == [
        "20260715T080000Z-aaaaaaaa",
        "20260715T080000Z-aaaaaaaa-02",
        "20260715T080000Z-aaaaaaaa-03",
    ]


def test_finalize_is_atomic_and_never_overwrites_baseline(tmp_path: Path):
    store = RunStore(tmp_path, now=lambda: FIXED_TIME)
    stage = store.create_stage("/course/lesson.ipynb", "a" * 64)
    baseline = b'{"run_id":"one"}\n'

    run_dir = store.finalize_stage(
        stage,
        {
            "baseline.json": baseline,
            "report.md": b"# Report\n",
            ".notebook-coach/report-state.json": b"{}\n",
        },
    )

    assert (run_dir / "baseline.json").read_bytes() == baseline
    assert not stage.stage_dir.exists()

    replacement_stage = store.create_stage("/course/lesson.ipynb", "a" * 64)
    replacement_stage = replacement_stage.__class__(
        run_id=stage.run_id,
        source_path=stage.source_path,
        source_sha256=stage.source_sha256,
        stage_dir=replacement_stage.stage_dir,
        final_dir=run_dir,
    )
    with pytest.raises(FileExistsError):
        store.finalize_stage(
            replacement_stage,
            {"baseline.json": b'{"run_id":"replacement"}\n'},
        )

    assert (run_dir / "baseline.json").read_bytes() == baseline
    assert replacement_stage.stage_dir.exists()


def test_multiple_matching_runs_never_choose_latest(
    tmp_path: Path, baseline_factory
):
    store = RunStore(tmp_path)
    baseline_factory(store, source_path="/course/lesson.ipynb", run_id="r1")
    baseline_factory(store, source_path="/course/lesson.ipynb", run_id="r2")

    with pytest.raises(AmbiguousRunError) as exc:
        store.resolve("/course/lesson.ipynb")

    assert exc.value.candidates == ["r1", "r2"]


def test_explicit_run_requires_source_mismatch_confirmation(
    tmp_path: Path, baseline_factory
):
    store = RunStore(tmp_path)
    run_dir = baseline_factory(
        store,
        source_path="/course/original.ipynb",
        run_id="r1",
    )

    with pytest.raises(SourceMismatchError):
        store.resolve("/course/other.ipynb", explicit_run=run_dir)

    assert (
        store.resolve(
            "/course/other.ipynb",
            explicit_run=run_dir,
            allow_source_mismatch=True,
        )
        == run_dir.resolve()
    )


def test_attempt_ids_remain_monotonic_after_cancellation(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    store = RunStore(tmp_path)
    store.initialize_execution_ledger(run_dir)

    first = store.reserve_attempt(run_dir, {"target": "source"})
    store.transition_attempt(
        run_dir,
        first.attempt_id,
        expected_status="prepared",
        new_status="cancelled",
    )
    second = RunStore(tmp_path).reserve_attempt(
        run_dir, {"target": "source"}
    )

    assert first.attempt_id == "A001"
    assert second.attempt_id == "A002"
    ledger = json.loads(
        (run_dir / ".notebook-coach/execution-ledger.json").read_text()
    )
    assert [entry["status"] for entry in ledger["entries"]] == [
        "cancelled",
        "prepared",
    ]


def test_concurrent_attempt_reservations_are_unique(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    RunStore(tmp_path).initialize_execution_ledger(run_dir)

    def reserve(_index: int) -> str:
        return RunStore(tmp_path).reserve_attempt(
            run_dir, {"target": "source"}
        ).attempt_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        attempt_ids = list(pool.map(reserve, range(12)))

    assert sorted(attempt_ids) == [f"A{index:03d}" for index in range(1, 13)]
    assert len(set(attempt_ids)) == 12
