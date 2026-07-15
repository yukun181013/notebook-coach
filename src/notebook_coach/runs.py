"""Immutable run directories and a small persistent execution ledger."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from notebook_coach import SCHEMA_VERSION


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_LEGAL_TRANSITIONS = {
    "prepared": {"running", "cancelled", "expired"},
    "running": {"completed", "failed", "expired"},
}


class RunStoreError(ValueError):
    """Base class for stable run-store errors."""


class RunNotFoundError(RunStoreError):
    """Raised when no finalized run matches a source."""


class AmbiguousRunError(RunStoreError):
    """Raised instead of guessing among several matching runs."""

    def __init__(self, candidates: list[str]) -> None:
        self.candidates = sorted(candidates)
        super().__init__(
            "Multiple runs match the source; choose one explicitly: "
            + ", ".join(self.candidates)
        )


class SourceMismatchError(RunStoreError):
    """Raised when an explicit run belongs to another source path."""


class LedgerTransitionError(RunStoreError):
    """Raised for an invalid or stale execution-ledger transition."""


@dataclass(frozen=True)
class Stage:
    run_id: str
    source_path: str
    source_sha256: str
    stage_dir: Path
    final_dir: Path


@dataclass(frozen=True)
class AttemptReservation:
    attempt_id: str
    entry: dict[str, Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _safe_source_name(path: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(path).stem).strip("-.")
    return cleaned or "notebook"


def _source_id(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:8]


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise RunStoreError("Run artifact paths must stay inside the run directory.")
    return path


def _write_new_file(path: Path, body: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path.name}.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_file(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RunStore:
    """Create, finalize, and resolve notebook-coach run directories."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve(strict=False)
        self._now = now

    @property
    def staging_root(self) -> Path:
        return self.output_root / ".staging"

    def _source_directory(self, source_path: str) -> Path:
        return self.output_root / (
            f"{_safe_source_name(source_path)}-{_source_id(source_path)}"
        )

    @contextmanager
    def _creation_lock(self) -> Iterator[None]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_root / ".run-creation.lock"
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def create_stage(
        self, source_path: str | Path, source_sha256: str
    ) -> Stage:
        if (
            not isinstance(source_sha256, str)
            or _SHA256_PATTERN.fullmatch(source_sha256) is None
        ):
            raise RunStoreError("source_sha256 must be a lowercase SHA-256 value.")

        canonical = _canonical_path(source_path)
        timestamp = self._now().astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base_run_id = f"{timestamp}-{source_sha256[:8]}"
        source_directory = self._source_directory(canonical)

        with self._creation_lock():
            self.staging_root.mkdir(parents=True, exist_ok=True)
            suffix = 1
            while True:
                run_id = base_run_id if suffix == 1 else f"{base_run_id}-{suffix:02d}"
                stage_dir = self.staging_root / run_id
                final_dir = source_directory / run_id
                if not stage_dir.exists() and not final_dir.exists():
                    stage_dir.mkdir()
                    break
                suffix += 1

        stage = Stage(
            run_id=run_id,
            source_path=canonical,
            source_sha256=source_sha256,
            stage_dir=stage_dir,
            final_dir=final_dir,
        )
        try:
            _write_new_file(
                stage.stage_dir / "stage.json",
                _json_bytes(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": stage.run_id,
                        "source_path": stage.source_path,
                        "source_sha256": stage.source_sha256,
                        "final_dir": str(stage.final_dir),
                    }
                ),
            )
        except Exception:
            shutil.rmtree(stage.stage_dir, ignore_errors=True)
            raise
        return stage

    def load_stage(self, stage_dir: str | Path) -> Stage:
        directory = Path(stage_dir).expanduser().resolve(strict=False)
        try:
            metadata = json.loads((directory / "stage.json").read_text("utf-8"))
            return Stage(
                run_id=metadata["run_id"],
                source_path=metadata["source_path"],
                source_sha256=metadata["source_sha256"],
                stage_dir=directory,
                final_dir=Path(metadata["final_dir"]).resolve(strict=False),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise RunStoreError("Stage metadata is missing or invalid.") from None

    def finalize_stage(self, stage: Stage, files: dict[str, bytes]) -> Path:
        if "baseline.json" not in files:
            raise RunStoreError("Finalized runs require baseline.json.")
        if not stage.stage_dir.is_dir():
            raise RunStoreError("The diagnosis stage no longer exists.")
        if stage.final_dir.exists():
            raise FileExistsError("Refusing to overwrite an existing run.")

        checked_files: list[tuple[Path, bytes]] = []
        for relative_name, body in files.items():
            if not isinstance(relative_name, str) or not isinstance(
                body, (bytes, bytearray)
            ):
                raise RunStoreError("Run artifacts must map relative names to bytes.")
            checked_files.append((_safe_relative_path(relative_name), bytes(body)))

        stage.final_dir.parent.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        assembly = Path(
            tempfile.mkdtemp(
                prefix=f".finalize-{stage.run_id}-", dir=self.staging_root
            )
        )
        try:
            for relative_path, body in sorted(
                checked_files, key=lambda item: item[0].as_posix()
            ):
                _write_new_file(assembly / relative_path, body)
            if stage.final_dir.exists():
                raise FileExistsError("Refusing to overwrite an existing run.")
            os.replace(assembly, stage.final_dir)
        except Exception:
            shutil.rmtree(assembly, ignore_errors=True)
            raise

        shutil.rmtree(stage.stage_dir)
        return stage.final_dir

    def _load_baseline(self, run_dir: Path) -> dict[str, Any]:
        try:
            value = json.loads((run_dir / "baseline.json").read_text("utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            raise RunStoreError("Run baseline.json is missing or invalid.") from None
        if not isinstance(value, dict):
            raise RunStoreError("Run baseline.json is missing or invalid.")
        return value

    @staticmethod
    def _baseline_source_path(baseline: dict[str, Any]) -> str:
        source = baseline.get("source")
        if not isinstance(source, dict):
            raise RunStoreError("Run baseline source metadata is invalid.")
        path = source.get("path", source.get("canonical_path"))
        if not isinstance(path, str):
            raise RunStoreError("Run baseline source path is invalid.")
        return _canonical_path(path)

    def resolve(
        self,
        source_path: str | Path,
        explicit_run: Path | None = None,
        allow_source_mismatch: bool = False,
    ) -> Path:
        canonical = _canonical_path(source_path)
        if explicit_run is not None:
            run_dir = Path(explicit_run).expanduser().resolve(strict=False)
            baseline = self._load_baseline(run_dir)
            actual_source = self._baseline_source_path(baseline)
            if actual_source != canonical and not allow_source_mismatch:
                raise SourceMismatchError(
                    "Explicit run source does not match the requested notebook."
                )
            return run_dir

        source_directory = self._source_directory(canonical)
        candidates: list[Path] = []
        if source_directory.is_dir():
            for child in source_directory.iterdir():
                if not child.is_dir() or not (child / "baseline.json").is_file():
                    continue
                baseline = self._load_baseline(child)
                if self._baseline_source_path(baseline) == canonical:
                    candidates.append(child.resolve())

        candidates.sort(key=lambda path: path.name)
        if not candidates:
            raise RunNotFoundError("No run matches the requested notebook.")
        if len(candidates) > 1:
            raise AmbiguousRunError([path.name for path in candidates])
        return candidates[0]

    @staticmethod
    def _ledger_paths(run_dir: str | Path) -> tuple[Path, Path]:
        state_dir = Path(run_dir).expanduser().resolve(strict=False) / ".notebook-coach"
        return state_dir / "execution-ledger.json", state_dir / "execution-ledger.lock"

    @contextmanager
    def _locked_ledger(self, run_dir: str | Path) -> Iterator[Path]:
        ledger_path, lock_path = self._ledger_paths(run_dir)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield ledger_path
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _read_ledger(path: Path) -> dict[str, Any]:
        try:
            ledger = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            raise RunStoreError("Execution ledger is missing or invalid.") from None
        if (
            not isinstance(ledger, dict)
            or ledger.get("schema_version") != SCHEMA_VERSION
            or not isinstance(ledger.get("next_attempt"), int)
            or ledger["next_attempt"] < 1
            or not isinstance(ledger.get("entries"), list)
        ):
            raise RunStoreError("Execution ledger is missing or invalid.")
        return ledger

    def initialize_execution_ledger(self, run_dir: Path) -> None:
        with self._locked_ledger(run_dir) as ledger_path:
            if ledger_path.exists():
                self._read_ledger(ledger_path)
                return
            _write_new_file(
                ledger_path,
                _json_bytes(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "next_attempt": 1,
                        "entries": [],
                    }
                ),
            )

    def reserve_attempt(
        self, run_dir: Path, metadata: dict[str, Any]
    ) -> AttemptReservation:
        if not isinstance(metadata, dict):
            raise RunStoreError("Attempt metadata must be a mapping.")
        self.initialize_execution_ledger(run_dir)
        with self._locked_ledger(run_dir) as ledger_path:
            ledger = self._read_ledger(ledger_path)
            number = ledger["next_attempt"]
            attempt_id = f"A{number:03d}"
            entry = {
                "attempt_id": attempt_id,
                "status": "prepared",
                "created_at": self._now().astimezone(timezone.utc).isoformat(),
                "metadata": dict(metadata),
            }
            ledger["next_attempt"] = number + 1
            ledger["entries"].append(entry)
            _replace_file(ledger_path, _json_bytes(ledger))
        return AttemptReservation(attempt_id=attempt_id, entry=dict(entry))

    def transition_attempt(
        self,
        run_dir: Path,
        attempt_id: str,
        expected_status: str,
        new_status: str,
        details: dict | None = None,
    ) -> None:
        with self._locked_ledger(run_dir) as ledger_path:
            ledger = self._read_ledger(ledger_path)
            entry = next(
                (
                    item
                    for item in ledger["entries"]
                    if isinstance(item, dict)
                    and item.get("attempt_id") == attempt_id
                ),
                None,
            )
            if entry is None:
                raise LedgerTransitionError("Attempt ID does not exist.")
            current_status = entry.get("status")
            if current_status != expected_status:
                raise LedgerTransitionError("Attempt status changed unexpectedly.")
            if new_status not in _LEGAL_TRANSITIONS.get(current_status, set()):
                raise LedgerTransitionError("Attempt status transition is not allowed.")
            entry["status"] = new_status
            entry["updated_at"] = self._now().astimezone(timezone.utc).isoformat()
            if details is not None:
                if not isinstance(details, dict):
                    raise LedgerTransitionError("Attempt details must be a mapping.")
                entry["details"] = dict(details)
            _replace_file(ledger_path, _json_bytes(ledger))
