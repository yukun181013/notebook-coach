"""Machine-readable command-line entry points for notebook-coach."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from notebook_coach.contracts import ContractError
from notebook_coach.execution import (
    ExecutionBlockedError,
    cancel_execution,
    execute_request,
    prepare_execution,
)
from notebook_coach.notebooks import NotebookInputError
from notebook_coach.revisions import RevisionError, apply_execution_review
from notebook_coach.runs import (
    AmbiguousRunError,
    RunNotFoundError,
    RunStore,
    RunStoreError,
    SourceMismatchError,
)
from notebook_coach.workflows import (
    ConfirmationRequiredError,
    WorkflowError,
    finalize_diagnosis,
    finalize_verification,
    prepare_diagnosis,
    prepare_verification,
    validate_run,
)


COMMANDS = (
    "prepare-diagnosis",
    "finalize-diagnosis",
    "resolve-run",
    "prepare-verification",
    "finalize-verification",
    "prepare-execution",
    "execute",
    "cancel-execution",
    "apply-execution-review",
    "validate-run",
)


class CLIInputError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class _JSONArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise CLIInputError("invalid_arguments", "Command arguments are invalid.")


def _handle_prepare_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    prepared = prepare_diagnosis(
        args.source,
        args.output_root,
        cells=args.cells,
    )
    return {
        "status": "awaiting_model_assessment",
        "run_id": prepared.stage.run_id,
        "stage": str(prepared.stage.stage_dir.resolve()),
        "assessment_path": str(prepared.assessment_path.resolve()),
        "risk": prepared.risk,
    }


def _handle_finalize_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = finalize_diagnosis(args.stage)
    return {"status": "finalized", "run_dir": str(run_dir)}


def _handle_resolve_run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = RunStore(args.output_root).resolve(
        args.source,
        explicit_run=Path(args.run) if args.run else None,
        allow_source_mismatch=args.confirm_source_mismatch,
    )
    return {"status": "resolved", "run_dir": str(run_dir.resolve())}


def _handle_validate_run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = validate_run(args.run_dir)
    return {"status": "valid", "run_dir": str(run_dir)}


def _handle_prepare_execution(args: argparse.Namespace) -> dict[str, Any]:
    prepared = prepare_execution(
        args.run,
        phase=args.phase,
        target=args.target,
        cell_timeout=args.cell_timeout,
        total_timeout=args.total_timeout,
    )
    return {
        "status": "awaiting_execution_confirmation",
        "request_path": str(prepared.request_path),
        "request": prepared.request,
    }


def _handle_execute(args: argparse.Namespace) -> dict[str, Any]:
    if not args.request or not args.confirmed_target_sha256:
        raise ExecutionBlockedError(
            "execution_confirmation_required",
            "Execution requires a prepared request and confirmed target hash.",
        )
    log_path = execute_request(args.request, args.confirmed_target_sha256)
    return {"status": "executed", "log_path": str(log_path.resolve())}


def _handle_cancel_execution(args: argparse.Namespace) -> dict[str, Any]:
    cancel_execution(args.request)
    return {"status": "cancelled"}


def _handle_prepare_verification(args: argparse.Namespace) -> dict[str, Any]:
    prepared = prepare_verification(
        args.source,
        args.run,
        confirm_source_mismatch=args.confirm_source_mismatch,
        confirm_environment_mismatch=args.confirm_environment_mismatch,
    )
    return {
        "status": "awaiting_model_assessment",
        "run_id": prepared.run_id,
        "stage": str(prepared.stage_dir),
        "assessment_path": str(prepared.assessment_path),
        "source_snapshot_path": str(prepared.source_snapshot_path),
        "challenge_snapshot_path": str(prepared.challenge_snapshot_path),
        "challenge_verifiability": prepared.challenge_verifiability,
    }


def _handle_finalize_verification(args: argparse.Namespace) -> dict[str, Any]:
    result = finalize_verification(args.stage)
    return {
        "status": "finalized",
        "verification_report": str(result.report_path.resolve()),
        "verification_state": str(result.state_path.resolve()),
    }


def _handle_apply_execution_review(args: argparse.Namespace) -> dict[str, Any]:
    result = apply_execution_review(args.run, args.log, args.review)
    return {
        "status": "revised",
        "revision": result.revision,
        "report_path": str(result.report_path.resolve()),
    }


def _handle_not_implemented(_args: argparse.Namespace) -> dict[str, Any]:
    raise WorkflowError(
        "not_implemented", "This command is not available in the static MVP."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _JSONArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=_JSONArgumentParser
    )

    prepare = subparsers.add_parser("prepare-diagnosis")
    prepare.add_argument("source")
    prepare.add_argument("--output-root", default="notebook-coach-output")
    prepare.add_argument("--cells")
    prepare.set_defaults(handler=_handle_prepare_diagnosis)

    finalize = subparsers.add_parser("finalize-diagnosis")
    finalize.add_argument("--stage", required=True)
    finalize.set_defaults(handler=_handle_finalize_diagnosis)

    resolve = subparsers.add_parser("resolve-run")
    resolve.add_argument("source")
    resolve.add_argument("--output-root", default="notebook-coach-output")
    resolve.add_argument("--run")
    resolve.add_argument("--confirm-source-mismatch", action="store_true")
    resolve.set_defaults(handler=_handle_resolve_run)

    validate = subparsers.add_parser("validate-run")
    validate.add_argument("run_dir")
    validate.set_defaults(handler=_handle_validate_run)

    prepare_execution_parser = subparsers.add_parser("prepare-execution")
    prepare_execution_parser.add_argument("--run", required=True)
    prepare_execution_parser.add_argument(
        "--phase", required=True, choices=("diagnosis", "verification")
    )
    prepare_execution_parser.add_argument(
        "--target", required=True, choices=("source", "challenge")
    )
    prepare_execution_parser.add_argument("--cell-timeout", type=int, default=30)
    prepare_execution_parser.add_argument("--total-timeout", type=int, default=120)
    prepare_execution_parser.set_defaults(handler=_handle_prepare_execution)

    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--request")
    execute_parser.add_argument("--confirmed-target-sha256")
    execute_parser.set_defaults(handler=_handle_execute)

    cancel_parser = subparsers.add_parser("cancel-execution")
    cancel_parser.add_argument("--request", required=True)
    cancel_parser.set_defaults(handler=_handle_cancel_execution)

    prepare_verification_parser = subparsers.add_parser("prepare-verification")
    prepare_verification_parser.add_argument("source")
    prepare_verification_parser.add_argument("--run", required=True)
    prepare_verification_parser.add_argument(
        "--confirm-source-mismatch", action="store_true"
    )
    prepare_verification_parser.add_argument(
        "--confirm-environment-mismatch", action="store_true"
    )
    prepare_verification_parser.set_defaults(handler=_handle_prepare_verification)

    finalize_verification_parser = subparsers.add_parser("finalize-verification")
    finalize_verification_parser.add_argument("--stage", required=True)
    finalize_verification_parser.set_defaults(handler=_handle_finalize_verification)

    apply_review_parser = subparsers.add_parser("apply-execution-review")
    apply_review_parser.add_argument("--run", required=True)
    apply_review_parser.add_argument("--log", required=True)
    apply_review_parser.add_argument("--review", required=True)
    apply_review_parser.set_defaults(handler=_handle_apply_execution_review)

    implemented = {
        "prepare-diagnosis",
        "finalize-diagnosis",
        "resolve-run",
        "validate-run",
        "prepare-execution",
        "execute",
        "cancel-execution",
        "prepare-verification",
        "finalize-verification",
        "apply-execution-review",
    }
    for command in COMMANDS:
        if command in implemented:
            continue
        command_parser = subparsers.add_parser(command)
        command_parser.set_defaults(handler=_handle_not_implemented)

    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        payload = args.handler(args)
    except AmbiguousRunError as error:
        _print_json(
            {
                "code": "ambiguous_run",
                "message": "Multiple runs match; choose one explicitly.",
                "candidates": error.candidates,
            },
            stream=sys.stderr,
        )
        return 3
    except ExecutionBlockedError as error:
        _print_json(
            {"code": error.code, "message": str(error)}, stream=sys.stderr
        )
        return 4
    except RevisionError as error:
        _print_json(
            {"code": error.code, "message": str(error)}, stream=sys.stderr
        )
        return 2
    except ConfirmationRequiredError as error:
        _print_json(
            {"code": error.code, "message": str(error), **error.details},
            stream=sys.stderr,
        )
        return 2
    except (ContractError, WorkflowError, CLIInputError) as error:
        _print_json(
            {"code": error.code, "message": str(error)}, stream=sys.stderr
        )
        return 2
    except NotebookInputError as error:
        _print_json(
            {"code": "invalid_notebook", "message": str(error)},
            stream=sys.stderr,
        )
        return 2
    except SourceMismatchError:
        _print_json(
            {
                "code": "source_mismatch",
                "message": "Explicit run belongs to a different source notebook.",
            },
            stream=sys.stderr,
        )
        return 2
    except RunNotFoundError:
        _print_json(
            {"code": "run_not_found", "message": "No matching run was found."},
            stream=sys.stderr,
        )
        return 2
    except RunStoreError:
        _print_json(
            {"code": "run_invalid", "message": "Run data is missing or invalid."},
            stream=sys.stderr,
        )
        return 2
    except Exception:
        _print_json(
            {"code": "runtime_failure", "message": "Command failed unexpectedly."},
            stream=sys.stderr,
        )
        return 5
    _print_json(payload, stream=sys.stdout)
    return 0


def _print_json(value: dict[str, Any], *, stream) -> None:
    print(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        file=stream,
    )
