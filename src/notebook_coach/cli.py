import argparse


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


def _handle_command(_args: argparse.Namespace) -> int:
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in COMMANDS:
        command_parser = subparsers.add_parser(command)
        command_parser.set_defaults(handler=_handle_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
