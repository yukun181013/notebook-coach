from notebook_coach import RUBRIC_VERSION, SCHEMA_VERSION, __version__
from notebook_coach.cli import build_parser


def test_package_exports_versions_and_cli_commands():
    assert __version__ == "0.1.0"
    assert SCHEMA_VERSION == "1.0"
    assert RUBRIC_VERSION == "1.0"
    assert set(build_parser()._subparsers._group_actions[0].choices) == {
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
    }
