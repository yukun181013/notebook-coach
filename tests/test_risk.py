from __future__ import annotations

from collections.abc import Callable

import pytest

from notebook_coach.notebooks import build_snapshot
from notebook_coach.risk import scan_snapshot


SnapshotFactory = Callable[[list[str]], dict]
FIXTURE_SECRET = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"


def _finding_for(result: dict, category: str) -> dict:
    return next(
        finding
        for finding in result["findings"]
        if finding["category"] == category
    )


@pytest.mark.parametrize(
    ("source", "sensitive_fragment", "category"),
    [
        (
            "import os; secret=os.system('echo notebook-coach-sensitive')",
            "notebook-coach-sensitive",
            "shell",
        ),
        (
            "import os; password=os.remove('/tmp/notebook-coach-sensitive')",
            "/tmp/notebook-coach-sensitive",
            "filesystem_delete",
        ),
        (
            "from pathlib import Path; "
            "token=Path('/tmp/notebook-coach-sensitive').unlink()",
            "/tmp/notebook-coach-sensitive",
            "filesystem_delete",
        ),
        (
            "api_key=open('.env').read()",
            ".env",
            "credential_read",
        ),
    ],
    ids=["shell", "os-remove", "path-unlink", "credential-read"],
)
def test_real_snapshot_retains_risk_findings_after_source_redaction(
    notebook_factory,
    source: str,
    sensitive_fragment: str,
    category: str,
):
    path = notebook_factory(code=source)

    snapshot = build_snapshot(path)
    visible_source = snapshot["cells"][0]["source"]["text"]

    assert "[REDACTED]" in visible_source
    assert sensitive_fragment not in repr(snapshot)

    result = scan_snapshot(snapshot)

    assert _finding_for(result, category)["severity"] == "high"
    assert result["blocked"] is True


@pytest.mark.parametrize(
    "risk_metadata",
    [
        None,
        {"source_sha256": "f" * 64, "categories": []},
        {"source_sha256": "source-hash", "categories": ["unknown"]},
        {"source_sha256": "source-hash", "categories": "shell"},
    ],
    ids=["missing", "hash-mismatch", "unknown-category", "non-list-categories"],
)
def test_redacted_source_without_trusted_risk_metadata_fails_closed(
    snapshot_factory: SnapshotFactory,
    risk_metadata,
):
    snapshot = snapshot_factory(["password = 'synthetic-secret-value'"])
    source_hash = snapshot["cells"][0]["source"]["sha256"]
    if risk_metadata is None:
        snapshot["cells"][0].pop("risk", None)
    else:
        snapshot["cells"][0]["risk"] = {
            key: source_hash if value == "source-hash" else value
            for key, value in risk_metadata.items()
        }

    result = scan_snapshot(snapshot)

    assert _finding_for(result, "risk_metadata")["severity"] == "high"
    assert result["blocked"] is True


def test_blocks_shell_network_subprocess_and_delete_calls(
    snapshot_factory: SnapshotFactory,
):
    snapshot = snapshot_factory(
        [
            "!curl https://example.com",
            "import subprocess; subprocess.run(['echo', 'x'])",
            "import os; os.remove('/tmp/x')",
            "import requests; requests.get('https://example.com')",
        ]
    )

    result = scan_snapshot(snapshot)

    assert result["blocked"] is True
    assert {item["category"] for item in result["findings"]} >= {
        "shell",
        "subprocess",
        "filesystem_delete",
        "network",
    }


def test_safe_math_notebook_is_not_blocked(snapshot_factory: SnapshotFactory):
    snapshot = snapshot_factory(
        [
            "import math\nvalue = math.sqrt(9)",
            "import numpy as np\nweights = scores.softmax(dim=-1)",
        ]
    )

    result = scan_snapshot(snapshot)

    assert result == {"blocked": False, "findings": []}


@pytest.mark.parametrize(
    ("source", "category"),
    [
        ("import os\nos.system('echo safe')", "shell"),
        ("import subprocess", "subprocess"),
        ("import subprocess as sp\nsp.run(['echo', 'x'])", "subprocess"),
        ("from subprocess import run\nrun(['echo', 'x'])", "subprocess"),
        ("subprocess.run(['echo', 'x'])", "subprocess"),
        ("import requests", "network"),
        ("import httpx as client\nclient.get('https://example.com')", "network"),
        ("import aiohttp", "network"),
        ("import urllib.request", "network"),
        ("import socket\nsocket.create_connection(('example.com', 443))", "network"),
        ("requests.get('https://example.com')", "network"),
    ],
)
def test_ast_detects_risky_imports_and_calls(
    snapshot_factory: SnapshotFactory,
    source: str,
    category: str,
):
    result = scan_snapshot(snapshot_factory([source]))

    finding = _finding_for(result, category)
    assert finding["severity"] == "high"
    assert result["blocked"] is True


@pytest.mark.parametrize(
    "source",
    [
        "import os\nos.remove('/tmp/x')",
        "import os\nos.unlink('/tmp/x')",
        "import os\nos.rmdir('/tmp/x')",
        "import shutil\nshutil.rmtree('/tmp/x')",
        "from pathlib import Path\nPath('/tmp/x').unlink()",
        "from pathlib import Path\nPath('/tmp/x').rmdir()",
        "from pathlib import Path\nPath(target).unlink()",
    ],
)
def test_ast_detects_filesystem_delete_calls(
    snapshot_factory: SnapshotFactory,
    source: str,
):
    result = scan_snapshot(snapshot_factory([source]))

    assert _finding_for(result, "filesystem_delete")["severity"] == "high"
    assert result["blocked"] is True


@pytest.mark.parametrize(
    "source",
    [
        (
            "from pathlib import Path\n"
            "target = Path('/tmp/x')\n"
            "target.unlink()"
        ),
        (
            "from pathlib import Path\n"
            "target = Path('/tmp/x')\n"
            "target.rmdir()"
        ),
        "from pathlib import Path\nPath.unlink(Path('/tmp/x'))",
        "from pathlib import Path\nPath.rmdir(Path('/tmp/x'))",
        (
            "from pathlib import Path as P\n"
            "target: P = P('/tmp/x')\n"
            "target.unlink()"
        ),
        (
            "import pathlib as paths\n"
            "target: paths.Path = paths.Path('/tmp/x')\n"
            "target.rmdir()"
        ),
    ],
)
def test_ast_tracks_path_instances_and_class_method_delete_calls(
    snapshot_factory: SnapshotFactory,
    source: str,
):
    result = scan_snapshot(snapshot_factory([source]))

    assert _finding_for(result, "filesystem_delete")["severity"] == "high"
    assert result["blocked"] is True


@pytest.mark.parametrize("method", ["unlink", "rmdir"])
def test_path_instance_alias_propagates_to_delete_calls(
    snapshot_factory: SnapshotFactory,
    method: str,
):
    source = (
        "from pathlib import Path\n"
        "p = Path('/tmp/x')\n"
        "q = p\n"
        f"q.{method}()"
    )

    result = scan_snapshot(snapshot_factory([source]))

    assert _finding_for(result, "filesystem_delete")["severity"] == "high"
    assert result["blocked"] is True


@pytest.mark.parametrize(
    ("sources", "expected_cell_index"),
    [
        (
            [
                "from pathlib import Path",
                "Path('/tmp/x').unlink()",
            ],
            1,
        ),
        (
            [
                "from pathlib import Path",
                "p = Path('/tmp/x')",
                "p.rmdir()",
            ],
            2,
        ),
        (
            [
                "import os as x",
                "x.remove('/tmp/x')",
            ],
            1,
        ),
    ],
)
def test_ast_symbols_persist_across_code_cells(
    snapshot_factory: SnapshotFactory,
    sources: list[str],
    expected_cell_index: int,
):
    result = scan_snapshot(snapshot_factory(sources))

    finding = _finding_for(result, "filesystem_delete")
    assert finding["cell_index"] == expected_cell_index
    assert finding["severity"] == "high"
    assert result["blocked"] is True


def test_nested_function_rebinding_does_not_clear_outer_path_binding(
    snapshot_factory: SnapshotFactory,
):
    source = (
        "from pathlib import Path\n"
        "p = Path('/tmp/x')\n"
        "def unused():\n"
        "    p = object()\n"
        "p.unlink()"
    )

    result = scan_snapshot(snapshot_factory([source]))

    assert _finding_for(result, "filesystem_delete")["severity"] == "high"
    assert result["blocked"] is True


def test_control_flow_merges_path_bindings_conservatively(
    snapshot_factory: SnapshotFactory,
):
    snapshot = snapshot_factory(
        [
            (
                "from pathlib import Path\n"
                "p = Path('/tmp/x')\n"
                "if False:\n"
                "    p = object()\n"
                "p.unlink()"
            ),
            (
                "from pathlib import Path\n"
                "p = Path('/tmp/x')\n"
                "if condition:\n"
                "    p = object()\n"
                "else:\n"
                "    pass\n"
                "p.rmdir()"
            ),
            (
                "from pathlib import Path\n"
                "p = Path('/tmp/x')\n"
                "while condition:\n"
                "    p = object()\n"
                "p.unlink()"
            ),
            (
                "from pathlib import Path\n"
                "p = Path('/tmp/x')\n"
                "try:\n"
                "    might_fail()\n"
                "    p = object()\n"
                "except Exception:\n"
                "    pass\n"
                "p.rmdir()"
            ),
            (
                "from pathlib import Path\n"
                "p = Path('/tmp/x')\n"
                "if condition:\n"
                "    p = object()\n"
                "else:\n"
                "    p = object()\n"
                "p.unlink()"
            ),
        ]
    )

    result = scan_snapshot(snapshot)

    delete_findings = [
        finding
        for finding in result["findings"]
        if finding["category"] == "filesystem_delete"
    ]
    assert [finding["cell_index"] for finding in delete_findings] == [0, 1, 2, 3]


def test_match_cases_start_from_the_same_path_state(
    snapshot_factory: SnapshotFactory,
):
    source = (
        "from pathlib import Path\n"
        "p = Path('/tmp/x')\n"
        "match choice:\n"
        "    case 'replace':\n"
        "        p = object()\n"
        "    case 'delete':\n"
        "        p.unlink()"
    )

    result = scan_snapshot(snapshot_factory([source]))

    assert _finding_for(result, "filesystem_delete")["severity"] == "high"
    assert result["blocked"] is True


def test_match_pattern_binding_shadows_outer_path_identity(
    snapshot_factory: SnapshotFactory,
):
    source = (
        "from pathlib import Path\n"
        "class Fake:\n"
        "    def unlink(self): pass\n"
        "p = Path('/tmp/x')\n"
        "match {'p': Fake()}:\n"
        "    case {'p': p}:\n"
        "        p.unlink()"
    )

    result = scan_snapshot(snapshot_factory([source]))

    assert result == {"blocked": False, "findings": []}


@pytest.mark.parametrize("method", ["unlink", "rmdir"])
def test_try_handler_merges_intermediate_path_states(
    snapshot_factory: SnapshotFactory,
    method: str,
):
    source = (
        "from pathlib import Path\n"
        "def might_fail():\n"
        "    raise RuntimeError('boom')\n"
        "p = object()\n"
        "try:\n"
        "    p = Path('/tmp/x')\n"
        "    might_fail()\n"
        "    p = object()\n"
        "except RuntimeError:\n"
        "    pass\n"
        f"p.{method}()"
    )

    result = scan_snapshot(snapshot_factory([source]))

    assert _finding_for(result, "filesystem_delete")["severity"] == "high"
    assert result["blocked"] is True


@pytest.mark.parametrize(
    ("operation", "blocked"),
    [
        ("1 + 2", False),
        ("1 / 0", True),
        ("assert True", False),
        ("assert False", True),
        ("[1][0]", False),
        ("[][0]", True),
    ],
    ids=[
        "safe-binary-operation",
        "raising-binary-operation",
        "safe-assertion",
        "raising-assertion",
        "safe-subscription",
        "raising-subscription",
    ],
)
def test_try_handler_classifies_constant_expression_exception_states(
    snapshot_factory: SnapshotFactory,
    operation: str,
    blocked: bool,
):
    source = (
        "from pathlib import Path\n"
        "class Fake:\n"
        "    def unlink(self): pass\n"
        "fake = Fake()\n"
        "p = object()\n"
        "try:\n"
        "    p = Path('/tmp/x')\n"
        f"    {operation}\n"
        "    p = fake\n"
        "    raise RuntimeError('stop')\n"
        "except Exception:\n"
        "    p.unlink()"
    )

    result = scan_snapshot(snapshot_factory([source]))

    delete_findings = [
        finding
        for finding in result["findings"]
        if finding["category"] == "filesystem_delete"
    ]
    assert bool(delete_findings) is blocked
    assert result["blocked"] is blocked
    if blocked:
        assert delete_findings[0]["severity"] == "high"


def test_try_handler_ignores_uniterated_safe_literal_generator_creation(
    snapshot_factory: SnapshotFactory,
):
    source = (
        "from pathlib import Path\n"
        "class Fake:\n"
        "    def unlink(self): pass\n"
        "fake = Fake()\n"
        "p = object()\n"
        "try:\n"
        "    p = Path('/tmp/x')\n"
        "    generator = (1 / 0 for _ in (1, 2))\n"
        "    p = fake\n"
        "    raise RuntimeError('stop')\n"
        "except RuntimeError:\n"
        "    p.unlink()"
    )

    result = scan_snapshot(snapshot_factory([source]))

    assert result == {"blocked": False, "findings": []}


@pytest.mark.parametrize(
    "target",
    ["[item]", "(item,)", "item, *rest"],
    ids=["list", "tuple", "starred"],
)
def test_try_handler_records_generator_consumed_by_unpacking_assignment(
    snapshot_factory: SnapshotFactory,
    target: str,
):
    source = (
        "from pathlib import Path\n"
        "try:\n"
        "    p = Path('/tmp/x')\n"
        f"    {target} = (1 / 0 for _ in (1,))\n"
        "    p = None\n"
        "except ZeroDivisionError:\n"
        "    p.unlink()"
    )

    result = scan_snapshot(snapshot_factory([source]))

    assert _finding_for(result, "filesystem_delete")["severity"] == "high"
    assert result["blocked"] is True


@pytest.mark.parametrize(
    "target",
    ["[a, b]", "(a, b)", "a, *rest, b"],
    ids=["list", "tuple", "starred"],
)
def test_try_handler_records_unpacking_errors_before_binding(
    snapshot_factory: SnapshotFactory,
    target: str,
):
    source = (
        "from pathlib import Path\n"
        "p = None\n"
        "try:\n"
        "    p = Path('/tmp/x')\n"
        f"    {target} = (1,)\n"
        "    p = None\n"
        "except ValueError:\n"
        "    p.unlink()"
    )

    result = scan_snapshot(snapshot_factory([source]))

    assert _finding_for(result, "filesystem_delete")["severity"] == "high"
    assert result["blocked"] is True


def test_try_handler_uses_only_states_before_potential_exceptions(
    snapshot_factory: SnapshotFactory,
):
    source = (
        "from pathlib import Path\n"
        "class Fake:\n"
        "    def unlink(self): pass\n"
        "fake = Fake()\n"
        "p = object()\n"
        "try:\n"
        "    p = Path('/tmp/x')\n"
        "    p = fake\n"
        "    raise RuntimeError('stop')\n"
        "except RuntimeError:\n"
        "    p.unlink()"
    )

    result = scan_snapshot(snapshot_factory([source]))

    assert result == {"blocked": False, "findings": []}


def test_try_handler_ignores_exceptions_in_deferred_function_body(
    snapshot_factory: SnapshotFactory,
):
    source = (
        "from pathlib import Path\n"
        "p = object()\n"
        "try:\n"
        "    def deferred():\n"
        "        p = Path('/tmp/x')\n"
        "        might_fail()\n"
        "except RuntimeError:\n"
        "    pass\n"
        "p.unlink()"
    )

    result = scan_snapshot(snapshot_factory([source]))

    assert result == {"blocked": False, "findings": []}


def test_try_handler_records_comprehension_exceptions_at_handler_scope(
    snapshot_factory: SnapshotFactory,
):
    source = (
        "from pathlib import Path\n"
        "p = object()\n"
        "try:\n"
        "    values = [Path('/tmp/x') for _ in [0]]\n"
        "except RuntimeError:\n"
        "    pass\n"
        "p.unlink()"
    )

    result = scan_snapshot(snapshot_factory([source]))

    assert result == {"blocked": False, "findings": []}


def test_try_handler_preserves_outer_path_binding_when_class_body_raises(
    snapshot_factory: SnapshotFactory,
):
    source = (
        "from pathlib import Path\n"
        "try:\n"
        "    class Path:\n"
        "        raise RuntimeError('stop')\n"
        "except RuntimeError:\n"
        "    Path('/tmp/x').unlink()"
    )

    result = scan_snapshot(snapshot_factory([source]))

    assert _finding_for(result, "filesystem_delete")["severity"] == "high"
    assert result["blocked"] is True


def test_function_parameters_and_locals_are_scope_isolated(
    snapshot_factory: SnapshotFactory,
):
    snapshot = snapshot_factory(
        [
            (
                "from pathlib import Path\n"
                "def cleanup():\n"
                "    local = Path('/tmp/x')\n"
                "    local.unlink()"
            ),
            (
                "from pathlib import Path\n"
                "p = Path('/tmp/x')\n"
                "def cleanup(p):\n"
                "    p.unlink()"
            ),
            (
                "from pathlib import Path\n"
                "def remember():\n"
                "    local = Path('/tmp/x')\n"
                "local.unlink()"
            ),
        ]
    )

    result = scan_snapshot(snapshot)

    delete_findings = [
        finding
        for finding in result["findings"]
        if finding["category"] == "filesystem_delete"
    ]
    assert [finding["cell_index"] for finding in delete_findings] == [0]


def test_comprehension_targets_use_an_isolated_scope(
    snapshot_factory: SnapshotFactory,
):
    snapshot = snapshot_factory(
        [
            (
                "from pathlib import Path\n"
                "class Fake: pass\n"
                "def cleanup():\n"
                "    [None for Path in [Fake]]\n"
                "    Path('/tmp/x').unlink()"
            ),
            (
                "from pathlib import Path\n"
                "class Fake:\n"
                "    def unlink(self): pass\n"
                "[Path().unlink() for Path in [Fake]]"
            ),
            (
                "from pathlib import Path\n"
                "class Fake: pass\n"
                "[None for Path in [Fake]]\n"
                "Path('/tmp/x').rmdir()"
            ),
        ]
    )

    result = scan_snapshot(snapshot)

    delete_findings = [
        finding
        for finding in result["findings"]
        if finding["category"] == "filesystem_delete"
    ]
    assert [finding["cell_index"] for finding in delete_findings] == [0, 2]


def test_block_binding_targets_shadow_path_identity(
    snapshot_factory: SnapshotFactory,
):
    snapshot = snapshot_factory(
        [
            (
                "from pathlib import Path\n"
                "class Fake:\n"
                "    def unlink(self): pass\n"
                "for Path in [Fake]:\n"
                "    Path().unlink()"
            ),
            (
                "from pathlib import Path\n"
                "class Fake:\n"
                "    def rmdir(self): pass\n"
                "with manager() as Path:\n"
                "    Path().rmdir()"
            ),
            (
                "from pathlib import Path\n"
                "class Fake:\n"
                "    def unlink(self): pass\n"
                "try:\n"
                "    risky()\n"
                "except Exception as Path:\n"
                "    Path().unlink()"
            ),
            (
                "from pathlib import Path\n"
                "class Fake:\n"
                "    def unlink(self): pass\n"
                "for Path in [Fake]:\n"
                "    Path().unlink()\n"
                "Path('/tmp/x').unlink()"
            ),
        ]
    )

    result = scan_snapshot(snapshot)

    delete_findings = [
        finding
        for finding in result["findings"]
        if finding["category"] == "filesystem_delete"
    ]
    assert [finding["cell_index"] for finding in delete_findings] == [3]


@pytest.mark.parametrize(
    "source",
    [
        (
            "class Path:\n"
            "    def unlink(self): pass\n"
            "Path().unlink()"
        ),
        (
            "from pathlib import Path\n"
            "class Fake:\n"
            "    def unlink(self): pass\n"
            "Path = Fake\n"
            "Path().unlink()"
        ),
        (
            "from pathlib import Path as P\n"
            "class Fake:\n"
            "    def rmdir(self): pass\n"
            "P = Fake\n"
            "P().rmdir()"
        ),
        (
            "import pathlib as paths\n"
            "class Fake:\n"
            "    def unlink(self): pass\n"
            "class Namespace:\n"
            "    Path = Fake\n"
            "paths = Namespace()\n"
            "paths.Path().unlink()"
        ),
        (
            "from pathlib import Path\n"
            "class Path:\n"
            "    def rmdir(self): pass\n"
            "Path().rmdir()"
        ),
    ],
)
def test_only_imported_path_symbols_establish_constructor_identity(
    snapshot_factory: SnapshotFactory,
    source: str,
):
    result = scan_snapshot(snapshot_factory([source]))

    assert result == {"blocked": False, "findings": []}


def test_path_tracking_does_not_flag_arbitrary_same_named_methods(
    snapshot_factory: SnapshotFactory,
):
    snapshot = snapshot_factory(
        [
            (
                "from pathlib import Path\n"
                "target = Path('/tmp/x')\n"
                "target.unlink()"
            ),
            (
                "class Cache:\n"
                "    def unlink(self): pass\n"
                "    def rmdir(self): pass\n"
                "cache = Cache()\n"
                "cache.unlink()\n"
                "cache.rmdir()"
            ),
        ]
    )

    result = scan_snapshot(snapshot)

    delete_findings = [
        finding
        for finding in result["findings"]
        if finding["category"] == "filesystem_delete"
    ]
    assert [finding["cell_index"] for finding in delete_findings] == [0]


@pytest.mark.parametrize(
    ("source", "expected_categories"),
    [
        ("!curl https://example.com", {"shell"}),
        ("%%bash\necho hello", {"shell"}),
        ("%pip install example", {"package_install"}),
        ("!pip install example", {"shell", "package_install"}),
        ("%conda install example", {"package_install"}),
    ],
)
def test_shell_and_package_magics_are_high_risk(
    snapshot_factory: SnapshotFactory,
    source: str,
    expected_categories: set[str],
):
    result = scan_snapshot(snapshot_factory([source]))
    findings = {
        finding["category"]: finding for finding in result["findings"]
    }

    assert set(findings) >= expected_categories
    assert all(findings[category]["severity"] == "high" for category in expected_categories)
    assert result["blocked"] is True


@pytest.mark.parametrize(
    "source",
    [
        "open('.env')",
        "open('~/.ssh/id_rsa')",
        "from pathlib import Path\nPath('~/.aws/credentials').read_text()",
    ],
)
def test_obvious_credential_file_reads_are_blocked(
    snapshot_factory: SnapshotFactory,
    source: str,
):
    result = scan_snapshot(snapshot_factory([source]))

    assert _finding_for(result, "credential_read")["severity"] == "high"
    assert result["blocked"] is True


@pytest.mark.parametrize(
    "source",
    [
        (
            "from pathlib import Path\n"
            "p = Path('~/.ssh/id_rsa')\n"
            "p.read_text()"
        ),
        (
            "from pathlib import Path\n"
            "p = Path('~/.ssh/id_rsa')\n"
            "alias = p\n"
            "alias.read_text()"
        ),
        (
            "from pathlib import Path\n"
            "def read_key():\n"
            "    p = Path('~/.ssh/id_rsa')\n"
            "    return p.read_text()"
        ),
        (
            "from pathlib import Path\n"
            "if condition:\n"
            "    p = Path('~/.ssh/id_rsa')\n"
            "else:\n"
            "    p = Path('data.csv')\n"
            "p.read_text()"
        ),
    ],
    ids=["direct", "alias", "function-scope", "branch-merge"],
)
def test_credential_path_identity_propagates_to_read_calls(
    snapshot_factory: SnapshotFactory,
    source: str,
):
    result = scan_snapshot(snapshot_factory([source]))

    assert _finding_for(result, "credential_read")["severity"] == "high"
    assert result["blocked"] is True


def test_credential_path_identity_persists_across_cells(
    snapshot_factory: SnapshotFactory,
):
    result = scan_snapshot(
        snapshot_factory(
            [
                "from pathlib import Path\np = Path('~/.ssh/id_rsa')",
                "p.read_text()",
            ]
        )
    )

    finding = _finding_for(result, "credential_read")
    assert finding["cell_index"] == 1
    assert finding["severity"] == "high"
    assert result["blocked"] is True


def test_ordinary_path_instance_data_read_is_not_blocked(
    snapshot_factory: SnapshotFactory,
):
    source = "from pathlib import Path\np = Path('data.csv')\np.read_text()"

    result = scan_snapshot(snapshot_factory([source]))

    assert result == {"blocked": False, "findings": []}


def test_ordinary_data_file_read_is_not_blocked(snapshot_factory: SnapshotFactory):
    result = scan_snapshot(snapshot_factory(["rows = open('data.csv').read()"]))

    assert result == {"blocked": False, "findings": []}


def test_syntax_error_is_low_severity_and_does_not_block(
    snapshot_factory: SnapshotFactory,
):
    result = scan_snapshot(snapshot_factory(["def broken(:\n    pass"]))

    assert result["blocked"] is False
    assert result["findings"] == [
        {
            "cell_index": 0,
            "category": "syntax_error",
            "severity": "low",
            "explanation": "Cell source could not be parsed as Python.",
        }
    ]


def test_syntax_fallback_still_detects_obvious_risky_patterns(
    snapshot_factory: SnapshotFactory,
):
    source = "def broken(:\n    subprocess.run(['echo', 'x'])\n!curl https://example.com"

    result = scan_snapshot(snapshot_factory([source]))

    categories = {finding["category"] for finding in result["findings"]}
    assert categories >= {"syntax_error", "subprocess", "shell"}
    assert result["blocked"] is True


def test_deep_valid_expression_fails_closed_without_recursion_error(
    snapshot_factory: SnapshotFactory,
):
    source = "+".join(["x"] * 400)
    assert len(source) == 799

    try:
        result = scan_snapshot(snapshot_factory([source]))
    except RecursionError:
        pytest.fail("risk analysis leaked RecursionError")

    assert _finding_for(result, "analysis_limit")["severity"] == "high"
    assert result["blocked"] is True


def test_valid_comments_and_strings_do_not_trigger_risk_findings(
    snapshot_factory: SnapshotFactory,
):
    source = 'message = "subprocess.run and !curl"\n# subprocess.run(["x"])\n# !curl x'

    result = scan_snapshot(snapshot_factory([source]))

    assert result == {"blocked": False, "findings": []}


def test_output_contract_is_exact_deduplicated_and_stable(
    snapshot_factory: SnapshotFactory,
):
    snapshot = snapshot_factory(
        [
            "import requests\nimport requests",
            "import subprocess\nimport subprocess",
            "def broken(:",
        ]
    )

    first = scan_snapshot(snapshot)
    second = scan_snapshot(snapshot)

    assert first == second
    assert set(first) == {"blocked", "findings"}
    assert all(
        set(finding)
        == {"cell_index", "category", "severity", "explanation"}
        for finding in first["findings"]
    )
    assert {finding["severity"] for finding in first["findings"]} <= {
        "high",
        "low",
    }
    assert first["findings"] == sorted(
        first["findings"],
        key=lambda finding: (
            finding["cell_index"],
            finding["category"],
            finding["explanation"],
        ),
    )
    identities = [
        (
            finding["cell_index"],
            finding["category"],
            finding["explanation"],
        )
        for finding in first["findings"]
    ]
    assert len(identities) == len(set(identities))


def test_explanations_are_redacted_single_line_bounded_and_do_not_echo_source(
    snapshot_factory: SnapshotFactory,
):
    source = (
        "import subprocess\n"
        f"subprocess.run(['echo', '{FIXTURE_SECRET}', '/very/private/path'])"
    )

    result = scan_snapshot(snapshot_factory([source]))

    assert FIXTURE_SECRET not in repr(result)
    assert "/very/private/path" not in repr(result)
    for finding in result["findings"]:
        explanation = finding["explanation"]
        assert "\n" not in explanation
        assert "\r" not in explanation
        assert len(explanation) <= 200


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_source_text", "source.text"),
        ("non_string_source_text", "source.text"),
        ("non_integer_cell_index", "cell index"),
        ("negative_cell_index", "cell index"),
        ("duplicate_cell_index", "duplicate cell index"),
        ("unknown_cell_type", "cell_type"),
    ],
)
def test_invalid_snapshot_cell_shape_raises_clear_value_error(
    snapshot_factory: SnapshotFactory,
    mutation: str,
    message: str,
):
    snapshot = snapshot_factory(["import subprocess"])
    if mutation == "missing_source_text":
        del snapshot["cells"][0]["source"]["text"]
    elif mutation == "non_string_source_text":
        snapshot["cells"][0]["source"]["text"] = 42
    elif mutation == "non_integer_cell_index":
        snapshot["cells"][0]["index"] = "zero"
    elif mutation == "negative_cell_index":
        snapshot["cells"][0]["index"] = -1
    elif mutation == "duplicate_cell_index":
        snapshot["cells"].append(dict(snapshot["cells"][0]))
    else:
        snapshot["cells"][0]["cell_type"] = "unknown"

    with pytest.raises(ValueError, match=message):
        scan_snapshot(snapshot)


@pytest.mark.parametrize("cell_type", ["markdown", "raw"])
@pytest.mark.parametrize("source", [None, {}], ids=["none", "missing-text"])
def test_non_code_cells_validate_common_source_shape(cell_type: str, source):
    snapshot = {
        "cells": [
            {
                "index": 0,
                "cell_type": cell_type,
                "source": source,
            }
        ]
    }

    with pytest.raises(ValueError, match="source.text"):
        scan_snapshot(snapshot)
