from __future__ import annotations

from collections.abc import Callable

import pytest

from notebook_coach.risk import scan_snapshot


SnapshotFactory = Callable[[list[str]], dict]
FIXTURE_SECRET = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"


def _finding_for(result: dict, category: str) -> dict:
    return next(
        finding
        for finding in result["findings"]
        if finding["category"] == category
    )


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
    ],
)
def test_invalid_snapshot_cell_shape_raises_clear_value_error(
    snapshot_factory: SnapshotFactory,
    mutation: str,
    message: str,
):
    snapshot = snapshot_factory(["x = 1"])
    if mutation == "missing_source_text":
        del snapshot["cells"][0]["source"]["text"]
    elif mutation == "non_string_source_text":
        snapshot["cells"][0]["source"]["text"] = 42
    else:
        snapshot["cells"][0]["index"] = "zero"

    with pytest.raises(ValueError, match=message):
        scan_snapshot(snapshot)
