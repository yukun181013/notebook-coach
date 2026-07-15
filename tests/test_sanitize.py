import hashlib
import re

import pytest

from notebook_coach.sanitize import redact_text, summarize_text


def test_redacts_secret_without_returning_original_value():
    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
    cleaned, labels = redact_text(f"OPENAI_API_KEY={secret}")
    assert cleaned == "OPENAI_API_KEY=[REDACTED]"
    assert secret not in repr((cleaned, labels))
    assert labels == ["openai_api_key"]


def test_summarize_text_keeps_length_hash_and_marker():
    result = summarize_text("x" * 5000, max_chars=100)
    assert result["truncated"] is True
    assert result["original_chars"] == 5000
    assert len(result["text"]) <= 100
    assert len(result["sha256"]) == 64


def test_redacts_github_token_without_returning_original_value():
    secret = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"
    cleaned, labels = redact_text(f"value={secret}")

    assert cleaned == "value=[REDACTED]"
    assert secret not in repr((cleaned, labels))
    assert labels == ["github_token"]


def test_redacts_entire_pem_private_key_without_returning_original_value():
    secret = """-----BEGIN PRIVATE KEY-----
not-a-real-private-key-payload
-----END PRIVATE KEY-----"""
    cleaned, labels = redact_text(f"before\n{secret}\nafter")

    assert cleaned == "before\n[REDACTED]\nafter"
    assert secret not in repr((cleaned, labels))
    assert labels == ["pem_private_key"]


@pytest.mark.parametrize(
    ("name", "secret"),
    [
        ("service_api_key", "abcdefghijklmnopqrstuvwxyz123456"),
        ("access_token", "token-value-abcdefghijklmnopqrstuvwxyz"),
        ("auth_token", "auth-token-value-abcdefghijklmnopqrstuvwxyz"),
        ("database_password", "correct-horse-battery-staple"),
        ("client_secret", "secret-value-abcdefghijklmnopqrstuvwxyz"),
    ],
)
def test_redacts_sensitive_named_assignments(name, secret):
    cleaned, labels = redact_text(f'{name} = "{secret}"')

    assert name in cleaned
    assert "[REDACTED]" in cleaned
    assert secret not in repr((cleaned, labels))
    assert labels == ["sensitive_assignment"]


def test_redacts_entire_unquoted_sensitive_value_with_punctuation():
    secret = "correct#horse;battery-staple"

    cleaned, labels = redact_text(f"password={secret}")

    assert cleaned == "password=[REDACTED]"
    assert secret not in repr((cleaned, labels))
    assert labels == ["sensitive_assignment"]


def test_redacts_nested_expression_rhs_without_leaking_secret_fragments():
    text = 'password=get_secret("alpha,beta")'

    cleaned, labels = redact_text(text)

    assert cleaned == "password=[REDACTED]"
    assert all(
        fragment not in repr((cleaned, labels)) for fragment in ("alpha", "beta")
    )
    assert labels == ["sensitive_assignment"]


def test_summarize_text_redacts_nested_expression_rhs_without_leaking_fragments():
    text = 'password=get_secret("alpha,beta")'

    result = summarize_text(text)

    assert result["text"] == "password=[REDACTED]"
    assert all(fragment not in repr(result) for fragment in ("alpha", "beta"))


@pytest.mark.parametrize(
    ("text", "expected", "secret_fragments"),
    [
        (
            "password=correct horse battery staple",
            "password=[REDACTED]",
            ("correct", "horse", "battery", "staple"),
        ),
        (
            "password=[secret_value]",
            "password=[REDACTED]",
            ("secret_value",),
        ),
        (
            'password=["alpha", "beta"]\nvisible = "safe"',
            'password=[REDACTED]\nvisible = "safe"',
            ("alpha", "beta"),
        ),
    ],
)
def test_redacts_complete_equals_rhs_without_crossing_lines(
    text, expected, secret_fragments
):
    cleaned, labels = redact_text(text)

    assert cleaned == expected
    assert all(fragment not in repr((cleaned, labels)) for fragment in secret_fragments)
    assert labels == ["sensitive_assignment"]


@pytest.mark.parametrize(
    ("text", "expected", "secret_fragments"),
    [
        (
            "password=correct horse battery staple",
            "password=[REDACTED]",
            ("correct", "horse", "battery", "staple"),
        ),
        (
            "password=[secret_value]",
            "password=[REDACTED]",
            ("secret_value",),
        ),
        (
            'password=["alpha", "beta"]\nvisible = "safe"',
            'password=[REDACTED]\nvisible = "safe"',
            ("alpha", "beta"),
        ),
    ],
)
def test_summarize_text_redacts_complete_equals_rhs(
    text, expected, secret_fragments
):
    result = summarize_text(text)

    assert result["text"] == expected
    assert all(fragment not in repr(result) for fragment in secret_fragments)


@pytest.mark.parametrize(
    ("text", "expected", "secret_fragments"),
    [
        (
            'password = [\n    "alpha",\n    "beta",\n]\nvisible = "safe"',
            'password = [REDACTED]\nvisible = "safe"',
            ("alpha", "beta"),
        ),
        (
            'password = """correct\nhorse battery staple"""\nvisible = "safe"',
            'password = """[REDACTED]"""\nvisible = "safe"',
            ("correct", "horse", "battery", "staple"),
        ),
        (
            "password: |\n  correct\n  horse battery staple\nvisible: safe",
            "password: [REDACTED]\nvisible: safe",
            ("correct", "horse", "battery", "staple"),
        ),
    ],
)
def test_redacts_multiline_rhs_without_consuming_following_safe_line(
    text, expected, secret_fragments
):
    cleaned, labels = redact_text(text)
    summary = summarize_text(text)

    assert cleaned == expected
    assert summary["text"] == expected
    assert all(fragment not in repr((cleaned, labels)) for fragment in secret_fragments)
    assert all(fragment not in repr(summary) for fragment in secret_fragments)
    assert labels == ["sensitive_assignment"]


@pytest.mark.parametrize(
    ("text", "expected", "secret"),
    [
        (
            "call(password=secret_value)",
            "call(password=[REDACTED])",
            "secret_value",
        ),
        (
            'password=secret; print("safe")',
            'password=[REDACTED]; print("safe")',
            "secret",
        ),
        (
            "call(password=secret_value, retries=3)",
            "call(password=[REDACTED], retries=3)",
            "secret_value",
        ),
    ],
)
def test_redacts_equals_rhs_and_preserves_following_structure(text, expected, secret):
    cleaned, labels = redact_text(text)

    assert cleaned == expected
    assert secret not in repr((cleaned, labels))
    assert labels == ["sensitive_assignment"]


def test_redacts_rhs_before_no_space_semicolon_statement():
    text = 'password=secret;print("safe")'

    cleaned, labels = redact_text(text)

    assert cleaned == 'password=[REDACTED];print("safe")'
    assert "secret" not in repr((cleaned, labels))
    assert labels == ["sensitive_assignment"]


def test_redacts_python_annotated_sensitive_assignment_rhs():
    secret = "synthetic-secret-value"

    cleaned, labels = redact_text(f'api_key: str = "{secret}"')

    assert cleaned == 'api_key: str = "[REDACTED]"'
    assert secret not in repr((cleaned, labels))
    assert labels == ["sensitive_assignment"]


def test_redacts_python_annotated_assignment_without_spaces_around_equals():
    secret = "synthetic-secret-value"

    cleaned, labels = redact_text(f'api_key: str="{secret}"')

    assert cleaned == 'api_key: str="[REDACTED]"'
    assert secret not in repr((cleaned, labels))
    assert labels == ["sensitive_assignment"]


@pytest.mark.parametrize(
    "text",
    [
        "api_key: str",
        "api_key: Optional[str]",
        "def connect(api_key: str): ...",
    ],
)
def test_does_not_redact_python_type_annotations_without_assignments(text):
    assert redact_text(text) == (text, [])
    assert summarize_text(text)["text"] == text


@pytest.mark.parametrize(
    "text",
    [
        "def connect(api_key: pydantic.SecretStr): ...",
        "api_key: type[str]",
        'api_key: "CredentialType"',
    ],
)
def test_does_not_redact_general_python_annotations_without_assignments(text):
    assert redact_text(text) == (text, [])
    assert summarize_text(text)["text"] == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            'api_key: pydantic.SecretStr = "synthetic-secret-value"',
            'api_key: pydantic.SecretStr = "[REDACTED]"',
        ),
        (
            'api_key: "CredentialType" = "synthetic-secret-value"',
            'api_key: "CredentialType" = "[REDACTED]"',
        ),
    ],
)
def test_redacts_assignments_with_general_python_annotations(text, expected):
    secret = "synthetic-secret-value"

    cleaned, labels = redact_text(text)

    assert cleaned == expected
    assert secret not in repr((cleaned, labels))
    assert labels == ["sensitive_assignment"]


def test_redacts_entire_yaml_sensitive_assignment_value():
    text = "password: correct horse battery staple"

    cleaned, labels = redact_text(text)

    assert cleaned == "password: [REDACTED]"
    assert all(word not in cleaned for word in ("correct", "horse", "battery", "staple"))
    assert labels == ["sensitive_assignment"]


@pytest.mark.parametrize(
    ("prefix", "suffix"),
    [
        ("{", "}"),
        ("[{", "}]"),
    ],
)
def test_redacts_complete_json_secret_and_preserves_closing_structure(prefix, suffix):
    secret = "correct,horse,battery,staple"
    text = f'{prefix}"password": "{secret}"{suffix}'

    cleaned, labels = redact_text(text)

    assert cleaned == f'{prefix}"password": "[REDACTED]"{suffix}'
    assert secret not in repr((cleaned, labels))
    assert labels == ["sensitive_assignment"]


@pytest.mark.parametrize(
    "target",
    [
        'config["api_key"]',
        "config['api_key']",
        'os.environ["SERVICE_TOKEN"]',
        "os.environ['SERVICE_TOKEN']",
    ],
)
def test_redacts_sensitive_string_key_subscript_assignment(target):
    secret = "synthetic-secret-value"

    cleaned, labels = redact_text(f'{target} = "{secret}"')

    assert cleaned == f'{target} = "[REDACTED]"'
    assert secret not in repr((cleaned, labels))
    assert labels == ["sensitive_assignment"]


def test_does_not_redact_ordinary_short_numeric_token_metadata():
    text = "token_count = 32"

    assert redact_text(text) == (text, [])


@pytest.mark.parametrize(
    "text",
    [
        'tokenizer = AutoTokenizer.from_pretrained("gpt2")',
        "max_tokens = 128",
        "token_ids = [1, 2, 3]",
        'secretary = "Alice"',
    ],
)
def test_does_not_redact_common_non_secret_llm_variables(text):
    assert redact_text(text) == (text, [])
    assert summarize_text(text)["text"] == text


def test_summarize_text_redacts_before_truncating_and_returns_only_safe_fields():
    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
    text = f"OPENAI_API_KEY={secret}\n" + ("x" * 200)

    result = summarize_text(text, max_chars=80)

    assert set(result) == {"text", "original_chars", "sha256", "truncated"}
    assert result["text"].startswith("OPENAI_API_KEY=[REDACTED]")
    assert len(result["text"]) <= 80
    assert result["original_chars"] == len(text)
    assert result["truncated"] is True
    assert secret not in repr(result)


def test_summarize_text_redacts_common_assignment_forms_without_crossing_lines():
    secret = "synthetic-secret-value"
    text = "\n".join(
        [
            f'api_key: str = "{secret}"',
            "password: correct horse battery staple",
            f'config["api_key"] = "{secret}"',
            f'os.environ["SERVICE_TOKEN"] = "{secret}"',
        ]
    )

    result = summarize_text(text, max_chars=1000)

    assert result["text"] == "\n".join(
        [
            'api_key: str = "[REDACTED]"',
            "password: [REDACTED]",
            'config["api_key"] = "[REDACTED]"',
            'os.environ["SERVICE_TOKEN"] = "[REDACTED]"',
        ]
    )
    assert secret not in repr(result)
    assert all(
        word not in result["text"]
        for word in ("correct", "horse", "battery", "staple")
    )


def test_summarize_text_redacts_complete_json_secret_without_leaking_fragments():
    secret = "correct,horse,battery,staple"
    text = f'{{"password": "{secret}"}}'

    result = summarize_text(text)

    assert result["text"] == '{"password": "[REDACTED]"}'
    assert secret not in repr(result)
    assert all(part not in result["text"] for part in secret.split(","))


def test_summarize_text_hashes_original_text_stably():
    original = "stable input with unicode: 台灣"

    first = summarize_text(original)
    second = summarize_text(original)
    expected = hashlib.sha256(original.encode("utf-8")).hexdigest()

    assert first["sha256"] == expected
    assert re.fullmatch(r"[0-9a-f]{64}", first["sha256"])
    assert second["sha256"] == first["sha256"]


def test_summarize_text_marks_short_text_as_not_truncated():
    text = "short text"

    result = summarize_text(text, max_chars=len(text) + 10)

    assert result["text"] == text
    assert result["truncated"] is False


def test_summarize_text_marks_exact_boundary_as_not_truncated():
    text = "exact boundary"

    result = summarize_text(text, max_chars=len(text))

    assert result["text"] == text
    assert result["truncated"] is False


def test_summarize_text_redacts_secret_before_cross_boundary_truncation():
    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
    text = f"value={secret}\nsafe suffix"

    result = summarize_text(text, max_chars=18)

    assert result["text"].startswith("value=[REDACTED]")
    assert "sk-proj" not in result["text"]
    assert secret[:12] not in result["text"]
    assert len(result["text"]) <= 18
    assert result["truncated"] is True


def test_summarize_text_allows_zero_character_limit():
    result = summarize_text("visible", max_chars=0)

    assert result["text"] == ""
    assert result["truncated"] is True


def test_summarize_text_rejects_negative_character_limit():
    with pytest.raises(ValueError, match="max_chars"):
        summarize_text("visible", max_chars=-1)
