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


def test_redacts_python_annotated_sensitive_assignment_rhs():
    secret = "synthetic-secret-value"

    cleaned, labels = redact_text(f'api_key: str = "{secret}"')

    assert cleaned == 'api_key: str = "[REDACTED]"'
    assert secret not in repr((cleaned, labels))
    assert labels == ["sensitive_assignment"]


def test_redacts_entire_yaml_sensitive_assignment_value():
    text = "password: correct horse battery staple"

    cleaned, labels = redact_text(text)

    assert cleaned == "password: [REDACTED]"
    assert all(word not in cleaned for word in ("correct", "horse", "battery", "staple"))
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
